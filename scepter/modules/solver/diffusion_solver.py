# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.cuda.amp as amp
from torch.distributed.fsdp import (BackwardPrefetch, CPUOffload,
                                    FullStateDictConfig,
                                    FullyShardedDataParallel, MixedPrecision,
                                    ShardingStrategy, StateDictType)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from scepter.modules.data.dataset import DATASETS
from scepter.modules.opt.lr_schedulers import LR_SCHEDULERS
from scepter.modules.opt.optimizers import OPTIMIZERS
from scepter.modules.solver import BaseSolver
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config, dict_to_yaml
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.distribute import we
from scepter.modules.utils.probe import ProbeData

sharding_strategy_map = {
    'full_shard': ShardingStrategy.FULL_SHARD,
    'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP
}


@SOLVERS.register_class()
class LatentDiffusionSolver(BaseSolver):
    para_dict = {
        'MAX_STEPS': {
            'value': 100000,
            'description': 'The total steps for training.',
        },
        'USE_AMP': {
            'value':
            False,
            'description':
            'Use amp to surpport mix precision or not, default is False.',
        },
        'DTYPE': {
            'value': 'float32',
            'description': 'The precision for training.',
        },
        'USE_FAIRSCALE': {
            'value': False,
            'description':
            'Use fairscale as the backend of ddp, default False.',
        },
        'USE_FSDP': {
            'value': False,
            'description': 'Use fsdp as the backend of ddp, default False.',
        },
        'SHARDING_STRATEGY': {
            'value':
            'shard_grad_op',
            'description':
            f'The shard strategy for fsdp, select from {list(sharding_strategy_map.keys())}',
        },
        'IMAGE_LOG_STEP': {
            'value': 2000,
            'description': 'The interval for image log.',
        },
        'LOAD_MODEL_ONLY': {
            'value':
            False,
            'description':
            'Only load the model rather than the optimizer and schedule, default is False.',
        },
        'CHANNELS_LAST': {
            'value': False,
            'description': 'The channels last, default is False.',
        },
        'SAMPLE_ARGS': {
            'value':
            None,
            'description':
            'Sampling related parameters, default is None( use default sample args ).',
        },
        'TUNER': {
            'value': None,
            'description': 'Tuner config, default is None.',
        },
        'FREEZE': {
            'value':
            None,
            'description':
            'Specify freezing and training parameters, default is None.',
        }
    }
    para_dict.update(BaseSolver.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.max_steps = cfg.MAX_STEPS
        if self.max_steps > 0:
            self.max_epochs = -1
        self.use_amp = cfg.get('USE_AMP', False)
        self.dtype = getattr(torch, cfg.DTYPE)
        self.use_fairscale = cfg.get('USE_FAIRSCALE', False)
        self.use_fsdp = cfg.get('USE_FSDP', False)
        if self.use_fairscale and self.use_fsdp:
            raise 'fairscale and fsdp is not allowed used meanwhile.'
        elif self.use_fairscale:
            self.logger.info('Use fairscale as the backend of ddp.')
        elif self.use_fsdp:
            self.logger.info('Use fsdp as the backend of ddp.')
        else:
            self.logger.info('Use default backend.')
        self.model_shard = cfg.get('SHARDING_STRATEGY', 'full_shard')
        self.image_log_step = cfg.get('IMAGE_LOG_STEP', 2000)
        self._image_out = defaultdict(list)
        self.load_model_only = cfg.get('LOAD_MODEL_ONLY', False)
        self.channels_last = cfg.get('CHANNELS_LAST', False)
        self.current_batch_data = defaultdict(dict)
        self.sample_args = cfg.get('SAMPLE_ARGS', None)
        self.tuner_cfg = cfg.get('TUNER', None)
        self.freeze_cfg = cfg.get('FREEZE', None)

    def set_up(self):
        self.construct_data()
        self.construct_model()
        self.construct_metrics()
        self.model_to_device()
        if 'train' in self.datas and self.cfg.have('OPTIMIZER'):
            if we.world_size > 1:
                all_batch_size = self.datas['train'].batch_size * we.world_size
            else:
                all_batch_size = self.datas['train'].batch_size
            self.cfg.OPTIMIZER.LEARNING_RATE *= all_batch_size
            self.cfg.OPTIMIZER.LEARNING_RATE /= 640
        self.init_opti()

    def construct_hook(self):
        # initialize data
        assert self.cfg.have('TRAIN_HOOKS') or self.cfg.have(
            'EVAL_HOOKS') or self.cfg.have('TEST_HOOKS')
        if self.cfg.have('TRAIN_HOOKS'):
            self.hooks_dict['train'] = self._load_hook(self.cfg.TRAIN_HOOKS)
        if self.cfg.have('EVAL_HOOKS'):
            self.hooks_dict['eval'] = self._load_hook(self.cfg.EVAL_HOOKS)
        if self.cfg.have('TEST_HOOKS'):
            self.hooks_dict['test'] = self._load_hook(self.cfg.TEST_HOOKS)

    def construct_data(self):
        # assert self.cfg.have("TRAIN_DATA") or self.cfg.have("EVAL_DATA") or self.cfg.have("TEST_DATA")
        if self.cfg.have('TRAIN_DATA'):
            train_data = DATASETS.build(self.cfg.TRAIN_DATA,
                                        logger=self.logger)
            self.datas['train'] = train_data
            self._epoch_max_iter['train'] = len(train_data.dataloader)
            self._mode_set.add('train')
        if self.cfg.have('EVAL_DATA') and 'train' in self._mode_set:
            eval_data = DATASETS.build(self.cfg.EVAL_DATA, logger=self.logger)
            self.datas['eval'] = eval_data
            self._epoch_max_iter['eval'] = len(eval_data.dataloader)
            self._mode_set.add('eval')
        if self.cfg.have('TEST_DATA'):
            test_data = DATASETS.build(self.cfg.TEST_DATA, logger=self.logger)
            self.datas['test'] = test_data
            self._epoch_max_iter['test'] = len(test_data.dataloader)
            self._mode_set.add('test')

    def construct_model(self):
        super().construct_model()
        if self.tuner_cfg:
            self.model = self.add_tuner(self.tuner_cfg, self.model)
        if self.freeze_cfg:
            freeze_cfg = Config.get_plain_cfg(self.freeze_cfg)
            self.model = self.freeze(freeze_cfg, self.model)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.print_model_params_status()

        if we.debug:
            module_keys = [key for key, _ in self.model.named_modules()]
            self.logger.info(module_keys)

    def model_to_device(self):
        self.model = self.model.to(we.device_id)

    def init_opti(self):
        if hasattr(self.model, 'ignored_parameters'):
            train_params, ignored_params = self.model.parameters(
            ), self.model.ignored_parameters()
        else:
            train_params, ignored_params = self.model.parameters(), None
        if we.is_distributed:
            if self.use_fairscale:
                from fairscale.nn.data_parallel import ShardedDataParallel
                from fairscale.optim.oss import OSS
                self.optimizer = OSS(params=train_params,
                                     optim=torch.optim.AdamW,
                                     lr=self.cfg.OPTIMIZER.LEARNING_RATE)
                self.model = ShardedDataParallel(self.model, self.optimizer)
            elif self.use_fsdp:
                mixed_precision = MixedPrecision(param_dtype=self.dtype,
                                                 reduce_dtype=self.dtype,
                                                 buffer_dtype=self.dtype)
                sharding_strategy = sharding_strategy_map[self.model_shard]
                self.model = FullyShardedDataParallel(
                    self.model,
                    mixed_precision=mixed_precision,
                    cpu_offload=CPUOffload(offload_params=False),
                    sharding_strategy=sharding_strategy,
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    device_id=torch.cuda.current_device(),
                    ignored_parameters=ignored_params)
                self.optimizer = OPTIMIZERS.build(self.cfg.OPTIMIZER,
                                                  logger=self.logger,
                                                  parameters=train_params)
            else:
                assert not self.model.use_ema
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[torch.cuda.current_device()],
                    output_device=torch.cuda.current_device(),
                    find_unused_parameters=True)
                self.optimizer = OPTIMIZERS.build(self.cfg.OPTIMIZER,
                                                  logger=self.logger,
                                                  parameters=train_params)
        else:
            self.optimizer = OPTIMIZERS.build(self.cfg.OPTIMIZER,
                                              logger=self.logger,
                                              parameters=train_params)

        if self.cfg.have('LR_SCHEDULER') and self.optimizer is not None:
            self.cfg.LR_SCHEDULER.TOTAL_STEPS = self.max_steps
            self.lr_scheduler = LR_SCHEDULERS.build(self.cfg.LR_SCHEDULER,
                                                    logger=self.logger,
                                                    optimizer=self.optimizer)

        if self.cfg.DTYPE == 'float16':
            if we.is_distributed:
                if self.use_fairscale:
                    from fairscale.optim.grad_scaler import ShardedGradScaler
                    self.scaler = ShardedGradScaler(enabled=True)
                elif self.use_fsdp:
                    from torch.distributed.fsdp.sharded_grad_scaler import \
                        ShardedGradScaler
                    self.scaler = ShardedGradScaler()
                else:
                    self.scaler = amp.GradScaler()
            else:
                self.scaler = amp.GradScaler()
        else:
            self.scaler = None

    def load_checkpoint(self, checkpoint: dict):
        """
        Load checkpoint function
        :param checkpoint: all tensors are on cpu, you need to transfer to gpu by hand
        :return:
        """
        if 'model' in checkpoint:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint['model'])
        else:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
        if not self.load_model_only:
            if 'optimizer' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler'])

    def save_checkpoint(self) -> dict:
        """
        Save checkpoint function, you need to transfer all tensors to cpu by hand
        :return:
        """
        ckpt = dict()
        if we.is_distributed:
            if self.use_fsdp:
                save_policy = FullStateDictConfig(offload_to_cpu=True,
                                                  rank0_only=True)
                with FullyShardedDataParallel.state_dict_type(
                        self.model, StateDictType.FULL_STATE_DICT,
                        save_policy):
                    ckpt['model'] = self.model.state_dict()
            else:
                if hasattr(self.model, 'module'):
                    ckpt['model'] = self.model.module.state_dict()
                else:
                    ckpt['model'] = self.model.state_dict()
        else:
            ckpt['model'] = self.model.state_dict()
        if self.optimizer and not self.use_fairscale:
            ckpt['optimizer'] = self.optimizer.state_dict()
        if self.scaler:
            ckpt['scaler'] = self.scaler.state_dict()
        return ckpt

    def save_pretrained(self):
        if hasattr(self.model, 'save_pretrained'):
            ckpt = self.model.save_pretrained()
        elif hasattr(self.model, 'module') and hasattr(self.model.module,
                                                       'save_pretrained'):
            ckpt = self.model.module.save_pretrained()
        else:
            ckpt = dict()
        if hasattr(self.model, 'save_pretrained_config'):
            cfg = self.model.save_pretrained_config()
        elif hasattr(self.model, 'module') and hasattr(
                self.model.module, 'save_pretrained_config'):
            cfg = self.model.module.save_pretrained_config()
        else:
            cfg = copy.deepcopy(self.cfg.MODEL.cfg_dict)
        if 'FILE_SYSTEM' in cfg:
            cfg.pop('FILE_SYSTEM')
        return ckpt, cfg

    def solve(self):
        self.before_solve()
        if 'train' in self._mode_set:
            self.run_train()
        if 'test' in self._mode_set:
            self.run_test()
        self.after_solve()

    def run_train(self):
        self.train_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        data_iter = iter(self.datas[self._mode].dataloader)
        self.print_memory_status()
        for step in range(self.max_steps):
            if 'eval' in self._mode_set and (self.eval_interval > 0 and
                                             step % self.eval_interval == 0):
                self.run_eval()
                self.train_mode()
            batch_data = next(data_iter)
            self.before_iter(self.hooks_dict[self._mode])
            if self.sample_args:
                batch_data.update(self.sample_args.get_lowercase_dict())
            if 'meta' in batch_data:
                self.register_probe({
                    'data_key':
                    ProbeData(batch_data['meta'].get('data_key', []),
                              view_distribute=True)
                })
            self.register_probe({
                'prompt': batch_data['prompt'],
                'batch_size': len(batch_data['prompt'])
            })
            self.current_batch_data[self.mode] = batch_data
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_train(
                    transfer_data_to_cuda(batch_data),
                    step,
                    step=self.total_iter,
                    rank=we.rank)
                self._iter_outputs[self._mode] = self._reduce_scalar(results)
            self.after_iter(self.hooks_dict[self._mode])
            if we.debug:
                self.print_trainable_params_status(prefix='model.')
            if 'eval' in self._mode_set and (self.eval_interval > 0
                                             and step == self.max_steps - 1):
                self.run_eval()
        self.after_all_iter(self.hooks_dict[self._mode])

    @torch.no_grad()
    def run_eval(self):
        self.eval_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        all_results = []
        for batch_idx, batch_data in tqdm(
                enumerate(self.datas[self._mode].dataloader)):
            self.before_iter(self.hooks_dict[self._mode])
            if self.sample_args:
                batch_data.update(self.sample_args.get_lowercase_dict())
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_eval(transfer_data_to_cuda(batch_data),
                                             batch_idx,
                                             step=self.total_iter,
                                             rank=we.rank)
                all_results.extend(results)
            self.after_iter(self.hooks_dict[self._mode])
        log_data, log_label, ori_label = [], [], []
        for result in all_results:
            # the inference image use
            if 'hint' in result:
                merge_image = torch.cat([
                    result['hint'][:result['image'].shape[0]], result['image']
                ],
                                        dim=2)
                log_data.append((merge_image.permute(1, 2, 0).cpu().numpy() *
                                 255).astype(np.uint8))
            else:
                log_data.append(
                    (result['image'].permute(1, 2, 0).cpu().numpy() *
                     255).astype(np.uint8))
            log_label.append(result['prompt'] + ' NegPrompt: ' +
                             result['n_prompt'])
            ori_label.append(result['prompt'])

        self.register_probe({'test_label': log_label})
        self.register_probe({
            'test_image':
            ProbeData(log_data,
                      is_image=True,
                      build_html=True,
                      build_label=log_label)
        })

        log_data, log_label, ori_label = [], [], []
        for result in all_results:
            # the inference image use
            if 'train_n_image' in result:
                if 'hint' in result:
                    merge_image = torch.cat([
                        result['hint'][:result['train_n_image'].shape[0]],
                        result['train_n_image']
                    ],
                                            dim=2)
                    log_data.append(
                        (merge_image.permute(1, 2, 0).cpu().numpy() *
                         255).astype(np.uint8))
                else:
                    log_data.append((result['train_n_image'].permute(
                        1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                log_label.append(result['prompt'] + 'NegPrompt' +
                                 result['train_n_prompt'])
                ori_label.append(result['prompt'])
        if len(log_data) > 0:
            self.register_probe({'test_train_n_label': log_label})
            self.register_probe({
                'test_train_n_image':
                ProbeData(log_data,
                          is_image=True,
                          build_html=True,
                          build_label=log_label)
            })
        self.after_all_iter(self.hooks_dict[self._mode])

    @torch.no_grad()
    def run_test(self):
        self.test_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        all_results = []
        for batch_idx, batch_data in tqdm(
                enumerate(self.datas[self._mode].dataloader)):
            self.before_iter(self.hooks_dict[self._mode])
            if self.sample_args:
                batch_data.update(self.sample_args.get_lowercase_dict())
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_eval(transfer_data_to_cuda(batch_data),
                                             batch_idx,
                                             step=self.total_iter,
                                             rank=we.rank)
                all_results.extend(results)
            self.after_iter(self.hooks_dict[self._mode])
        log_data, log_label = [], []
        for result in all_results:
            # the inference image use
            log_data.append((result['image'].permute(1, 2, 0).cpu().numpy() *
                             255).astype(np.uint8))
            log_label.append(result['prompt'] +
                             " <font color='red'> |NegPrompt| </font> " +
                             result['n_prompt'])

        self.register_probe({'test_label': log_label})
        self.register_probe({
            'test_image':
            ProbeData(log_data,
                      is_image=True,
                      build_html=True,
                      build_label=log_label)
        })

        log_data, log_label = [], []
        for result in all_results:
            # the inference image use
            if 'train_n_image' in result:
                log_data.append(
                    (result['train_n_image'].permute(1, 2, 0).cpu().numpy() *
                     255).astype(np.uint8))
                log_label.append(result['prompt'] +
                                 " <font color='red'> |NegPrompt| </font> " +
                                 result['train_n_prompt'])
        if len(log_data) > 0:
            self.register_probe({'test_train_n_label': log_label})
            self.register_probe({
                'test_train_n_image':
                ProbeData(log_data,
                          is_image=True,
                          build_html=True,
                          build_label=log_label)
            })

        self.after_all_iter(self.hooks_dict[self._mode])

    def add_tuner(self, tuner_cfg, model=None):
        from scepter.modules.model.registry import TUNERS

        if model is None:
            model = self.model
        swift_cfg_dict = {}
        for t_id, t_cfg in enumerate(tuner_cfg):
            cfg_name = t_cfg['NAME']
            init_config = TUNERS.build(t_cfg, logger=self.logger)()
            if init_config is None:
                continue
            swift_cfg_dict[f'{t_id}_{cfg_name}'] = init_config
        if len(swift_cfg_dict) > 0:
            from swift import Swift
            model = Swift.prepare_model(self.model, config=swift_cfg_dict)
        return model

    def freeze(self, freeze_cfg, model=None):
        """ Freeze or train the model based on the config.
        """
        if model is None:
            model = self.model
        freeze_part = freeze_cfg[
            'FREEZE_PART'] if 'FREEZE_PART' in freeze_cfg else []
        train_part = freeze_cfg[
            'TRAIN_PART'] if 'TRAIN_PART' in freeze_cfg else []

        if hasattr(model, 'module'):
            freeze_model = model.module
        else:
            freeze_model = model

        if freeze_part:
            if isinstance(freeze_part, dict):
                if 'BACKBONE' in freeze_part:
                    part = freeze_part['BACKBONE']
                    for name, param in freeze_model.backbone.named_parameters(
                    ):
                        freeze_flag = sum([p in name for p in part]) > 0
                        if freeze_flag:
                            param.requires_grad = False
                elif 'HEAD' in freeze_part:
                    part = freeze_part['HEAD']
                    for name, param in freeze_model.head.named_parameters():
                        freeze_flag = sum([p in name for p in part]) > 0
                        if freeze_flag:
                            param.requires_grad = False
            elif isinstance(freeze_part, list):
                for name, param in freeze_model.named_parameters():
                    freeze_flag = sum([p in name for p in freeze_part]) > 0
                    if freeze_flag:
                        param.requires_grad = False
        if train_part:
            if isinstance(train_part, dict):
                if 'BACKBONE' in train_part:
                    part = train_part['BACKBONE']
                    for name, param in freeze_model.backbone.named_parameters(
                    ):
                        freeze_flag = sum([p in name for p in part]) > 0
                        if freeze_flag:
                            param.requires_grad = True
                elif 'HEAD' in train_part:
                    part = train_part['HEAD']
                    for name, param in freeze_model.head.named_parameters():
                        freeze_flag = sum([p in name for p in part]) > 0
                        if freeze_flag:
                            param.requires_grad = True
            elif isinstance(train_part, list):
                for name, param in freeze_model.named_parameters():
                    freeze_flag = sum([p in name for p in train_part]) > 0
                    if freeze_flag:
                        param.requires_grad = True
        return model

    @torch.no_grad()
    def log_image(self, batch_data):
        self.eval_mode()
        if we.is_distributed:
            if hasattr(self.model, 'module'):
                images = self.model.module.log_images(**batch_data)
            else:
                images = self.model.log_images(**batch_data)
        else:
            images = self.model.log_images(**batch_data)
        self.train_mode()
        return images

    @staticmethod
    def get_config_template():
        return dict_to_yaml('solvername',
                            __class__.__name__,
                            LatentDiffusionSolver.para_dict,
                            set_name=True)

    @property
    def image_out(self):
        return self._image_out[self._mode]

    def collect_log_vars(self) -> OrderedDict:
        ret = OrderedDict()
        if self.is_train_mode and self.optimizer is not None:
            for idx, pg in enumerate(self.optimizer.param_groups):
                ret[f'pg{idx}_lr'] = pg['lr']
        if self.is_train_mode and self.scaler is not None:
            ret['scale'] = self.scaler.get_scale()
        return ret

    @property
    def probe_data(self):
        if not we.debug and self.mode == 'train':
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                outputs = self.log_image(
                    transfer_data_to_cuda(self.current_batch_data[self.mode]))
            log_data, log_label = [], []
            for result in outputs:
                if 'hint' in result:
                    merge_image = torch.cat([
                        result['orig'],
                        result['hint'][:result['orig'].shape[0]],
                        result['recon']
                    ],
                                            dim=2)
                else:
                    merge_image = torch.cat([result['orig'], result['recon']],
                                            dim=2)
                log_data.append((merge_image.permute(1, 2, 0).cpu().numpy() *
                                 255).astype(np.uint8))
                log_label.append('recon image: ' + result['prompt'] +
                                 " <font color='red'> |NegPrompt| </font> " +
                                 result['n_prompt'])

            self.register_probe({
                'train_image':
                ProbeData(log_data,
                          is_image=True,
                          build_html=True,
                          build_label=log_label)
            })
            self.register_probe({'train_label': log_label})

            # the inference image use
            log_data, log_label = [], []
            for result in outputs:
                if 'train_n_image' in result:
                    if 'hint' in result:
                        merge_image = torch.cat([
                            result['orig'],
                            result['hint'][:result['orig'].shape[0]],
                            result['train_n_image']
                        ],
                                                dim=2)
                    else:
                        merge_image = torch.cat(
                            [result['orig'], result['train_n_image']], dim=2)
                    log_data.append(
                        (merge_image.permute(1, 2, 0).cpu().numpy() *
                         255).astype(np.uint8))
                    log_label.append(
                        'recon image: ' + result['prompt'] +
                        " <font color='red'> |NegPrompt| </font> " +
                        result['train_n_prompt'])

            if len(log_data) > 0:
                self.register_probe({'train_n_label': log_label})
                self.register_probe({
                    'train_n_image':
                    ProbeData(log_data,
                              is_image=True,
                              build_html=True,
                              build_label=log_label)
                })
        return super().probe_data

    def print_memory_status(self):
        """Print the memory usage status of the model"""
        if torch.cuda.is_available():
            nvi_info = os.popen('nvidia-smi').read()
            gpu_mem = nvi_info.split('\n')[9].split('|')[2].split(
                '/')[0].strip()
        else:
            gpu_mem = ''
        return gpu_mem

    def print_trainable_params_status(self,
                                      model=None,
                                      logger=None,
                                      prefix=''):
        """Print the status and parameters of the model"""
        if model is None:
            model = self.model
        if logger is None:
            logger = self.logger

        for key, val in model.named_parameters():
            if val.requires_grad:
                if prefix in key:
                    logger.info(
                        f"param {key} value'sum {torch.sum(val)} with shape {val.shape}."
                    )

    def print_model_params_status(self, model=None, logger=None):
        """Print the status and parameters of the model"""
        if model is None:
            model = self.model
        if logger is None:
            logger = self.logger
        train_param_dict = {}
        forzen_param_dict = {}
        all_param_numel = 0
        if we.debug:
            for key, _ in model.named_modules():
                logger.info(f'sub modules {key}.')
        for key, val in model.named_parameters():
            if val.requires_grad:
                sub_key = '.'.join(key.split('.', 1)[-1].split('.', 2)[:2])
                if sub_key in train_param_dict:
                    train_param_dict[sub_key] += val.numel()
                else:
                    train_param_dict[sub_key] = val.numel()
            else:
                sub_key = '.'.join(key.split('.', 1)[-1].split('.', 1)[:1])
                if sub_key in forzen_param_dict:
                    forzen_param_dict[sub_key] += val.numel()
                else:
                    forzen_param_dict[sub_key] = val.numel()
            all_param_numel += val.numel()
            if we.debug:
                logger.info(key)
        train_param_numel = sum(train_param_dict.values())
        forzen_param_numel = sum(forzen_param_dict.values())
        logger.info(
            f'Load trainable params {train_param_numel} / {all_param_numel} = '
            f'{train_param_numel / all_param_numel:.2%}, '
            f'train part: {train_param_dict}.')
        logger.info(
            f'Load forzen params {forzen_param_numel} / {all_param_numel} = '
            f'{forzen_param_numel / all_param_numel:.2%}, '
            f'forzen part: {forzen_param_dict}.')
