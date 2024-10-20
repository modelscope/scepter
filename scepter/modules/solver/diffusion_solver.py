# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
import re
import warnings
from collections import OrderedDict, defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from scepter.modules.data.dataset import DATASETS
from scepter.modules.opt.lr_schedulers import LR_SCHEDULERS
from scepter.modules.opt.optimizers import OPTIMIZERS
from scepter.modules.solver import BaseSolver
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config, dict_to_yaml
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.distribute import we
from scepter.modules.utils.probe import ProbeData
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)
from torch.distributed.fsdp.wrap import (lambda_auto_wrap_policy,
                                         size_based_auto_wrap_policy)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

sharding_strategy_map = {
    'full_shard': ShardingStrategy.FULL_SHARD,
    'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
    'hybrid_shard': ShardingStrategy.HYBRID_SHARD
}


def shard_model(model,
                device_id,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
                fsdp_group = ['blocks'],
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                sync_module_states=False):
    wrap_modules = []
    for module_name in fsdp_group:
        if hasattr(model, module_name):
            if isinstance(getattr(model, module_name), (list, tuple, nn.ModuleList)):
                wrap_modules.extend([m for m in getattr(model, module_name)])
            else:
                wrap_modules.extend([getattr(model, module_name)])
        else:
            warnings.warn("Can't find module {} in model".format(module_name))
    return FSDP(
        module=model,
        process_group=None,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            # size_based_auto_wrap_policy, min_num_params=int(1e6),
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in wrap_modules),
        mixed_precision=MixedPrecision(param_dtype=param_dtype,
                                       reduce_dtype=reduce_dtype,
                                       buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)


def get_module(instance, sub_module):
    sub_module_list = sub_module.split('.')
    for sub_mod in sub_module_list:
        if sub_mod == '':
            continue
        if hasattr(instance, sub_mod):
            instance = getattr(instance, sub_mod)
        else:
            return None
    return instance


def set_module(instance, sub_module, value):
    sub_module_list = sub_module.split('.')
    instance_list = []
    for sub_mod in sub_module_list:
        if hasattr(instance, sub_mod):
            instance_list.append((instance, sub_mod))
            instance = getattr(instance, sub_mod)
    instance = value
    if len(instance_list) > 0:
        for parents_instance, sub_mod in instance_list[::-1]:
            setattr(parents_instance, sub_mod, instance)
            instance = parents_instance


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
        'FSDP_REDUCE_DTYPE': {
            'value': 'float32',
            'description': 'The dtype of reduce in FSDP.'
        },
        'FSDP_BUFFER_DTYPE': {
            'value': 'float32',
            'description': 'The dtype of buffer in FSDP.'
        },
        'FSDP_SHARD_MODULES': {
            'value': ['model'],
            'description': 'The modules to be sharded in FSDP.'
        },
        'SAVE_MODULES': {
            'value':
            None,
            'description':
            'The modules to be saved, default is None to save all modules in checkpoint file.'
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
        self.reduce_dtype = getattr(torch,
                                    cfg.get('FSDP_REDUCE_DTYPE', 'float32'))
        self.buffer_dtype = getattr(torch,
                                    cfg.get('FSDP_BUFFER_DTYPE', 'float32'))
        self.shard_modules = cfg.get('FSDP_SHARD_MODULES', ['model'])
        self.save_modules = cfg.get('SAVE_MODULES', ['model'])
        self.train_modules = cfg.get('TRAIN_MODULES', ['model'])
        self.image_log_step = cfg.get('IMAGE_LOG_STEP', 2000)
        self._image_out = defaultdict(list)
        self.load_model_only = cfg.get('LOAD_MODEL_ONLY', False)
        self.channels_last = cfg.get('CHANNELS_LAST', False)
        self.current_batch_data = defaultdict(dict)
        self.sample_args = cfg.get('SAMPLE_ARGS', None)
        self.tuner_cfg = cfg.get('TUNER', None)
        self.freeze_cfg = cfg.get('FREEZE', None)
        self.log_train_num = cfg.get("LOG_TRAIN_NUM", -1)

    def set_up(self):
        self.construct_data()
        self.construct_model()
        self.construct_metrics()
        self.model_to_device()
        self.init_lr()
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

    def init_lr(self):
        rescale_lr = self.cfg.get('RESCALE_LR', False)
        if rescale_lr and 'train' in self.datas and self.cfg.have('OPTIMIZER'):
            if we.world_size > 1:
                all_batch_size = self.datas['train'].batch_size * we.world_size
            else:
                all_batch_size = self.datas['train'].batch_size
            self.cfg.OPTIMIZER.LEARNING_RATE *= all_batch_size
            self.cfg.OPTIMIZER.LEARNING_RATE /= 640

    def init_opti(self):
        import torch.cuda.amp as amp

        if we.is_distributed:
            if self.use_fairscale:
                from fairscale.nn.data_parallel import ShardedDataParallel
                from fairscale.optim.oss import OSS
                if hasattr(self.model, 'ignored_parameters'):
                    train_params, ignored_params = self.model.parameters(
                    ), self.model.ignored_parameters()
                else:
                    train_params, ignored_params = self.model.parameters(
                    ), None
                self.optimizer = OSS(params=train_params,
                                     optim=torch.optim.AdamW,
                                     lr=self.cfg.OPTIMIZER.LEARNING_RATE)
                self.model = ShardedDataParallel(self.model, self.optimizer)
            elif self.use_fsdp:
                shard_fn = partial
                if self.shard_modules is not None:
                    for module in self.shard_modules:
                        if isinstance(module, str):
                            sub_module = get_module(self.model, module)
                            if sub_module is not None:
                                sub_module = shard_model(
                                            sub_module,
                                            device_id=we.device_id,
                                            param_dtype=self.dtype,
                                            reduce_dtype=self.reduce_dtype,
                                            buffer_dtype=self.buffer_dtype,
                                            sharding_strategy=sharding_strategy_map[self.model_shard],
                                            sync_module_states=True)
                                set_module(self.model, module, sub_module)
                        elif isinstance(module, (dict, Config)):
                            sub_module = get_module(self.model, module["MODULE"])
                            if sub_module is not None:
                                sub_module = shard_model(
                                    sub_module,
                                    device_id=we.device_id,
                                    param_dtype=self.dtype,
                                    reduce_dtype=self.reduce_dtype,
                                    buffer_dtype=self.buffer_dtype,
                                    fsdp_group=module.get("FSDP_GROUP", ["blocks"]),
                                    sharding_strategy=sharding_strategy_map[self.model_shard],
                                    sync_module_states=True)
                                set_module(self.model, module["MODULE"], sub_module)
                else:
                    self.logger.warning(
                        'FSDP_SHARD_MODULES is None, which means wraping the whold model as the '
                        'fsdp instance. When using FSDP, it is necessary to specify the modules '
                        'to be wrapped; otherwise, there may be a situation where submodules are '
                        'not the root module, which can lead to unexpected issues. Specify the '
                        'modules to be wrapped by setting FSDP_SHARD_MODULES to a list of modules '
                        'that need wrapping.')
                    self.model = shard_fn(self.model)
                train_params = []
                if self.train_modules is None:
                    self.logger.warning(
                        'When using FSDP, it is necessary to explicitly specify the modules to be '
                        'trained or the modules for which gradients will be computed, otherwise, '
                        'there will be issues with gradient calculation.')
                    assert self.train_modules is None
                else:
                    self.logger.info(
                        f"The modules {','.join(self.train_modules)} 's parameters will be backwarded."
                    )
                for module in self.train_modules:
                    if hasattr(self.model, module):
                        current_module = getattr(self.model, module)
                        train_params += list(current_module.parameters())

                self.optimizer = OPTIMIZERS.build(self.cfg.OPTIMIZER,
                                                  logger=self.logger,
                                                  parameters=train_params)
            else:
                assert not self.model.use_ema
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[torch.cuda.current_device()],
                    output_device=torch.cuda.current_device(),
                    find_unused_parameters=False)
                self.optimizer = OPTIMIZERS.build(
                    self.cfg.OPTIMIZER,
                    logger=self.logger,
                    parameters=self.model.parameters())
        else:
            self.optimizer = OPTIMIZERS.build(
                self.cfg.OPTIMIZER,
                logger=self.logger,
                parameters=self.model.parameters())

        if self.cfg.have('LR_SCHEDULER') and self.optimizer is not None:
            self.cfg.LR_SCHEDULER.TOTAL_STEPS = self.max_steps
            self.lr_scheduler = LR_SCHEDULERS.build(self.cfg.LR_SCHEDULER,
                                                    logger=self.logger,
                                                    optimizer=self.optimizer)

        if self.cfg.DTYPE in ['float16']:
            if we.is_distributed:
                if self.use_fairscale:
                    from fairscale.optim.grad_scaler import ShardedGradScaler
                    self.scaler = ShardedGradScaler(enabled=True)
                elif self.use_fsdp:
                    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
                    self.scaler = ShardedGradScaler(enabled=True,
                                                    process_group=None)
                else:
                    self.scaler = amp.GradScaler()
            else:
                self.scaler = amp.GradScaler()
        else:
            self.scaler = None
        self.logger.info(self.model)
    def load_checkpoint(self, checkpoint: dict):
        """
        Load checkpoint function
        :param checkpoint: all tensors are on cpu, you need to transfer to gpu by hand
        :return:
        """
        if 'model' in checkpoint:
            if hasattr(self.model, 'module'):
                if self.save_modules is not None:
                    for module in self.save_modules:
                        current_module = get_module(self.model.module, module)
                        if current_module is not None and module in checkpoint[
                                'model']:
                            current_module.load_state_dict(
                                checkpoint['model'][module])
                            self.logger.info(
                                f'Load checkpoint for model.{module}')
                else:
                    self.model.module.load_state_dict(checkpoint['model'])
                    self.logger.info('Load checkpoint for model.')
            else:
                if self.save_modules is not None:
                    for module in self.save_modules:
                        current_module = get_module(self.model, module)
                        self.logger.info(f'Load checkpoint for model.{module}')
                        if current_module is not None and module in checkpoint[
                                'model']:
                            current_module.load_state_dict(
                                checkpoint['model'][module])
                else:
                    self.model.load_state_dict(checkpoint['model'])
                    self.logger.info('Load checkpoint for model.')
        else:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            self.logger.info('Load checkpoint for model.')
        if not self.load_model_only:
            if 'optimizer' in checkpoint and self.optimizer:
                if self.use_fsdp:
                    for module in self.train_modules:
                        current_module = get_module(self.model, module)
                        if current_module is not None and module in checkpoint[
                                'optimizer']:
                            state = FSDP.optim_state_dict_to_load(
                                current_module, self.optimizer,
                                checkpoint['optimizer'][module])
                            self.optimizer.load_state_dict(state)
                            self.logger.info(
                                f'Load checkpoint for optimizer {module}.')
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.logger.info(f'Load checkpoint for optimizer.')
            if 'scaler' in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler'])
                self.logger.info(f'Load checkpoint for scaler.')
        self.logger.info('Load checkpoint finished.')

    def save_checkpoint(self) -> dict:
        """
        Save checkpoint function, you need to transfer all tensors to cpu by hand
        :return:
        """
        ckpt = dict()
        if not self.use_fsdp and not we.rank == 0:
            return ckpt
        ckpt['model'] = OrderedDict()
        if we.is_distributed:
            if self.use_fsdp:
                save_policy = FullStateDictConfig(offload_to_cpu=True,
                                                  rank0_only=True)
                if self.shard_modules is not None:
                    if self.save_modules is None:
                        self.logger.warning(
                            'When using FSDP, after specifying the modules to be wrapped '
                            'using FSDP_SHARD_MODULES, please set the modules to be saved '
                            'in SAVE_MODULES. If set to None, it means all modules will '
                            'be saved. However, for nested modules, the system cannot '
                            'determine whether they are FSDP instances and need to be '
                            'explicitly set.')
                    assert self.save_modules is not None
                    for module in self.save_modules:
                        current_module = get_module(self.model, module)
                        if current_module is not None:
                            if isinstance(current_module, FSDP):
                                # print(module, current_module._is_root)
                                with FSDP.state_dict_type(
                                        current_module,
                                        StateDictType.FULL_STATE_DICT,
                                        save_policy):
                                    ckpt['model'][
                                        module] = current_module.state_dict()
                            else:
                                ckpt['model'][
                                    module] = current_module.state_dict()
                else:
                    with FSDP.state_dict_type(self.model,
                                              StateDictType.FULL_STATE_DICT,
                                              save_policy):
                        ckpt['model'] = self.model.state_dict()
            else:
                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model
                if self.save_modules is not None:
                    for module in self.save_modules:
                        current_module = get_module(self.model, module)
                        if current_module is not None:
                            ckpt['model'][module] = current_module.state_dict()
                else:
                    ckpt['model'] = model.state_dict()
        else:
            if hasattr(self.model, 'module'):
                model = self.model.module
            else:
                model = self.model
            if self.save_modules is not None:
                for module in self.save_modules:
                    current_module = get_module(self.model, module)
                    if current_module is not None:
                        ckpt['model'][module] = current_module.state_dict()
            else:
                ckpt['model'] = model.state_dict()
        if self.optimizer and not self.use_fairscale:
            if self.use_fsdp and we.is_distributed:
                ckpt['optimizer'] = OrderedDict()
                for module in self.train_modules:
                    if hasattr(self.model, module):
                        current_module = getattr(self.model, module)
                        ckpt['optimizer'][module] = FSDP.optim_state_dict(
                            current_module, self.optimizer)
            else:
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
            if 'meta' in batch_data and isinstance(batch_data['meta'], dict):
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
            if self.sample_args:
                self.current_batch_data[self.mode].update(
                    self.sample_args.get_lowercase_dict())
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
                self.train_mode()
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
            ret_images, ret_labels = [], []
            if 'hint' in result:
                ret_images.append((result['hint'][:result['image'].shape[0]].permute(1, 2, 0).cpu().numpy() *
                                   255).astype(np.uint8))
                ret_labels.append(f"Control Image")
            ret_images.append(
                (result['image'].permute(1, 2, 0).cpu().numpy() *
                 255).astype(np.uint8))
            ret_labels.append(result['prompt'] +
                              " <font color='red'> |NegPrompt| </font> " +
                              result['n_prompt'])
            log_data.append(ret_images)
            log_label.append(ret_labels)
            ori_label.append(result['prompt'])

        self.register_probe({'eval_label': log_label})
        self.register_probe({
            'eval_image':
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
        log_data, log_label, ori_label = [], [], []
        for result in all_results:
            # the inference image use
            ret_images, ret_labels = [], []
            if 'hint' in result:
                ret_images.append((result['hint'][:result['image'].shape[0]].permute(1, 2, 0).cpu().numpy() *
                                 255).astype(np.uint8))
                ret_labels.append(f"Control Image")
            ret_images.append(
                (result['image'].permute(1, 2, 0).cpu().numpy() *
                 255).astype(np.uint8))
            ret_labels.append(result['prompt'] +
                             " <font color='red'> |NegPrompt| </font> " +
                             result['n_prompt'])
            log_data.append(ret_images)
            log_label.append(ret_labels)
            ori_label.append(result['prompt'])

        self.register_probe({'test_label': log_label})
        self.register_probe({
            'test_image':
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

            self.logger.info([(key, param.shape) for key, param in model.named_parameters() if param.requires_grad])
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
            elif isinstance(train_part, str):
                for name, param in freeze_model.named_parameters():
                    if re.match(train_part, name):
                        param.requires_grad = True
        self.logger.info([(key, param.shape) for key, param in freeze_model.named_parameters() if param.requires_grad])
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
                            set_name=True,
                            exclude_keys=[
                                'EXTRA_KEYS', 'TRAIN_PRECISION', 'MAX_EPOCHS',
                                'NUM_FOLDS'
                            ])

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
            batch_data = transfer_data_to_cuda(self.current_batch_data[self.mode])
            self.eval_mode()
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                batch_data['log_num'] = self.log_train_num
                results = self.run_step_eval(batch_data)
                images = batch_data['image'] if 'image' in batch_data else [None] * len(results)
            self.train_mode()
            log_data, log_label = [], []
            for result, image in zip(results, images):
                ret_images, ret_labels = [], []
                if 'hint' in result:
                    ret_images.append((result['hint'][:result['image'].shape[0]].permute(1, 2, 0).cpu().numpy() *
                                     255).astype(np.uint8))
                if image is not None:
                    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
                    ret_images.append((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    ret_labels.append(f'target image')

                ret_images.append((result['image'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                ret_labels.append(result['prompt']
                                  + " <font color='red'> |NegPrompt| </font> "
                                  + result['n_prompt'])
                log_data.append(ret_images)
                log_label.append(ret_labels)
            self.register_probe({
                'train_image':
                ProbeData(log_data,
                          is_image=True,
                          build_html=True,
                          build_label=log_label)
            })
            self.register_probe({'train_label': log_label})
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
        frozen_param_dict = {}
        ema_param_dict = {}
        all_param_numel = 0
        if we.debug:
            for key, _ in model.named_modules():
                logger.info(f'sub modules {key}.')
        for key, val in model.named_parameters():
            if 'ema' in key:
                sub_key = '.'.join(key.split('.', 1)[-1].split('.', 1)[:1])
                if sub_key in ema_param_dict:
                    ema_param_dict[sub_key] += val.numel()
                else:
                    ema_param_dict[sub_key] = val.numel()
                continue
            if val.requires_grad:
                sub_key = '.'.join(key.split('.', 1)[-1].split('.', 2)[:2])
                if sub_key in train_param_dict:
                    train_param_dict[sub_key] += val.numel()
                else:
                    train_param_dict[sub_key] = val.numel()
            else:
                sub_key = '.'.join(key.split('.', 1)[-1].split('.', 1)[:1])
                if sub_key in frozen_param_dict:
                    frozen_param_dict[sub_key] += val.numel()
                else:
                    frozen_param_dict[sub_key] = val.numel()
            all_param_numel += val.numel()
            if we.debug:
                logger.info(key)
        train_param_numel = sum(train_param_dict.values())
        frozen_param_numel = sum(frozen_param_dict.values())
        logger.info(
            f'Load trainable params {train_param_numel} / {all_param_numel} = '
            f'{train_param_numel / all_param_numel:.2%}, '
            f'train part: {train_param_dict}.')
        logger.info(
            f'Load frozen params {frozen_param_numel} / {all_param_numel} = '
            f'{frozen_param_numel / all_param_numel:.2%}, '
            f'frozen part: {frozen_param_dict}.')
        if len(ema_param_dict) > 0:
            ema_param_numel = sum(ema_param_dict.values())
            logger.info(
                f'Load ema frozen params {ema_param_numel} / {all_param_numel} = '
                f'{ema_param_numel / all_param_numel:.2%}, '
                f'frozen part: {ema_param_dict}.')