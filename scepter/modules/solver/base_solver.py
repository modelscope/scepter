# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import numbers
import os
import warnings
from abc import ABCMeta
from collections import OrderedDict, defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel

from scepter.modules.data.dataset import DATASETS
from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.metric.registry import METRICS
from scepter.modules.model.registry import MODELS
from scepter.modules.opt.lr_schedulers import LR_SCHEDULERS
from scepter.modules.opt.optimizers import OPTIMIZERS
from scepter.modules.solver.hooks import HOOKS
from scepter.modules.utils.config import Config, dict_to_yaml
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.directory import get_relative_folder, osp_path
from scepter.modules.utils.distribute import dist, gather_data, we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger, init_logger
from scepter.modules.utils.probe import (ProbeData, merge_gathered_probe,
                                         register_data)

try:
    import pytorch_lightning as pl

    class PyLightningWrapper(pl.LightningModule):
        def __init__(self, solver):
            super().__init__()
            self.solver = solver
            self.model = self.solver.model

        def configure_optimizers(self):
            optimizer, lr_scheduler = self.solver.optimizer, self.solver.lr_scheduler
            if lr_scheduler is None:
                return [optimizer]
            return [optimizer], [lr_scheduler]

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            mode = 'train'
            self.solver._mode = mode
            self.solver._epoch = self.current_epoch
            self.solver._total_iter[mode] = self.global_step
            self.solver._iter[mode] = batch_idx
            if self.current_epoch >= 1:
                self.solver._epoch_max_iter[mode] = (
                    self.global_step - batch_idx) // self.current_epoch
            self.solver.before_iter(self.solver.hooks_dict[mode])
            results = self.solver.run_step_train(batch,
                                                 batch_idx,
                                                 step=self.global_step,
                                                 rank=self.global_rank)
            self.solver._iter_outputs[mode] = self._reduce_scalar(results)
            self.set_log(mode)
            self.solver.after_iter(self.solver.hooks_dict[mode])
            return results['loss']

        @torch.no_grad()
        def validation_step(self, batch, batch_idx):
            mode = 'eval'
            self.solver._mode = mode
            self.solver._iter[mode] = batch_idx
            self.solver.before_iter(self.solver.hooks_dict[mode])
            results = self.solver.run_step_eval(batch,
                                                batch_idx,
                                                step=self.global_step,
                                                rank=self.global_rank)
            self.solver._iter_outputs[mode] = self._reduce_scalar(results)
            self.set_log(mode)
            self.solver.after_iter(self.solver.hooks_dict[mode])

        @torch.no_grad()
        def test_step(self, batch, batch_idx):
            mode = 'test'
            self.solver._mode = mode
            self.solver.before_iter(self.hooks_dict[mode])
            results = self.solver.run_step_test(batch,
                                                batch_idx,
                                                step=self.global_step,
                                                rank=self.global_rank)
            self.solver._iter_outputs[mode] = self._reduce_scalar(results)
            self.set_log(mode)
            self.solver.after_iter(self.solver.hooks_dict[mode])

        # redefine the folowing function
        def on_train_epoch_start(self) -> None:
            self.solver.before_epoch(self.solver.hooks_dict['train'])

        def on_train_epoch_end(self) -> None:
            self.solver.after_epoch(self.solver.hooks_dict['train'])

        def on_validation_epoch_start(self) -> None:
            self.solver.before_epoch(self.solver.hooks_dict['eval'])

        def on_validation_epoch_end(self) -> None:
            self.solver.after_epoch(self.solver.hooks_dict['eval'])

        def on_test_epoch_start(self) -> None:
            self.solver.before_epoch(self.solver.hooks_dict['test'])

        def on_test_epoch_end(self) -> None:
            self.solver.after_epoch(self.solver.hooks_dict['test'])

        def setup(self, stage: str) -> None:
            self.solver.logger = get_logger(name='scepter')
            self.solver._prefix = FS.init_fs_client(self.solver.file_system,
                                                    logger=self.solver.logger)
            self.solver._local_rank = self.global_rank
            we.rank = self.global_rank
            local_devices = os.environ.get(
                'LOCAL_WORLD_SIZE') or torch.cuda.device_count()
            local_devices = int(local_devices)
            we.device_count = local_devices
            we.device_id = self.global_rank % local_devices
            self.solver.set_up_pre()
            super().setup(stage)

        def on_fit_start(self):
            super().on_fit_start()
            self.solver.before_solve()

        def on_fit_end(self):
            self.solver.after_solve()
            super().on_fit_end()

        def _reduce_scalar(self, data_dict: dict):
            """ Only reduce all scalar tensor values if distributed.
            Any way, loss tensor will be specially processed just in case.

            Args:
                data_dict: Dict result returned by model.

            Returns:
                A new data dict whose tensor scalar values is all-reduced.

            """
            if isinstance(data_dict, OrderedDict):
                keys = data_dict.keys()
            else:
                keys = sorted(list(data_dict.keys()))

            ret = OrderedDict()
            for key in keys:
                value = data_dict[key]
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    ret[key] = value.data.clone()
                else:
                    ret[key] = value
            return ret

        def set_log(self, mode):
            now_outputs = self.solver._iter_outputs[mode]
            extra_vars = self.solver.collect_log_vars()
            now_outputs.update(extra_vars)
            for k, v in now_outputs.items():
                if k == 'batch_size':
                    continue
                if isinstance(v, torch.Tensor) and v.ndim == 0 or isinstance(
                        v, numbers.Number):
                    if mode == 'train':
                        self.log(f'{mode}_{k}',
                                 v,
                                 prog_bar=True,
                                 logger=True,
                                 on_step=True,
                                 rank_zero_only=True,
                                 sync_dist=True)
                    else:
                        self.log(f'{mode}_{k}',
                                 v,
                                 prog_bar=True,
                                 logger=True,
                                 on_epoch=True,
                                 rank_zero_only=True,
                                 sync_dist=True)

except Exception as e:
    warnings.warn(f'{e}')


class BaseSolver(object, metaclass=ABCMeta):
    """ Base Solver.
        To initialize the solver.
        We have to initialize the data, model, optimizer and schedule.
        To process the common processing we also have to initialize the hooks.
        How to support Pytorch_lightning framework? Take a simple task as an examples.
    """
    para_dict = {
        'TRAIN_PRECISION': {
            'value': 32,
            'description': 'The precision for train process.'
        },
        'FILE_SYSTEM': {},
        'ACCU_STEP': {
            'value':
            1,
            'description':
            'When use ddp, the grad accumulate steps for each process.'
        },
        'RESUME_FROM': {
            'value': '',
            'description': 'Resume from some state of training!'
        },
        'MAX_EPOCHS': {
            'value': 10,
            'description': 'Max epochs for training.'
        },
        'NUM_FOLDS': {
            'value': 1,
            'description': 'Num folds for training.'
        },
        'WORK_DIR': {
            'value': '',
            'description': 'Save dir of the training log or model.'
        },
        'LOG_FILE': {
            'value': '',
            'description': 'Save log path.'
        },
        'EVAL_INTERVAL': {
            'value': 1,
            'description': 'Eval the model interval.'
        },
        'EXTRA_KEYS': {
            'value': [],
            'description': 'The extra keys for metric.'
        },
        'TRAIN_DATA': {
            'description': 'Train data config.'
        },
        'EVAL_DATA': {
            'description': 'Eval data config.'
        },
        'TEST_DATA': {
            'description': 'Test data config.'
        },
        'TRAIN_HOOKS': [],
        'EVAL_HOOKS': [],
        'TEST_HOOKS': [],
        'MODEL': {},
        'OPTIMIZER': {},
        'LR_SCHEDULER': {},
        'METRICS': []
    }

    def __init__(self, cfg, logger=None):
        # initialize some hyperparameters
        self.file_system = cfg.get('FILE_SYSTEM', None)
        self.work_dir: str = cfg.WORK_DIR
        self.pl_dir = self.work_dir
        self.log_file = osp_path(self.work_dir, cfg.LOG_FILE)
        self.optimizer, self.lr_scheduler = None, None
        self.cfg = cfg
        self.logger = logger
        self.resume_from: str = cfg.RESUME_FROM
        self.max_epochs: int = cfg.MAX_EPOCHS
        self.use_pl = we.use_pl
        self.train_precision = self.cfg.get('TRAIN_PRECISION', 32)
        self._mode_set = set()
        self._mode = 'train'
        self.probe_ins = {}
        self.clear_probe_ins = {}
        self._num_folds: int = 1
        if not self.use_pl:
            world_size = we.world_size
            if world_size > 1:
                self._num_folds: int = cfg.NUM_FOLDS
            if cfg.have('MODE'):
                self._mode_set.add(cfg.MODE)
                self._mode = cfg.MODE
            if we.is_distributed:
                self.accu_step = cfg.get('ACCU_STEP', 1)

        self.do_step = True
        self.hooks_dict = {'train': [], 'eval': [], 'test': []}
        self.datas = {}
        # Other initialized parameters
        self._epoch: int = 0
        # epoch_max_iter, iter, total_iter, iter_outputs, epoch_outputs
        # values is different according to self._mode
        self._epoch_max_iter: defaultdict = defaultdict(int)
        self._iter: defaultdict = defaultdict(int)
        self._total_iter: defaultdict = defaultdict(int)
        self._iter_outputs = defaultdict(dict)
        self._agg_iter_outputs = defaultdict(dict)
        self._epoch_outputs = defaultdict(dict)
        self._probe_data = defaultdict(dict)
        self._dist_data = defaultdict(dict)
        self._model_parameters = 0
        self._model_flops = 0
        self._loss = None  # loss tensor
        self._local_rank = we.rank
        if isinstance(self.file_system, list):
            for file_sys in self.file_system:
                FS.init_fs_client(file_sys, logger=self.logger)
        elif self.file_system is not None:
            FS.init_fs_client(self.file_system, logger=self.logger)
        self._prefix = FS.get_fs_client(self.work_dir).get_prefix()
        if not FS.exists(self.work_dir):
            FS.make_dir(self.work_dir)
        self.logger.info(
            f"Parse work dir {self.work_dir}'s prefix is {self._prefix}")

    def set_up_pre(self):
        # initialize Enviranment
        if self._local_rank == 0:
            if self.log_file.startswith('file://'):
                save_folder = get_relative_folder(self.log_file, -1)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
            elif not self.log_file.startswith(self._prefix):
                self.log_file = os.path.join(self._prefix, self.log_file)
            init_logger(self.logger,
                        log_file=self.log_file,
                        dist_launcher='pytorch')
        self.construct_hook()

    def __setattr__(self, key, value):
        if isinstance(value, BaseModel):
            self.probe_ins[key] = value.probe_data
            self.clear_probe_ins[key] = value.clear_probe
        super().__setattr__(key, value)

    def set_up(self):
        self.construct_data()
        self.construct_model()
        self.construct_metrics()
        if not self.use_pl:
            self.model_to_device()
        self.init_opti()
        if self.use_pl:
            self.local_work_dir, _ = FS.map_to_local(self.work_dir)
            os.makedirs(self.local_work_dir, exist_ok=True)
            # resume
            resume_local_file = None
            if self.resume_from is not None and FS.exists(self.resume_from):
                with FS.get_from(self.resume_from,
                                 wait_finish=True) as local_file:
                    self.logger.info(
                        f'Loading checkpoint from {self.resume_from}')
                    resume_local_file = local_file
            self.pl_ins = PyLightningWrapper(self)
            self.pl_trainer = pl.Trainer(
                default_root_dir=self.local_work_dir,
                max_epochs=self.max_epochs,
                precision=self.train_precision,
                accelerator='auto',
                devices='auto',
                check_val_every_n_epoch=self.eval_interval,
                resume_from_checkpoint=resume_local_file)
            self.pl_dir = os.path.join(
                self.work_dir,
                '/'.join(self.pl_trainer.log_dir.split('/')[-2:]))

    def construct_data(self):
        def one_device_init():
            # initialize data
            assert self.cfg.have('TRAIN_DATA') or self.cfg.have(
                'EVAL_DATA') or self.cfg.have('TEST_DATA')
            if self.cfg.have('TRAIN_DATA') and ('train' in self._mode_set
                                                or len(self._mode_set) < 1):
                self.cfg.TRAIN_DATA.NUM_FOLDS = self.num_folds
                train_data = DATASETS.build(self.cfg.TRAIN_DATA,
                                            logger=self.logger)
                self.datas['train'] = train_data
                self._mode_set.add('train')
                if not self.use_pl:
                    self._epoch_max_iter['train'] = len(
                        train_data.dataloader) // self.num_folds + 1
                else:
                    self._epoch_max_iter['train'] = -1
            if self.cfg.have('EVAL_DATA'):
                eval_data = DATASETS.build(self.cfg.EVAL_DATA,
                                           logger=self.logger)
                self.datas['eval'] = eval_data
                self._mode_set.add('eval')
                if not self.use_pl:
                    self._epoch_max_iter['eval'] = len(eval_data.dataloader)
                else:
                    self._epoch_max_iter['eval'] = -1
            if self.cfg.have('TEST_DATA'):
                test_data = DATASETS.build(self.cfg.TEST_DATA,
                                           logger=self.logger)
                self.datas['test'] = test_data
                if not self.use_pl:
                    self._epoch_max_iter['test'] = len(test_data.dataloader)
                else:
                    self._epoch_max_iter['test'] = -1
                self._mode_set.add('test')

        one_device_init()

    def construct_hook(self):
        # initialize data
        assert self.use_pl or self.cfg.have('TRAIN_HOOKS') or self.cfg.have(
            'EVAL_HOOKS') or self.cfg.have('TEST_HOOKS')
        if self.cfg.have('TRAIN_HOOKS') and ('train' in self._mode_set
                                             or len(self._mode_set) < 1):
            self.hooks_dict['train'] = self._load_hook(self.cfg.TRAIN_HOOKS)
        if self.cfg.have('EVAL_HOOKS'):
            assert self.cfg.have('EVAL_HOOKS')
            self.hooks_dict['eval'] = self._load_hook(self.cfg.EVAL_HOOKS)
        if self.cfg.have('TEST_HOOKS') and ('test' in self._mode_set
                                            or 'eval' not in self._mode_set):
            self.hooks_dict['test'] = self._load_hook(self.cfg.TEST_HOOKS)

    def construct_model(self):
        # initialize Model
        assert self.cfg.have('MODEL')
        self.model = MODELS.build(self.cfg.MODEL, logger=self.logger)

    def construct_metrics(self):
        # Initial metric
        self.metrics = []
        self.eval_interval = self.cfg.get('EVAL_INTERVAL', 1)

    def model_to_device(self, tg_model_ins=None):
        # Initialize distributed model
        if tg_model_ins is None:
            tg_model = self.model
        else:
            tg_model = tg_model_ins
        if we.is_distributed and we.sync_bn is True:
            self.logger.info('Convert BatchNorm to Synchronized BatchNorm...')
            tg_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tg_model)
        tg_model = tg_model.to(we.device_id)
        if we.is_distributed:
            tg_model = DistributedDataParallel(
                tg_model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                broadcast_buffers=True)
            self.logger.info('Transfer to ddp ...')
        if tg_model_ins is None:
            self.model = tg_model
        else:
            return tg_model

    def init_opti(self):
        if self.cfg.have('OPTIMIZER'):
            self.optimizer = OPTIMIZERS.build(
                self.cfg.OPTIMIZER,
                logger=self.logger,
                parameters=self.model.parameters())
        if self.cfg.have('LR_SCHEDULER') and self.optimizer is not None:
            self.lr_scheduler = LR_SCHEDULERS.build(self.cfg.LR_SCHEDULER,
                                                    logger=self.logger,
                                                    optimizer=self.optimizer)

    def solve(self, epoch=None, every_epoch=False):
        if not self.use_pl:
            if epoch is not None:
                self.epoch = epoch
            self.before_solve()
            if self.epoch >= self.max_epochs:
                self.logger.info(
                    f'Nothing to do because current epoch {self.epoch} greater max epoches {self.epoch}'
                )
            while self.epoch < self.max_epochs:
                self.solve_train()
                self.solve_eval()
                self.solve_test()
                if 'train' not in self._mode_set and not every_epoch:
                    break
            self.after_solve()
        else:
            train_dataloader = None
            if 'train' in self.datas:
                train_dataloader = self.datas['train'].dataloader
            val_dataloader = None
            if 'eval' in self.datas:
                val_dataloader = self.datas['eval'].dataloader
            self.pl_trainer.fit(self.pl_ins,
                                train_dataloaders=train_dataloader,
                                val_dataloaders=val_dataloader)
            if 'test' in self.datas:
                self.pl_trainer.test(self.pl_ins,
                                     dataloaders=self.datas['test'].dataloader)

    def solve_train(self):
        current_mode = 'train'
        if current_mode in self._mode_set:
            self.logger.info(
                f'Begin to solve {current_mode} at Epoch [{self.epoch}/{self.max_epochs}]...'
            )
            self.before_epoch(self.hooks_dict[current_mode])
            self.run_train()
            self.after_epoch(self.hooks_dict[current_mode])

    def solve_eval(self):
        current_mode = 'eval'
        if current_mode in self._mode_set and self.epoch % self.eval_interval == 0:
            self.logger.info(
                f'Begin to solve {current_mode} at Epoch [{self.epoch}/{self.max_epochs}]...'
            )
            self.before_epoch(self.hooks_dict[current_mode])
            self.run_eval()
            self.after_epoch(self.hooks_dict[current_mode])

    def solve_test(self):
        current_mode = 'test'
        if current_mode in self._mode_set:
            self.logger.info(
                f'Begin to solve {current_mode} at Epoch [{self.epoch}/{self.max_epochs}]...'
            )
            self.before_epoch(self.hooks_dict[current_mode])
            self.run_test()
            self.after_epoch(self.hooks_dict[current_mode])

    def before_solve(self):
        for k, hooks in self.hooks_dict.items():
            [t.before_solve(self) for t in hooks]

    def after_solve(self):
        for k, hooks in self.hooks_dict.items():
            [t.after_solve(self) for t in hooks]

    def run_train(self):
        self.train_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        for batch_idx, batch_data in enumerate(
                self.datas[self._mode].dataloader):
            self.before_iter(self.hooks_dict[self._mode])
            results = self.run_step_train(transfer_data_to_cuda(batch_data),
                                          batch_idx,
                                          step=self.total_iter,
                                          rank=we.rank)
            self._iter_outputs[self._mode] = self._reduce_scalar(results)
            self.after_iter(self.hooks_dict[self._mode])
        self.after_all_iter(self.hooks_dict[self._mode])

    def run_step_train(self, batch_data, batch_idx=0, step=None, rank=None):
        results = self.model(**batch_data)
        return results

    @torch.no_grad()
    def run_eval(self):
        self.eval_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        for batch_idx, batch_data in enumerate(
                self.datas[self._mode].dataloader):
            self.before_iter(self.hooks_dict[self._mode])
            results = self.run_step_eval(transfer_data_to_cuda(batch_data),
                                         batch_idx,
                                         step=self.total_iter,
                                         rank=we.rank)
            self._iter_outputs[self._mode] = self._reduce_scalar(results)
            self.after_iter(self.hooks_dict[self._mode])
        self.after_all_iter(self.hooks_dict[self._mode])

    def run_step_eval(self, batch_data, batch_idx=0, step=None, rank=None):
        results = self.model(**batch_data)
        return results

    @torch.no_grad()
    def run_test(self):
        self.test_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        for batch_idx, batch_data in enumerate(
                self.datas[self._mode].dataloader):
            self.before_iter(self.hooks_dict[self._mode])
            results = self.run_step_test(transfer_data_to_cuda(batch_data),
                                         batch_idx,
                                         step=self.total_iter,
                                         rank=we.rank)
            self._iter_outputs[self._mode] = self._reduce_scalar(results)
            self.after_iter(self.hooks_dict[self._mode])
        self.after_all_iter(self.hooks_dict[self._mode])

    def run_step_test(self, batch_data, batch_idx=0, step=None, rank=None):
        results = self.model(**batch_data)
        return results

    @torch.no_grad()
    def register_flops(self, data, keys=[]):
        from fvcore.nn import FlopCountAnalysis
        if len(keys) < 1:
            keys = list(data.keys())
        for key in data:
            if isinstance(data[key], torch.Tensor):
                batch_one_data = data[key][0, ...]
                batch_one_data = torch.unsqueeze(batch_one_data, dim=0)
                data[key] = batch_one_data
            elif isinstance(data[key], list):
                data[key] = data[key][0]

        tensor = [data[k] for k in keys]
        flops = FlopCountAnalysis(self.model, tuple(tensor))
        self._model_flops = flops.total()

    def before_epoch(self, hooks):
        [t.before_epoch(self) for t in hooks]

    def before_all_iter(self, hooks):
        [t.before_all_iter(self) for t in hooks]

    def before_iter(self, hooks):
        if not self.use_pl and self.is_train_mode:
            self._epoch = self._total_iter[self._mode] // self._epoch_max_iter[
                self._mode] + 1
            if self._iter[self._mode] % self._epoch_max_iter[self._mode] == 0:
                self._iter[self._mode] = 0
        [t.before_iter(self) for t in hooks]

    def after_iter(self, hooks):
        [t.after_iter(self) for t in hooks]
        if not self.use_pl:
            self._total_iter[self._mode] += 1
            self._iter[self._mode] += 1
        self.clear_probe()

    def after_all_iter(self, hooks):
        [t.after_all_iter(self) for t in hooks]

    def after_epoch(self, hooks):
        [t.after_epoch(self) for t in hooks]
        self._iter.clear()
        self._iter_outputs.clear()
        self._epoch_outputs.clear()
        if self.use_pl:
            FS.put_dir_from_local_dir(self.local_work_dir, self.work_dir)

    def collect_log_vars(self) -> OrderedDict:
        ret = OrderedDict()
        if self.is_train_mode and self.optimizer is not None:
            for idx, pg in enumerate(self.optimizer.param_groups):
                ret[f'pg{idx}_lr'] = pg['lr']
        return ret

    def load_checkpoint(self, checkpoint: dict):
        """
        Load checkpoint function
        :param checkpoint: all tensors are on cpu, you need to transfer to gpu by hand
        :return:
        """
        pass

    def save_checkpoint(self) -> dict:
        """
        Save checkpoint function, you need to transfer all tensors to cpu by hand
        :return:
        """
        pass

    @property
    def num_folds(self) -> int:
        return self._num_folds

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, new_epoch):
        self._epoch = new_epoch

    @property
    def iter(self) -> int:
        return self._iter[self._mode]

    @property
    def probe_data(self):
        return self._probe_data[self._mode]

    @property
    def total_iter(self) -> int:
        return self._total_iter[self._mode]

    @property
    def epoch_max_iter(self) -> int:
        return self._epoch_max_iter[self._mode]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def iter_outputs(self) -> dict:
        return self._iter_outputs[self._mode]

    @property
    def agg_iter_outputs(self) -> dict:
        return self._agg_iter_outputs

    @agg_iter_outputs.setter
    def agg_iter_outputs(self, new_outputs):
        assert type(new_outputs) is dict
        self._agg_iter_outputs[self._mode] = new_outputs

    @property
    def epoch_outputs(self) -> dict:
        return self._epoch_outputs

    @property
    def is_train_mode(self):
        return self._mode == 'train'

    @property
    def is_eval_mode(self):
        return self._mode == 'eval'

    @property
    def is_test_mode(self):
        return self._mode == 'test'

    def train_mode(self):
        self.model.train()
        self._mode = 'train'

    def eval_mode(self):
        self.model.eval()
        self._mode = 'eval'

    def test_mode(self):
        self.model.eval()
        self._mode = 'test'

    def register_probe(self, probe_data: dict):
        probe_da, dist_da = register_data(probe_data,
                                          key_prefix=__class__.__name__)
        self._probe_data[self.mode].update(probe_da)
        for key in dist_da:
            if key not in self._dist_data[self.mode]:
                self._dist_data[self.mode][key] = dist_da[key]
            else:
                for k, v in dist_da[key].items():
                    if k in self._dist_data[self.mode][key]:
                        self._dist_data[self.mode][key][k] += v
                    else:
                        self._dist_data[self.mode][key][k] = v

    @property
    def probe_data(self):  # noqa
        gather_probe_data = gather_data(self._probe_data[self.mode])
        _dist_data_list = gather_data([self._dist_data[self.mode] or {}])
        if not we.rank == 0:
            self._probe_data[self.mode] = {}
            self._dist_data[self.mode] = {}
        # Iterate recurse the sub class's probe data.
        for k, func in self.probe_ins.items():
            for kk, vv in func().items():
                self._probe_data[self.mode][f'{k}/{kk}'] = vv
        if gather_probe_data is not None:
            # Before processing, just merge the data.
            self._probe_data[self.mode] = merge_gathered_probe(
                gather_probe_data)
        if _dist_data_list is not None and len(_dist_data_list) > 0:
            reduce_dist_data = {}
            for one_data in _dist_data_list:
                for k, v in one_data.items():
                    if k in reduce_dist_data:
                        for kk, vv in v.items():
                            if kk in reduce_dist_data[k]:
                                reduce_dist_data[k][kk] += vv
                            else:
                                reduce_dist_data[k][kk] = vv
                    else:
                        reduce_dist_data[k] = v
            self._dist_data[self.mode] = reduce_dist_data
        self._probe_data[
            self.mode][f'{__class__.__name__}_distribute'] = ProbeData(
                self._dist_data[self.mode])
        norm_dist_data = {}
        for key, value in self._dist_data[self.mode].items():
            total = 0
            for k, v in value.items():
                total += v
            norm_v = {}
            for k, v in value.items():
                norm_v[k] = v / total
            norm_dist_data[key] = norm_v
        self._probe_data[
            self.mode][f'{__class__.__name__}_norm_distribute'] = ProbeData(
                norm_dist_data)
        ret_data = copy.deepcopy(self._probe_data[self.mode])
        self._probe_data[self.mode] = {}
        return ret_data

    def clear_probe(self):
        self._probe_data[self.mode].clear()
        # Iterate recurse the sub class's probe data.
        for k, func in self.clear_probe_ins.items():
            func()

    def _load_hook(self, hooks):
        ret_hooks = []
        if hooks is not None and len(hooks) > 0:
            for hook_cfg in hooks:
                if self.use_pl:
                    if 'backward' in hook_cfg.NAME.lower(
                    ) or 'lrhook' in hook_cfg.NAME.lower(
                    ) or 'samplerhook' in hook_cfg.NAME.lower():
                        self.logger.info(
                            f'Hook {hook_cfg.NAME} is not useful when use PytorchLightning!'
                        )
                        continue
                ret_hooks.append(HOOKS.build(hook_cfg, logger=self.logger))
        ret_hooks.sort(key=lambda a: a.priority)
        return ret_hooks

    def get_optim_parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

    def _reduce_scalar(self, data_dict: dict):
        """ Only reduce all scalar tensor values if distributed.
        Any way, loss tensor will be specially processed just in case.

        Args:
            data_dict: Dict result returned by model.

        Returns:
            A new data dict whose tensor scalar values is all-reduced.

        """
        if 'loss' in data_dict:
            self.loss = data_dict['loss']
            data_dict['loss'] = self.loss.data.clone()

        if isinstance(data_dict, OrderedDict):
            keys = data_dict.keys()
        else:
            keys = sorted(list(data_dict.keys()))

        ret = OrderedDict()
        # print([(key, type(data_dict[key])) for key in keys], f"{dist.get_rank()}", f"{self.iter}")
        for key in keys:
            value = data_dict[key]
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                if dist.is_available() and dist.is_initialized():
                    value = value.data.clone()
                    dist.all_reduce(value.div_(dist.get_world_size()))
                ret[key] = value
            else:
                ret[key] = value

        return ret

    def _build_metrics(self, cfgs, logger=None):
        if isinstance(cfgs, (list, tuple)):
            for cfg in cfgs:
                self._build_metrics(cfg, logger=logger)
        elif isinstance(cfgs, Config):
            fn = METRICS.build(cfgs, logger)
            keys = cfgs.KEYS
            self.metrics.append({'fn': fn, 'keys': keys})
            self._collect_keys.update(keys)

    def print_memory_status(self):
        if torch.cuda.is_available():
            nvi_info = os.popen('nvidia-smi').read()
            gpu_mem = nvi_info.split('\n')[9].split('|')[2].split(
                '/')[0].strip()
        else:
            gpu_mem = ''
        return gpu_mem

    def print_model_params_status(self, model=None, logger=None):
        """Print the status and parameters of the model"""
        if model is None:
            model = self.model
        if logger is None:
            logger = self.logger
        train_param_dict = {}
        forzen_param_dict = {}
        all_param_numel = 0
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

    @staticmethod
    def get_config_template():
        '''
        { "ENV" :
            { "description" : "",
              "A" : {
                    "value": 1.0,
                    "description": ""
               }
            }
        }
        :return:
        '''
        return dict_to_yaml('solvername',
                            __class__.__name__,
                            BaseSolver.para_dict,
                            set_name=True)
