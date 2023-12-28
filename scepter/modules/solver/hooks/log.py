# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import os
import os.path as osp
import time
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import LogAgg, time_since

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    warnings.warn(f'Runing without tensorboard! {e}')

_DEFAULT_LOG_PRIORITY = 100


def _format_float(x):
    try:
        if abs(x) - int(abs(x)) < 0.01:
            return '{:.6f}'.format(x)
        else:
            return '{:.4f}'.format(x)
    except Exception:
        return 'NaN'


def _print_v(x):
    if isinstance(x, float):
        return _format_float(x)
    elif isinstance(x, torch.Tensor) and x.ndim == 0:
        return _print_v(x.item())
    else:
        return f'{x}'


def _print_iter_log(solver, outputs, final=False, start_time=0, mode=None):
    extra_vars = solver.collect_log_vars()
    outputs.update(extra_vars)
    s = []
    for k, v in outputs.items():
        if k in ('data_time', 'time'):
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            s.append(f'{k}: ' + _print_v(v[0]) + f'({_print_v(v[1])})')
        else:
            s.append(f'{k}: ' + _print_v(v))
    if 'time' in outputs:
        v = outputs['time']
        s.insert(0, 'time: ' + _print_v(v[0]) + f'({_print_v(v[1])})')
    if 'data_time' in outputs:
        v = outputs['data_time']
        s.insert(0, 'data_time: ' + _print_v(v[0]) + f'({_print_v(v[1])})')

    if solver.max_epochs == -1:
        assert solver.max_steps > 0
        percent = (solver.total_iter +
                   1 if not final else solver.total_iter) / solver.max_steps
        now_status = time_since(start_time, percent)
        solver.logger.info(
            f'Stage [{mode}] '
            f'iter: [{solver.total_iter + 1 if not final else solver.total_iter}/{solver.max_steps}], '
            f"{', '.join(s)}, "
            f'[{now_status}]')
    else:
        assert solver.max_epochs > 0 and solver.epoch_max_iter > 0
        percent = (solver.total_iter + 1 if not final else solver.total_iter
                   ) / (solver.epoch_max_iter * solver.max_epochs)
        now_status = time_since(start_time, percent)
        solver.logger.info(
            f'Epoch [{solver.epoch}/{solver.max_epochs}], stage [{mode}] '
            f'iter: [{solver.total_iter + 1 if not final else solver.total_iter}/{solver.epoch_max_iter * solver.max_epochs}], '  # noqa
            f'iter: [{solver.iter + 1 if not final else solver.iter}/{solver.epoch_max_iter}], '
            f"{', '.join(s)}, "
            f'[{now_status}]')


def print_memory_status():
    if torch.cuda.is_available():
        nvi_info = os.popen('nvidia-smi').read()
        gpu_mem = nvi_info.split('\n')[9].split('|')[2].split('/')[0].strip()
        gpu_mem = int(gpu_mem.replace('MiB', ''))
    else:
        gpu_mem = 0
    return gpu_mem


@HOOKS.register_class()
class LogHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_LOG_PRIORITY,
            'description': 'the priority for processing!'
        },
        'LOG_INTERVAL': {
            'value': 10,
            'description': 'the interval for log print!'
        },
        'SHOW_GPU_MEM': {
            'value': False,
            'description': 'to show the gpu memory'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(LogHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_LOG_PRIORITY)
        self.log_interval = cfg.get('LOG_INTERVAL', 10)
        self.show_gpu_mem = cfg.get('SHOW_GPU_MEM', False)
        self.log_agg_dict = defaultdict(LogAgg)

        self.last_log_step = ('train', 0)

        self.time = time.time()
        self.start_time = time.time()
        self.data_time = 0

    def before_all_iter(self, solver):
        self.time = time.time()
        self.last_log_step = (solver.mode, 0)

    def before_iter(self, solver):
        data_time = time.time() - self.time
        self.data_time = data_time

    def after_iter(self, solver):
        log_agg = self.log_agg_dict[solver.mode]
        iter_time = time.time() - self.time
        self.time = time.time()
        outputs = solver.iter_outputs.copy()
        outputs['time'] = iter_time
        outputs['data_time'] = self.data_time
        if 'batch_size' in outputs:
            batch_size = outputs.pop('batch_size')
        else:
            batch_size = 1
        if self.show_gpu_mem:
            outputs['nvidia-smi'] = print_memory_status()
        log_agg.update(outputs, batch_size)
        if (solver.iter + 1) % self.log_interval == 0:
            _print_iter_log(solver,
                            log_agg.aggregate(self.log_interval),
                            start_time=self.start_time,
                            mode=solver.mode)
            self.last_log_step = (solver.mode, solver.iter + 1)

    def after_all_iter(self, solver):
        outputs = self.log_agg_dict[solver.mode].aggregate(
            solver.iter - self.last_log_step[1])
        solver.agg_iter_outputs = {
            key: value[1]
            for key, value in outputs.items()
        }
        current_log_step = (solver.mode, solver.iter)
        if current_log_step != self.last_log_step:
            _print_iter_log(solver,
                            outputs,
                            final=True,
                            start_time=self.start_time,
                            mode=solver.mode)
            self.last_log_step = current_log_step

        for _, value in self.log_agg_dict.items():
            value.reset()

    def after_epoch(self, solver):
        outputs = solver.epoch_outputs
        mode_s = []
        for mode_name, kvs in outputs.items():
            if len(kvs) == 0:
                return
            s = [f'{k}: ' + _print_v(v) for k, v in kvs.items()]
            mode_s.append(f"{mode_name} -> {', '.join(s)}")
        if len(mode_s) > 1:
            states = '\n\t'.join(mode_s)
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], \n\t'
                f'{states}')
        elif len(mode_s) == 1:
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], {mode_s[0]}')
        # summary

        for mode in self.log_agg_dict:
            solver.logger.info(f'Current Epoch {mode} Summary:')
            log_agg = self.log_agg_dict[mode]
            _print_iter_log(solver,
                            log_agg.aggregate(self.log_interval),
                            start_time=self.start_time,
                            mode=mode)
            if not mode == 'train':
                self.log_agg_dict[mode].reset()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            LogHook.para_dict,
                            set_name=True)


@HOOKS.register_class()
class TensorboardLogHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_LOG_PRIORITY,
            'description': 'the priority for processing!'
        },
        'LOG_DIR': {
            'value': None,
            'description': 'the dir for tensorboard log!'
        },
        'LOG_INTERVAL': {
            'value': 10000,
            'description': 'the interval for log upload!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(TensorboardLogHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_LOG_PRIORITY)
        self.log_dir = cfg.get('LOG_DIR', None)
        self.log_interval = cfg.get('LOG_INTERVAL', 1000)
        self._local_log_dir = None
        self.writer: Optional[SummaryWriter] = None

    def before_solve(self, solver):
        if we.rank != 0:
            return

        if self.log_dir is None:
            self.log_dir = osp.join(solver.work_dir, 'tensorboard')

        self._local_log_dir, _ = FS.map_to_local(self.log_dir)
        os.makedirs(self._local_log_dir, exist_ok=True)
        self.writer = SummaryWriter(self._local_log_dir)
        solver.logger.info(f'Tensorboard: save to {self.log_dir}')

    def after_iter(self, solver):
        if self.writer is None:
            return
        outputs = solver.iter_outputs.copy()
        extra_vars = solver.collect_log_vars()
        outputs.update(extra_vars)
        mode = solver.mode
        for key, value in outputs.items():
            if key == 'batch_size':
                continue
            if isinstance(value, torch.Tensor):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = value.item()
            elif isinstance(value, np.ndarray):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = float(value)
            elif isinstance(value, numbers.Number):
                # Must be number
                pass
            else:
                continue

            self.writer.add_scalar(f'{mode}/iter/{key}',
                                   value,
                                   global_step=solver.total_iter)
        if solver.total_iter % self.log_interval:
            self.writer.flush()
            # Put to remote file systems every epoch
            FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_epoch(self, solver):
        if self.writer is None:
            return
        outputs = solver.epoch_outputs.copy()
        for mode, kvs in outputs.items():
            for key, value in kvs.items():
                self.writer.add_scalar(f'{mode}/epoch/{key}',
                                       value,
                                       global_step=solver.epoch)

        self.writer.flush()
        # Put to remote file systems every epoch
        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_solve(self, solver):
        if self.writer is None:
            return
        if self.writer:
            self.writer.close()

        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            TensorboardLogHook.para_dict,
                            set_name=True)
