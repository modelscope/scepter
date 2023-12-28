# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from collections import OrderedDict, defaultdict

import torch

from scepter.modules.solver.base_solver import BaseSolver
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.data import (transfer_data_to_cpu,
                                        transfer_data_to_cuda)
from scepter.modules.utils.distribute import gather_data, we
from scepter.modules.utils.file_system import FS


def _get_value(data: dict, key: str):
    """ Recursively get value from data by a multi-level key.

    Args:
        data (dict):
        key (str): 'data', 'meta.path', 'a.b.c'

    Returns:
        Value.

    """
    if not isinstance(data, dict):
        return None
    if key in data:
        return data[key]
    elif '.' in key:
        par_key = key.split('.')[0]
        sub_key = '.'.join(key.split('.')[1:])
        if par_key in data:
            return _get_value(data[par_key], sub_key)
    return None


@SOLVERS.register_class()
class TrainValSolver(BaseSolver):
    """ Standard train and eval steps solver

    Args:
        model (torch.nn.Module): Model to train or eval.

    """
    para_dict = {
        'DO_FINAL_EVAL': {
            'value': False,
            'description': 'If do final evaluation or not.'
        },
        'SAVE_EVAL_DATA': {
            'value': False,
            'description': 'If save the evaluation data or not.'
        }
    }
    para_dict.update(BaseSolver.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        if not self.use_pl:
            if 'train' in self.datas and self.cfg.have('OPTIMIZER'):
                self.cfg.OPTIMIZER.LEARNING_RATE *= self.datas[
                    'train'].batch_size
                if we.world_size > 1:
                    self.cfg.OPTIMIZER.LEARNING_RATE *= we.world_size
                    self.cfg.OPTIMIZER.LEARNING_RATE *= self.accu_step
                self.cfg.OPTIMIZER.LEARNING_RATE /= 96

    def construct_metrics(self):
        # Initial metric
        super().construct_metrics()
        self.metrics = []
        if self.cfg.have('METRICS'):
            self.extra_keys = self.cfg.get('EXTRA_KEYS', [])
            self._collect_keys = set()
            self.do_final_eval = self.cfg.get('DO_FINAL_EVAL', False)
            self.save_eval_data = self.cfg.get('SAVE_EVAL_DATA', False)
            if self.do_final_eval or self.save_eval_data:
                self._build_metrics(self.cfg.METRICS)
                self._collect_keys.update(list(self.extra_keys or []))
                self._collect_keys = sorted(list(self._collect_keys))
                if len(self._collect_keys) > 0:
                    self.logger.info(
                        f"{', '.join(self._collect_keys)} will be collected during eval epoch"
                    )

    @torch.no_grad()
    def run_eval(self):
        self.eval_mode()
        collect_data = defaultdict(list)
        rank, world_size = we.rank, we.world_size
        self.before_all_iter(self.hooks_dict[self._mode])
        for data in self.datas[self._mode].dataloader:
            self.before_iter(self.hooks_dict[self._mode])
            data_gpu = transfer_data_to_cuda(data)
            result = self.model(**data_gpu)
            self._iter_outputs[self._mode] = self._reduce_scalar(result)
            if self.do_final_eval or self.save_eval_data:
                # Collect data
                if isinstance(result, torch.Tensor):
                    data_gpu['result'] = result
                elif isinstance(result, dict):
                    data_gpu.update(result)

                step_data = OrderedDict()
                for key in self._collect_keys:
                    value = _get_value(data_gpu, key)
                    if value is None:
                        raise ValueError(
                            f'Cannot get valid value from model input or output data with key {key}'
                        )
                    step_data[key] = value

                step_data = transfer_data_to_cpu(step_data)

                for key, value in step_data.items():
                    if isinstance(value, torch.Tensor):
                        collect_data[key].append(value.clone())
                    else:
                        collect_data[key].append(value)

            self.after_iter(self.hooks_dict[self._mode])
        self.after_all_iter(self.hooks_dict[self._mode])

        if self.do_final_eval or self.save_eval_data:
            # Concat collect_data
            concat_collect_data = OrderedDict()
            for key, tensors in collect_data.items():
                if isinstance(tensors[0], torch.Tensor):
                    concat_collect_data[key] = torch.cat(tensors)
                elif isinstance(tensors[0], list):
                    concat_collect_data[key] = sum(tensors, [])
                else:
                    concat_collect_data[key] = tensors

            # If distributed and use DistributedSampler
            # Gather all collect data to rank 0
            if world_size > 1 and type(
                    self.datas[self._mode].sampler
            ) is torch.utils.data.DistributedSampler:
                concat_collect_data = {
                    key: gather_data(concat_collect_data[key])
                    for key in self._collect_keys
                }

            # Do final evaluate
            if self.do_final_eval and rank == 0:
                for metric in self.metrics:
                    self._epoch_outputs[self._mode].update(metric['fn'](
                        *[concat_collect_data[key] for key in metric['keys']]))

            # Save all data
            if self.save_eval_data and rank == 0:
                # minus 1, means index
                save_path = osp.join(
                    self.work_dir,
                    'eval_{:05d}.pth'.format(self.epoch + self.num_folds))
                with FS.put_to(save_path) as local_file:
                    torch.save(concat_collect_data, local_file)

    def load_checkpoint(self, checkpoint: dict):
        self._epoch = checkpoint['epoch']
        for mode_name, total_iter in checkpoint['total_iters'].items():
            self._total_iter[mode_name] = total_iter
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['checkpoint'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self._epoch += 1  # Move to next epoch

    def save_checkpoint(self) -> dict:
        checkpoint = {
            'epoch': self._epoch,
            'total_iters': self._total_iter,
            'state_dict': self.model.state_dict(),
            'checkpoint': self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        return checkpoint

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
                            TrainValSolver.para_dict,
                            set_name=True)
