# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import torch
from safetensors.torch import save_file

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

_DEFAULT_CHECKPOINT_PRIORITY = 300


@HOOKS.register_class()
class SafetensorsHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_CHECKPOINT_PRIORITY,
            'description': 'the priority for processing!'
        },
        'INTERVAL': {
            'value': 1,
            'description': 'the interval of saving checkpoint!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(SafetensorsHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_CHECKPOINT_PRIORITY)
        self.interval = cfg.get('INTERVAL', 5000)
        self.save_name_prefix = cfg.get('SAVE_NAME_PREFIX', 'ldm_step')

    def after_iter(self, solver):
        if solver.total_iter != 0 and (
            (solver.total_iter + 1) % self.interval == 0
                or solver.total_iter == solver.max_steps - 1):
            state_dict, metadata = solver.save_safetensors()
            solver.logger.info(
                f'Saving safetensors after {solver.total_iter + 1} steps')
            if we.rank == 0:
                save_path = osp.join(
                    solver.work_dir, 'safetensors/{}-{}.safetensors'.format(
                        self.save_name_prefix, solver.total_iter + 1))
                with FS.put_to(save_path) as local_path:
                    with open(local_path, 'wb'):
                        save_file(state_dict, local_path, metadata)
            torch.cuda.synchronize()
            if we.is_distributed:
                torch.distributed.barrier()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            SafetensorsHook.para_dict,
                            set_name=True)
