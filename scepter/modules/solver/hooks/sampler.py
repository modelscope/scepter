# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml

_DEFAULT_SAMPLER_PRIORITY = 400


@HOOKS.register_class()
class DistSamplerHook(Hook):
    """ DistributedDataSampler needs to set_epoch to shuffle sample indexes
    """
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_SAMPLER_PRIORITY,
            'description': 'the priority for processing!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(DistSamplerHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_SAMPLER_PRIORITY)

    def before_epoch(self, solver):
        for name, data_ins in solver.datas.items():
            if name == 'train':
                data_loader = data_ins.dataloader
                solver.logger.info(
                    f'distribute sampler set_epoch to {solver.epoch}')
                if hasattr(data_loader.sampler, 'set_epoch'):
                    data_loader.sampler.set_epoch(solver.epoch)
                elif hasattr(data_loader.batch_sampler.sampler, 'set_epoch'):
                    data_loader.batch_sampler.sampler.set_epoch(solver.epoch)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            DistSamplerHook.para_dict,
                            set_name=True)
