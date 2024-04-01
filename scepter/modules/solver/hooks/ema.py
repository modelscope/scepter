# -*- coding: utf-8 -*-
import torch
from torch.distributed.fsdp import (FullStateDictConfig,
                                    FullyShardedDataParallel, StateDictType)

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we


@HOOKS.register_class()
class ModelEmaHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': 100,
            'description': 'the priority for processing!'
        },
        'BETA': {
            'value': 0.9999,
            'description': ''
        },
    }]

    def __init__(self, cfg, logger=None):
        super(ModelEmaHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', 100)
        self.beta = cfg.get('BETA', 0.9999)

    def after_iter(self, solver):
        if solver.model.use_ema:
            model_ema = solver.model.model_ema
            model = solver.model.model
            self.ema(model_ema, model, use_fsdp=solver.use_fsdp)

    @torch.no_grad()
    def ema(self, net_ema, net, use_fsdp=True):
        if we.is_distributed:
            if use_fsdp:
                save_policy = FullStateDictConfig(offload_to_cpu=False,
                                                  rank0_only=False)
                with FullyShardedDataParallel.state_dict_type(
                        net, StateDictType.FULL_STATE_DICT, save_policy):
                    nonema_state = net.state_dict()
            elif hasattr(net, 'module'):
                nonema_state = net.module.state_dict()
            else:
                nonema_state = net.state_dict()
        else:
            nonema_state = net.state_dict()

        for k, v in net_ema.named_parameters():
            v.copy_(nonema_state[k].lerp(v, self.beta))

    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            ModelEmaHook.para_dict,
                            set_name=True)
