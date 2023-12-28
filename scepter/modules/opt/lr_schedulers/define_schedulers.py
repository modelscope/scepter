# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from torch.optim.lr_scheduler import _LRScheduler

from scepter.modules.opt.lr_schedulers.base_scheduler import BaseScheduler
from scepter.modules.opt.lr_schedulers.registry import LR_SCHEDULERS
from scepter.modules.utils.config import dict_to_yaml


class PolyLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        power (float): Multiplicative factor of learning rate decay.
        end_epoch (int): Total epoches for the task.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    def __init__(self,
                 optimizer,
                 power,
                 end_epoch,
                 last_epoch=-1,
                 verbose=False):
        self.power = power
        self.end_epoch = end_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        assert self.last_epoch >= 0
        current_epoch = self.last_epoch
        if self.last_epoch > self.end_epoch:
            current_epoch = self.end_epoch
        factor = (1 - current_epoch / self.end_epoch)**self.power
        return [base_lr * factor for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        assert self.last_epoch >= 0
        current_epoch = self.last_epoch
        if self.last_epoch > self.end_epoch:
            current_epoch = self.end_epoch
        factor = (1 - current_epoch / self.end_epoch)**self.power
        return [base_lr * factor for base_lr in self.base_lrs]


@LR_SCHEDULERS.register_class()
class LinoPolyLR(BaseScheduler):
    para_dict = {
        'END_EPOCH': {
            'value': 1,
            'description': 'The total epoches!'
        },
        'POWER': {
            'value': 1,
            'description': 'the lr decay rate!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(LinoPolyLR, self).__init__(cfg, logger=logger)
        self.power = cfg.get('POWER', 1)
        self.end_epoch = cfg.END_EPOCH
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return PolyLR(optimizer, self.power, self.end_epoch, last_epoch=-1)

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
        return dict_to_yaml('LR_SCHEDULERS',
                            __class__.__name__,
                            LinoPolyLR.para_dict,
                            set_name=True)
