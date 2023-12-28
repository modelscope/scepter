# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from scepter.modules.opt.lr_schedulers import LR_SCHEDULERS
from scepter.modules.opt.lr_schedulers.base_scheduler import BaseScheduler
from scepter.modules.utils.config import dict_to_yaml


@LR_SCHEDULERS.register_class()
class WarmupToConstantLR(BaseScheduler):
    para_dict = {
        'WARMUP_STEPS': {
            'value': 10000,
            'description': 'warmup steps'
        }
    }

    def __init__(self, cfg, logger=None):
        super(WarmupToConstantLR, self).__init__(cfg, logger=logger)
        warmup_steps = cfg.get('WARMUP_STEPS', 10000)
        self.warmup_func = lambda step: min(1.0, step / warmup_steps)

    def __call__(self, optimizer):
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=[self.warmup_func])

    @staticmethod
    def get_config_template():
        return dict_to_yaml('LR_SCHEDULERS',
                            __class__.__name__,
                            WarmupToConstantLR.para_dict,
                            set_name=True)


class AnnealingLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_steps,
                 total_steps,
                 decay_mode='cosine',
                 min_lr=0.0,
                 last_step=-1):
        assert decay_mode in ['linear', 'cosine', 'none']
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                group.setdefault('initial_lr', group['lr'])
        self.base_lrs = [
            group['initial_lr'] for group in optimizer.param_groups
        ]
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_mode = decay_mode
        self.min_lr = min_lr
        self.current_step = last_step + 1
        self.step(self.current_step)

    def get_lr(self):
        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            return [
                base_lr * self.current_step / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            ratio = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps)
            ratio = min(1.0, max(0.0, ratio))
            if self.decay_mode == 'linear':
                return [base_lr * (1 - ratio) for base_lr in self.base_lrs]
            elif self.decay_mode == 'cosine':
                return [
                    base_lr * (math.cos(math.pi * ratio) + 1.0) / 2.0
                    for base_lr in self.base_lrs
                ]
            else:
                return self.base_lrs

    def step(self, current_step=None):
        if current_step is None:
            current_step = self.current_step + 1
        self.current_step = current_step
        new_lrs = self.get_lr()
        new_lrs = [max(self.min_lr, new_lr) for new_lr in new_lrs]
        for new_lr, group in zip(new_lrs, self.optimizer.param_groups):
            group['lr'] = new_lr

    def state_dict(self):
        return {
            'base_lrs': self.base_lrs,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'decay_mode': self.decay_mode,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict['base_lrs']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.decay_mode = state_dict['decay_mode']
        self.current_step = state_dict['current_step']


@LR_SCHEDULERS.register_class()
class StepAnnealingLR(BaseScheduler):
    para_dict = {
        'WARMUP_STEPS': {
            'value': 0,
            'description': 'Setting warmup steps.'
        },
        'TOTAL_STEPS': {
            'value': 10000000,
            'description': 'The total training steps.'
        },
        'DECAY_MODE': {
            'value': 'cosine',
            'description': 'The lr decay mode, default is cosine.'
        },
        'MIN_LR': {
            'value': 0.0,
            'description': 'The minimum learning rate, default is 0.0.'
        },
        'LAST_STEP': {
            'value': -1,
            'description':
            'The the last step before last runing, default is -1.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(StepAnnealingLR, self).__init__(cfg, logger=logger)
        self.warmup_steps = cfg.WARMUP_STEPS
        self.total_steps = cfg.TOTAL_STEPS
        self.decay_mode = cfg.get('DECAY_MODE', 'cosine')
        self.min_lr = cfg.get('MIN_LR', 0.0)
        self.last_step = cfg.get('LAST_STEP', -1)

    def __call__(self, optimizer):
        return AnnealingLR(optimizer,
                           warmup_steps=self.warmup_steps,
                           total_steps=self.total_steps,
                           decay_mode=self.decay_mode,
                           min_lr=self.min_lr,
                           last_step=self.last_step)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('LR_SCHEDULERS',
                            __class__.__name__,
                            StepAnnealingLR.para_dict,
                            set_name=True)
