# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings

import torch

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml

_DEFAULT_BACKWARD_PRIORITY = 0


@HOOKS.register_class()
class BackwardHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_BACKWARD_PRIORITY,
            'description': 'the priority for processing!'
        },
        'GRADIENT_CLIP': {
            'value': -1,
            'description': 'the gradient clip max_norm value for parameters!'
        },
        'ACCUMULATE_STEP': {
            'value':
            1,
            'description':
            'the gradient accumulate steps for backward step, default is 1'
        },
        'EMPTY_CACHE_STEP': {
            'value': -1,
            'description': 'the memory empty step!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(BackwardHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_BACKWARD_PRIORITY)
        self.gradient_clip = cfg.get('GRADIENT_CLIP', -1)
        self.empty_cache_step = cfg.get('EMPTY_CACHE_STEP', -1)
        self.accumulate_step = cfg.get('ACCUMULATE_STEP', 1)
        self.current_step = 0

    def grad_clip(self, parameters):
        torch.nn.utils.clip_grad_norm_(parameters=parameters,
                                       max_norm=self.gradient_clip,
                                       norm_type=2)

    def after_iter(self, solver):
        if (hasattr(solver, 'use_fsdp')
                and solver.use_fsdp) and self.accumulate_step > 1:
            self.logger.info("Fsdp don't surpport gradient accumulate.")
            self.accumulate_step = 1
        if solver.optimizer is not None and solver.is_train_mode:
            if solver.loss is None:
                warnings.warn(
                    'solver.loss should not be None in train mode, remember to call solver._reduce_scalar()!'
                )
                return
            if solver.scaler is not None:
                solver.scaler.scale(solver.loss).backward()
                if self.gradient_clip > 0:
                    solver.scaler.unscale_(solver.optimizer)
                    self.grad_clip(solver.train_parameters())
                self.current_step += 1
                if self.current_step % self.accumulate_step == 0:
                    solver.scaler.step(solver.optimizer)
                    solver.scaler.update()
                    solver.optimizer.zero_grad()
            else:
                solver.loss.backward()
                if self.gradient_clip > 0:
                    self.grad_clip(solver.train_parameters())
                self.current_step += 1
                if self.current_step % self.accumulate_step == 0:
                    solver.optimizer.step()
                    solver.optimizer.zero_grad()
            if solver.lr_scheduler:
                if self.current_step % self.accumulate_step == 0:
                    solver.lr_scheduler.step()
            if self.current_step % self.accumulate_step == 0:
                self.current_step = 0
            solver.loss = None
        if self.empty_cache_step > 0 and solver.total_iter % self.empty_cache_step == 0:
            torch.cuda.empty_cache()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            BackwardHook.para_dict,
                            set_name=True)
