# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import warnings

import torch
from scepter.modules.utils.file_system import FS

from scepter.modules.utils.distribute import we

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
        },
        'DO_PROFILE': {
            'value': False,
            'description': 'whether to do profiling!'
        },
        'PROFILE_DIR': {
            'value': None,
            'description': 'the dir for profiling!'
        },
        'PROFILE_WAIT': {
            'value': 1,
            'description': 'the wait steps for profiling!'
        },
        'PROFILE_WARMUP': {
            'value': 1,
            'description': 'the warmup steps for profiling!'
        },
        'PROFILE_ACTIVE': {
            'value': 3,
            'description': 'the active steps for profiling!'
        },
        'REPEAT': {
            'value': 1,
            'description': 'the repeat steps for profiling!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(BackwardHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_BACKWARD_PRIORITY)
        self.gradient_clip = cfg.get('GRADIENT_CLIP', -1)
        self.empty_cache_step = cfg.get('EMPTY_CACHE_STEP', -1)
        self.accumulate_step = cfg.get('ACCUMULATE_STEP', 1)
        self.current_step = 0
        self.wait = cfg.get('PROFILE_WAIT', 1)
        self.warmup = cfg.get('PROFILE_WARMUP', 1)
        self.active = cfg.get('PROFILE_ACTIVE', 3)
        self.repeat = cfg.get('REPEAT', 1)
        self.do_profile = cfg.get('DO_PROFILE', False)
        self.profile_dir = cfg.get('PROFILE_DIR', None)
        self.profile_step = 0
        self.prof = None

    def before_solve(self, solver):
        if we.rank != 0:
            return
        if self.profile_dir is None:
            self.log_dir = os.path.join(solver.work_dir, 'profile')
        self._local_log_dir, _ = FS.map_to_local(self.log_dir)
        os.makedirs(self._local_log_dir, exist_ok=True)
        if self.do_profile:
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self._local_log_dir),
                record_shapes=True,
                with_stack=True)
            self.prof.start()
            solver.logger.info(f'Profiler start ...')
            solver.logger.info(f'Profiler: save to {self.log_dir}')
    def profile(self, solver):
        if self.prof is None: return
        if we.rank == 0 and self.do_profile:
            if self.profile_step < self.wait + self.warmup + self.active:
                self.prof.step()
                self.profile_step += 1
            else:
                self.prof.stop()
                self.do_profile = False
                solver.logger.info(f'Profiler stop after {self.profile_step} steps')
                FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)
    def grad_clip(self, parameters):
        torch.nn.utils.clip_grad_norm_(parameters=parameters,
                                       max_norm=self.gradient_clip,
                                       norm_type=2)

    def after_iter(self, solver):
        if solver.optimizer is not None and solver.is_train_mode:
            if solver.loss is None:
                warnings.warn(
                    'solver.loss should not be None in train mode, remember to call solver._reduce_scalar()!'
                )
                return
            if solver.scaler is not None:
                solver.scaler.scale(solver.loss/self.accumulate_step).backward()
                if self.gradient_clip > 0:
                    solver.scaler.unscale_(solver.optimizer)
                    self.grad_clip(solver.train_parameters())
                self.current_step += 1
                # Suppose profiler run after backward, so we need to set backward_prev_step
                # as the previous one step before the backward step
                if self.current_step % self.accumulate_step == 0:
                    self.profile(solver)
                    solver.scaler.step(solver.optimizer)
                    solver.scaler.update()
                    solver.optimizer.zero_grad()
            else:
                (solver.loss/self.accumulate_step).backward()
                if self.gradient_clip > 0:
                    self.grad_clip(solver.train_parameters())
                self.current_step += 1
                # Suppose profiler run after backward, so we need to set backward_prev_step
                # as the previous one step before the backward step
                if self.current_step % self.accumulate_step == 0:
                    self.profile(solver)
                    solver.optimizer.step()
                    solver.optimizer.zero_grad()
            if solver.lr_scheduler:
                if self.current_step % self.accumulate_step == 0:
                    solver.lr_scheduler.step()
            if self.current_step % self.accumulate_step == 0:
                setattr(solver, 'backward_step', True)
                self.current_step = 0
            else:
                setattr(solver, 'backward_step', False)
            solver.loss = None
        if self.empty_cache_step > 0 and solver.total_iter % self.empty_cache_step == 0:
            torch.cuda.empty_cache()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            BackwardHook.para_dict,
                            set_name=True)
