# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from ...utils.config import dict_to_yaml
from .hook import Hook
from .registry import HOOKS

_DEFAULT_LR_PRIORITY = 200


def _get_lr_from_scheduler(lr_scheduler, cur_epoch):
    """Ugly solution to get lr by epoch.
    PyTorch lr scheduler get_lr() function is recommended to call in step()
    Here we mock the environment.

    Args:
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler):
        cur_epoch (number): int or float (when num_folds > 1)

    Returns:
        Learning rate at cur_epoch.
    """
    lr_scheduler._get_lr_called_within_step = True
    last_epoch_bk = lr_scheduler.last_epoch
    lr_scheduler.last_epoch = cur_epoch
    if hasattr(lr_scheduler, '_get_closed_form_lr'):
        lr = lr_scheduler._get_closed_form_lr()[0]
    else:
        lr = lr_scheduler.get_lr()[0]
    lr_scheduler._get_lr_called_within_step = False
    lr_scheduler.last_epoch = last_epoch_bk
    return lr


@HOOKS.register_class()
class LrHook(Hook):
    """ Learning rate updater hook.
    If warmup, warmup_end_lr will be calculated by lr_scheduler at warmup_epochs.
    Lr in warmup period is set based on warmup_func.
    If set_by_epoch, lr is set at end of epoch. Otherwise, lr is set before training iteration.

    Args:
        set_by_epoch (bool): Reset learning rate by epoch, we recommend true if solver.num_folds == 1
        warmup_func (str, None): Do not warm up if None, currently support linear warmup
        warmup_epochs (int):
        warmup_start_lr (float):
    """
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_LR_PRIORITY,
            'description': 'the priority for processing!'
        },
        'WARMUP_FUNC': {
            'value': 'linear',
            'description': 'Only linear warmup supported!'
        },
        'WARMUP_EPOCHS': {
            'value': 1,
            'description': 'The warmup epochs!'
        },
        'WARMUP_START_LR': {
            'value': 0.0001,
            'description': 'The warmup start learning rate!'
        },
        'SET_BY_EPOCH': {
            'value': True,
            'description': 'Set the learning rate by epoch!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(LrHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_LR_PRIORITY)
        self.warmup_func = cfg.get('WARMUP_FUNC', 'linear')
        if self.warmup_func is not None:
            assert self.warmup_func in (
                'linear', ), 'Only linear warmup supported'
        self.warmup_epochs = cfg.get('WARMUP_EPOCHS', 1)
        self.warmup_start_lr = cfg.get('WARMUP_START_LR', 0.0001)
        self.warmup_end_lr = 0
        self.set_by_epoch = cfg.get('SET_BY_EPOCH', True)

    def before_solve(self, solver):
        if self.warmup_func is not None and self.warmup_epochs > 0:
            self.warmup_end_lr = _get_lr_from_scheduler(
                solver.lr_scheduler, self.warmup_epochs)
            for param_group in solver.optimizer.param_groups:
                param_group['lr'] = self.warmup_start_lr

    def _get_warmup_lr(self, cur_epoch):
        # if self.warmup_func == "linear":
        alpha = (self.warmup_end_lr -
                 self.warmup_start_lr) / self.warmup_epochs
        return self.warmup_start_lr + alpha * cur_epoch

    def after_epoch(self, solver):
        if solver.lr_scheduler is not None and solver.is_train_mode:
            if self.set_by_epoch:
                last_lr = solver.optimizer.param_groups[0]['lr']
                for _ in range(solver.num_folds):
                    solver.lr_scheduler.step()
                new_lr = solver.optimizer.param_groups[0]['lr']
                print(f'now {new_lr}')
                if self.warmup_func is not None and solver.epoch < self.warmup_epochs:
                    new_lr = self._get_warmup_lr(solver.epoch)
                    for param_group in solver.optimizer.param_groups:
                        param_group['lr'] = new_lr
                if last_lr != new_lr:
                    solver.logger.info(
                        f'Change learning rate from {last_lr} to {new_lr}')
                else:
                    solver.logger.info(f'Keep learning rate = {last_lr}')

    def before_iter(self, solver):
        if not self.set_by_epoch and solver.is_train_mode and solver.lr_scheduler is not None:
            cur_epoch_float = solver.epoch + solver.iter / solver.epoch_max_iter - 1
            # solver.logger.info(cur_epoch_float)
            if self.warmup_func is not None and cur_epoch_float < self.warmup_epochs:
                new_lr = self._get_warmup_lr(cur_epoch_float)
            else:
                new_lr = _get_lr_from_scheduler(solver.lr_scheduler,
                                                cur_epoch_float)
            for param_group in solver.optimizer.param_groups:
                param_group['lr'] = new_lr

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            LrHook.para_dict,
                            set_name=True)
