# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.optim.lr_scheduler as lr_sch

from scepter.modules.opt.lr_schedulers.base_scheduler import BaseScheduler
from scepter.modules.opt.lr_schedulers.registry import LR_SCHEDULERS
from scepter.modules.utils.config import dict_to_yaml

SUPPORT_TYPES = ('StepLR', 'CyclicLR', 'LambdaLR', 'MultiStepLR',
                 'ExponentialLR', 'CosineAnnealingLR',
                 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau')


@LR_SCHEDULERS.register_class()
class StepLR(BaseScheduler):
    para_dict = {
        'STEP_SIZE': {
            'value': 1,
            'description': 'the epoch step size!'
        },
        'GAMMA': {
            'value': 0.1,
            'description': 'the gamma!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(StepLR, self).__init__(cfg, logger=logger)
        self.step_size = cfg.STEP_SIZE
        self.gamma = cfg.get('GAMMA', 0.1)
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return lr_sch.StepLR(optimizer,
                             step_size=self.step_size,
                             gamma=0.1,
                             last_epoch=-1)

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
                            StepLR.para_dict,
                            set_name=True)


@LR_SCHEDULERS.register_class()
class CyclicLR(BaseScheduler):
    para_dict = {
        'BASE_LR': {
            'value': 0.1,
            'description': 'the base lr!'
        },
        'MAX_LR': {
            'value': 0.5,
            'description': 'the max lr!'
        },
        'STEP_SIZE_UP': {
            'value': 2000,
            'description': 'the step size up!'
        },
        'STEP_SIZE_DOWN': {
            'value': None,
            'description': 'the step size down!'
        },
        'MODE': {
            'value': 'triangular',
            'description': 'the mode triangular!'
        },
        'GAMMA': {
            'value': 1,
            'description': 'the gamma!'
        },
        'SCALE_FN': {
            'value': None,
            'description': 'the scale fn!'
        },
        'SCALE_MODE': {
            'value': 'cycle',
            'description': 'the scale mode!'
        },
        'CYCLE_MOMENTUM': {
            'value': True,
            'description': 'the cycle momentum!'
        },
        'BASE_MOMENTUM': {
            'value': 0.8,
            'description': 'the base momentum!'
        },
        'MAX_MOMENTUM': {
            'value': 0.9,
            'description': 'the max momentum!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(CyclicLR, self).__init__(cfg, logger=logger)
        self.base_lr = cfg.BASE_LR
        self.max_lr = cfg.MAX_LR
        self.step_size_up = cfg.get('STEP_SIZE_UP', 2000)
        self.step_size_down = cfg.get('STEP_SIZE_DOWN', None)
        self.mode = cfg.get('MODE', 'triangular')
        self.gamma = cfg.get('GAMMA', 1)
        self.scale_fn = cfg.get('SCALE_FN', None)
        self.scale_mode = cfg.get('SCALE_MODE', 'cycle')
        self.cycle_momentum = cfg.get('CYCLE_MOMENTUM', True)
        self.base_momentum = cfg.get('BASE_MOMENTUM', 0.8)
        self.max_momentum = cfg.get('MAX_MOMENTUM', 0.9)
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return lr_sch.CyclicLR(optimizer,
                               base_lr=self.base_lr,
                               max_lr=self.max_lr,
                               step_size_up=self.step_size_up,
                               step_size_down=self.step_size_down,
                               mode=self.mode,
                               gamma=self.gamma,
                               scale_fn=self.scale_fn,
                               scale_mode=self.scale_mode,
                               cycle_momentum=self.cycle_momentum,
                               base_momentum=self.base_momentum,
                               max_momentum=self.max_momentum,
                               last_epoch=self.last_epoch)

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
                            CyclicLR.para_dict,
                            set_name=True)


@LR_SCHEDULERS.register_class()
class LambdaLR(BaseScheduler):
    para_dict = {
        'LR_LAMBDA': {
            'value': 0.1,
            'description': 'the lr lambda!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(LambdaLR, self).__init__(cfg, logger=logger)
        self.lr_lambda = cfg.LR_LAMBDA
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return lr_sch.LambdaLR(optimizer,
                               lr_lambda=self.lr_lambda,
                               last_epoch=self.last_epoch)

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
                            LambdaLR.para_dict,
                            set_name=True)


@LR_SCHEDULERS.register_class()
class MultiStepLR(BaseScheduler):
    para_dict = {
        'MILESTONES': {
            'value': [10000, 20000],
            'description': 'the lr lambda!'
        },
        'GAMMA': {
            'value': 0.1,
            'description': 'the gamma!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(MultiStepLR, self).__init__(cfg, logger=logger)
        self.milestones = cfg.MILESTONES
        self.gamma = cfg.get('GAMMA', 0.1)
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return lr_sch.MultiStepLR(optimizer,
                                  milestones=self.milestones,
                                  gamma=self.gamma,
                                  last_epoch=self.last_epoch)

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
                            MultiStepLR.para_dict,
                            set_name=True)


@LR_SCHEDULERS.register_class()
class ExponentialLR(BaseScheduler):
    para_dict = {
        'GAMMA': {
            'value': 0.1,
            'description': 'the gamma!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ExponentialLR, self).__init__(cfg, logger=logger)
        self.gamma = cfg.get('GAMMA', 0.1)
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return lr_sch.ExponentialLR(optimizer,
                                    gamma=self.gamma,
                                    last_epoch=self.last_epoch)

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
                            ExponentialLR.para_dict,
                            set_name=True)


@LR_SCHEDULERS.register_class()
class CosineAnnealingLR(BaseScheduler):
    para_dict = {
        'T_MAX': {
            'value': 1.0,
            'description': 'the T max!'
        },
        'ETA_MIN': {
            'value': 0,
            'description': 'the eta min!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(CosineAnnealingLR, self).__init__(cfg, logger=logger)
        self.T_max = cfg.T_MAX
        self.eta_min = cfg.get('ETA_MIN', 0)
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return lr_sch.CosineAnnealingLR(optimizer,
                                        T_max=self.T_max,
                                        eta_min=self.eta_min,
                                        last_epoch=self.last_epoch)

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
                            CosineAnnealingLR.para_dict,
                            set_name=True)


@LR_SCHEDULERS.register_class()
class CosineAnnealingWarmRestarts(BaseScheduler):
    para_dict = {
        'T_0': {
            'value': 1.0,
            'description': 'the T 0!'
        },
        'T_MULT': {
            'value': 1.0,
            'description': 'the T mult!'
        },
        'ETA_MIN': {
            'value': 1.0,
            'description': 'the eta min!'
        },
        'LAST_EPOCH': {
            'value': -1,
            'description': 'the last epoch!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(CosineAnnealingWarmRestarts, self).__init__(cfg, logger=logger)
        self.T_0 = cfg.T_0
        self.T_mult = cfg.get('T_MULT', 1.0)
        self.eta_min = cfg.get('ETA_MIN', 1.0)
        self.last_epoch = cfg.get('LAST_EPOCH', -1)

    def __call__(self, optimizer):
        return lr_sch.CosineAnnealingWarmRestarts(optimizer,
                                                  T_0=self.T_0,
                                                  T_mult=self.T_mult,
                                                  eta_min=self.eta_min,
                                                  last_epoch=self.last_epoch)

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
                            CosineAnnealingWarmRestarts.para_dict,
                            set_name=True)


@LR_SCHEDULERS.register_class()
class ReduceLROnPlateau(BaseScheduler):
    para_dict = {
        'MODE': {
            'value': 'min',
            'description': 'the mode!'
        },
        'FACTOR': {
            'value': 0.1,
            'description': 'the factor!'
        },
        'PATIENCE': {
            'value': 10,
            'description': 'the patience!'
        },
        'THRESHOLD': {
            'value': 1e-4,
            'description': 'the threshold!'
        },
        'THRESHOLD_MODE': {
            'value': 'rel',
            'description': 'the threshold mode!'
        },
        'COOLDOWN': {
            'value': 0,
            'description': 'the cooldown!'
        },
        'MIN_LR': {
            'value': 0,
            'description': 'the min lr!'
        },
        'EPS': {
            'value': 1e-8,
            'description': 'the eps!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ReduceLROnPlateau, self).__init__(cfg, logger=logger)
        self.model = cfg.get('MODE', 'min')
        self.factor = cfg.get('FACTOR', 0.1)
        self.patience = cfg.get('PATIENCE', 10)
        self.threshold = cfg.get('THRESHOLD', 1e-4)
        self.threshold_mode = cfg.get('THRESHOLD_MODE', 'rel')
        self.cooldown = cfg.get('COOLDOWN', 0)
        self.min_lr = cfg.get('MIN_LR', 0)
        self.eps = cfg.get('EPS', 1e-8)

    def __call__(self, optimizer):
        return lr_sch.ReduceLROnPlateau(optimizer,
                                        mode=self.mode,
                                        factor=self.factor,
                                        patience=self.patience,
                                        threshold=self.threshold,
                                        threshold_mode=self.threshold_mode,
                                        cooldown=self.cooldown,
                                        min_lr=self.min_lr,
                                        eps=self.eps)

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
                            ReduceLROnPlateau.para_dict,
                            set_name=True)
