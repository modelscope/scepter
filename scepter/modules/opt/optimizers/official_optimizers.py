# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.optim as optim

from scepter.modules.opt.optimizers.base_optimizer import BaseOptimize
from scepter.modules.opt.optimizers.registry import OPTIMIZERS
from scepter.modules.utils.config import dict_to_yaml

SUPPORT_TYPES = ('Adadelta', 'Adagrad', 'Adam', 'Adamax', 'AdamW', 'ASGD',
                 'LBFGS', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam')


@OPTIMIZERS.register_class()
class SGD(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 0.1,
            'description': 'the initial learning rate!'
        },
        'MOMENTUM': {
            'value': 0,
            'description': 'the momentum!'
        },
        'DAMPENING': {
            'value': 0,
            'description': 'the dampening!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        },
        'NESTEROV': {
            'value': False,
            'description': 'the nesterov!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(SGD, self).__init__(cfg, logger=logger)
        self.lr = cfg.LEARNING_RATE
        self.momentum = cfg.get('MOMENTUM', 0)
        self.dampening = cfg.get('DAMPENING', 0)
        self.weight_decay = cfg.get('WEIGHT_DECAY', 0)
        self.nesterov = cfg.get('NESTEROV', False)

    def __call__(self, parameters):
        return optim.SGD(parameters,
                         lr=self.lr,
                         momentum=self.momentum,
                         dampening=self.dampening,
                         weight_decay=self.weight_decay,
                         nesterov=self.nesterov)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            SGD.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class Adadelta(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1.0,
            'description': 'the initial learning rate!'
        },
        'RHO': {
            'value': 0,
            'description': 'the rho!'
        },
        'EPS': {
            'value': 1e-6,
            'description': 'the eps!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(Adadelta, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1.0)
        self.rho = cfg.get('RHO', 0.0)
        self.eps = cfg.get('EPS', 1e-6)
        self.weight_decay = cfg.get('WEIGHT_DECAY', 0)

    def __call__(self, parameters):
        return optim.Adadelta(parameters,
                              lr=self.lr,
                              rho=self.rho,
                              eps=self.eps,
                              weight_decay=self.weight_decay)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            Adadelta.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class Adagrad(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1e-2,
            'description': 'the initial learning rate!'
        },
        'LEARNING_RATE_DECAY': {
            'value': 0,
            'description': 'the lr decay!'
        },
        'EPS': {
            'value': 1e-10,
            'description': 'the eps!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        },
        'INITIAL_ACCUMULATOR_VALUE': {
            'value': 0,
            'description': 'the initial accumulator value!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(Adagrad, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1e-2)
        self.lr_decay = cfg.get('LEARNING_RATE_DECAY', 0.0)
        self.weight_decay = cfg.get('WEIGHT_DECAY', 0)
        self.initial_accumulator_value = cfg.get('INITIAL_ACCUMULATOR_VALUE',
                                                 0)
        self.eps = cfg.get('EPS', 1e-10)

    def __call__(self, parameters):
        return optim.Adagrad(
            parameters,
            lr=self.lr,
            lr_decay=self.lr_decay,
            weight_decay=self.weight_decay,
            initial_accumulator_value=self.initial_accumulator_value,
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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            Adagrad.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class Adam(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1e-2,
            'description': 'the initial learning rate!'
        },
        'BETAS': {
            'value': [0.9, 0.999],
            'description': 'the rho!'
        },
        'EPS': {
            'value': 1e-6,
            'description': 'the eps!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        },
        'AMSGRAD': {
            'value': False,
            'description': 'the amsgrad!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(Adam, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1e-2)
        self.betas = cfg.get('BETAS', [0.9, 0.999])
        self.weight_decay = cfg.get('WEIGHT_DECAY', 0)
        self.amsgrad = cfg.get('AMSGRAD', False)
        self.eps = cfg.get('EPS', 1e-10)

    def __call__(self, parameters):
        return optim.Adam(parameters,
                          lr=self.lr,
                          betas=tuple(self.betas),
                          eps=self.eps,
                          weight_decay=self.weight_decay,
                          amsgrad=self.amsgrad)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            Adam.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class Adamax(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 2e-3,
            'description': 'the initial learning rate!'
        },
        'BETAS': {
            'value': [0.9, 0.999],
            'description': 'the rho!'
        },
        'EPS': {
            'value': 1e-8,
            'description': 'the eps!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(Adamax, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 2e-3)
        self.betas = cfg.get('BETAS', [0.9, 0.999])
        self.weight_decay = cfg.get('WEIGHT_DECAY', 0)
        self.eps = cfg.get('EPS', 1e-8)

    def __call__(self, parameters):
        return optim.Adamax(parameters,
                            lr=self.lr,
                            betas=tuple(self.betas),
                            eps=self.eps,
                            weight_decay=self.weight_decay)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            Adamax.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class AdamW(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1e-3,
            'description': 'the initial learning rate!'
        },
        'BETAS': {
            'value': [0.9, 0.999],
            'description': 'the rho!'
        },
        'EPS': {
            'value': 1e-8,
            'description': 'the eps!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        },
        'AMSGRAD': {
            'value': False,
            'description': 'the amsgrad!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(AdamW, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1e-3)
        self.betas = cfg.get('BETAS', [0.9, 0.999])
        self.weight_decay = cfg.get('WEIGHT_DECAY', 0)
        self.eps = cfg.get('EPS', 1e-8)
        self.amsgrad = cfg.get('AMSGRAD', False)

    def __call__(self, parameters):
        return optim.AdamW(parameters,
                           lr=self.lr,
                           betas=tuple(self.betas),
                           eps=self.eps,
                           weight_decay=self.weight_decay,
                           amsgrad=self.amsgrad)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            AdamW.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class ASGD(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1e-2,
            'description': 'the initial learning rate!'
        },
        'LAMBD': {
            'value': 1e-4,
            'description': 'the rho!'
        },
        'ALPHA': {
            'value': 0.75,
            'description': 'the alpha!'
        },
        'T0': {
            'value': 1e6,
            'description': 'the t0!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ASGD, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1e-2)
        self.lambd = cfg.get('LAMBD', 1e-4)
        self.alpha = cfg.get('ALPHA', 0.75)
        self.t0 = cfg.get('T0', 1e6)
        self.weight_decay = cfg.get('WEIGHT_DECAY', 0)

    def __call__(self, parameters):
        return optim.ASGD(parameters,
                          lr=self.lr,
                          lambd=self.lambd,
                          alpha=self.alpha,
                          t0=self.t0,
                          weight_decay=self.weight_decay)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            ASGD.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class LBFGS(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1.0,
            'description': 'the initial learning rate!'
        },
        'MAX_ITER': {
            'value': 20,
            'description': 'the max iter!'
        },
        'MAX_EVAL': {
            'value': None,
            'description': 'the max eval!'
        },
        'TOLERANCE_GRAD': {
            'value': 1e-7,
            'description': 'the tolerance grad!'
        },
        'TOLERANCE_CHANGE': {
            'value': 1e-9,
            'description': 'the tolerance change!'
        },
        'HISTORY_SIZE': {
            'value': 100,
            'description': 'the history size!'
        },
        'LINE_SEARCH_FN': {
            'value': None,
            'description': 'the line search fn!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(LBFGS, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1)
        self.max_iter = cfg.get('MAX_ITER', 20)
        self.max_eval = cfg.get('MAX_EVAL', None)
        self.tolerance_grad = cfg.get('TOLERANCE_GRAD', 1e-7)
        self.tolerance_change = cfg.get('TOLERANCE_CHANGE', 1e-9)
        self.history_size = cfg.get('HISTORY_SIZE', 100)
        self.line_search_fn = cfg.get('LINE_SEARCH_FN', None)

    def __call__(self, parameters):
        return optim.LBFGS(parameters,
                           lr=self.lr,
                           max_iter=self.max_iter,
                           max_eval=self.max_eval,
                           tolerance_grad=self.tolerance_grad,
                           tolerance_change=self.tolerance_change,
                           history_size=self.history_size,
                           line_search_fn=self.line_search_fn)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            LBFGS.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class RMSprop(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1e-2,
            'description': 'the initial learning rate!'
        },
        'ALPHA': {
            'value': 0.99,
            'description': 'the alpha!'
        },
        'EPS': {
            'value': 1e-8,
            'description': 'the eps!'
        },
        'WEIGHT_DECAY': {
            'value': 0,
            'description': 'the weight decay!'
        },
        'MOMENTUM': {
            'value': 0,
            'description': 'the momentum!'
        },
        'CENTERED': {
            'value': False,
            'description': 'the centered!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(RMSprop, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1e-2)
        self.alpha = cfg.get('ALPHA', 0.99)
        self.eps = cfg.get('EPS', 1e-8)
        self.weight_decay = cfg.get('WEIGHT_DECAY', False)
        self.momentum = cfg.get('MOMENTUM', 0)
        self.centered = cfg.get('CENTERED', False)

    def __call__(self, parameters):
        return optim.RMSprop(parameters,
                             lr=self.lr,
                             alpha=self.alpha,
                             eps=self.eps,
                             weight_decay=self.weight_decay,
                             momentum=self.momentum,
                             centered=self.centered)

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            RMSprop.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class Rprop(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1.0,
            'description': 'the initial learning rate!'
        },
        'ETAS': {
            'value': [0.5, 1.2],
            'description': 'the etas!'
        },
        'STEP_SIZES': {
            'value': [1e-6, 50],
            'description': 'the step sizes!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(Rprop, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1e-2)
        self.etas = cfg.get('ETAS', [0.5, 1.2])
        self.step_sizes = cfg.get('STEP_SIZES', [1e-6, 50])

    def __call__(self, parameters):
        return optim.Rprop(parameters,
                           lr=self.lr,
                           etas=tuple(self.etas),
                           step_sizes=tuple(self.step_sizes))

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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            Rprop.para_dict,
                            set_name=True)


@OPTIMIZERS.register_class()
class SparseAdam(BaseOptimize):
    para_dict = {
        'LEARNING_RATE': {
            'value': 1.0,
            'description': 'the initial learning rate!'
        },
        'BETAS': {
            'value': [0.9, 0.999],
            'description': 'the betas!'
        },
        'EPS': {
            'value': 1e-8,
            'description': 'the eps!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(SparseAdam, self).__init__(cfg, logger=logger)
        self.lr = cfg.get('LEARNING_RATE', 1e-3)
        self.betas = cfg.get('BETAS', [0.9, 0.999])
        self.eps = cfg.get('EPS', 1e-8)

    def __call__(self, parameters):
        return optim.SparseAdam(parameters,
                                lr=self.lr,
                                betas=tuple(self.betas),
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
        return dict_to_yaml('OPTIMIZER',
                            __class__.__name__,
                            SparseAdam.para_dict,
                            set_name=True)
