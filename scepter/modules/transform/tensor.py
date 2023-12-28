# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch

from scepter.modules.transform.registry import TRANSFORMS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, list):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'Unsupported type {type(data)}')


@TRANSFORMS.register_class()
class ToTensor(object):
    def __init__(self, cfg, logger=None):
        self.keys = cfg.KEYS

    def __call__(self, item):
        for key in self.keys:
            item[key] = to_tensor(item[key])
        return item

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
        para_dict = [{'KEYS': {'value': [], 'description': 'keys'}}]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class Select(object):
    def __init__(self, cfg, logger=None):
        self.keys = cfg.KEYS
        meta_keys = cfg.get('META_KEYS', [])
        if not isinstance(meta_keys, (list, tuple)):
            raise TypeError(
                f'Expected meta_keys to be list or tuple, got {type(meta_keys)}'
            )
        self.meta_keys = meta_keys

    def __call__(self, item):
        data = {}
        for key in self.keys:
            data[key] = item[key]
        if 'meta' in item and len(self.meta_keys) > 0:
            data['meta'] = {}
            for key in self.meta_keys:
                data['meta'][key] = item['meta'][key]
        return data

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
        para_dict = [{
            'KEYS': {
                'value': [],
                'description': 'keys'
            },
            'META_KEYS': {
                'value': [],
                'description': 'meta keys'
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class Rename(object):
    def __init__(self, cfg, logger=None):
        self.in_keys = cfg.IN_KEYS
        self.out_keys = cfg.OUT_KEYS

    def __call__(self, item):
        data = {}
        for idx, key in enumerate(self.in_keys):
            data[self.out_keys[idx]] = item[key]
        have_key_set = set(self.in_keys)
        for k, v in item.items():
            if k not in have_key_set:
                data[k] = v
        return data

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
        para_dict = [{
            'IN_KEYS': {
                'value': [],
                'description':
                'The keys need to rename, the other keys are outputed by default.'
            },
            'OUT_KEYS': {
                'value': [],
                'description':
                'The keys need to rename, the other keys are outputed by default.'
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class TensorToGPU(object):
    def __init__(self, cfg, logger=None):
        self.keys = cfg.KEYS
        self.device_id = we.rank

    def __call__(self, item):
        ret = {}
        for key, value in item.items():
            if key in self.keys and isinstance(
                    value, torch.Tensor) and torch.cuda.is_available():
                ret[key] = value.cuda(self.device_id, non_blocking=True)
            else:
                ret[key] = value
        return ret

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
        para_dict = [{
            'KEYS': {
                'value': [],
                'description': 'keys'
            },
            'DEVICE_ID': {
                'value':
                0,
                'description':
                "device id, which should be set according to current GPU's rank"
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)
