# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
from PIL import Image

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


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, (int, float, list, tuple, dict, Image.Image)):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
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
class ToNumpy(object):
    def __init__(self, cfg, logger=None):
        self.input_key = cfg.get('INPUT_KEY', 'img')
        self.output_key = cfg.get('OUTPUT_KEY', 'img')

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            item[self.output_key[idx]] = to_numpy(item[key])
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
        para_dict = [{
            'INPUT_KEY': {
                'value': [],
                'description': 'input_key'
            },
            'OUTPUT_KEY': {
                'value': [],
                'description': 'output_key'
            }
        }]
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
        self.input_key = cfg.INPUT_KEY
        self.output_key = cfg.OUTPUT_KEY

    def __call__(self, item):
        data = {}
        for idx, key in enumerate(self.input_key):
            data[self.output_key[idx]] = item[key]
        have_key_set = set(self.input_key)
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
            'INPUT_KEY': {
                'value': [],
                'description':
                'The keys need to rename, the other keys are outputed by default.'
            },
            'OUTPUT_KEY': {
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


@TRANSFORMS.register_class()
class RenameMeta(object):
    def __init__(self, cfg, logger=None):
        self.input_key = cfg.INPUT_KEY
        self.output_key = cfg.OUTPUT_KEY
        self.force = cfg.get('FORCE', False)
        self.move = cfg.get('MOVE', False)

    def __call__(self, item):
        if 'meta' in item:
            data = {}
            for idx, key in enumerate(self.input_key):
                data[self.output_key[idx]] = item['meta'][key]
            if not self.force:
                have_key_set = set(self.input_key)
            else:
                have_key_set = set(self.input_key + self.output_key)
            if not self.move:
                for k, v in item['meta'].items():
                    if k not in have_key_set:
                        data[k] = v
                item['meta'] = data
            else:
                for k, v in item.items():
                    if k not in have_key_set:
                        data[k] = v
                item.update(data)
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
        para_dict = [{
            'INPUT_KEY': {
                'value': [],
                'description':
                'The keys need to rename, the other keys are outputed by default.'
            },
            'OUTPUT_KEY': {
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
class TemplateStr(object):
    def __init__(self, cfg, logger=None):
        self.template_str = cfg.get('TEMPLATE_STR', '')
        self.meta_template_str = cfg.get('META_TEMPLATE_STR', '')

    def __call__(self, item):
        if self.template_str != '':
            for key, val in item.items():
                if isinstance(val, str) and f'{{{key}}}' in self.template_str:
                    template = self.template_str
                    val = template.replace(f'{{{key}}}', val)
                    item[key] = val
        if self.meta_template_str != '' and 'meta' in item:
            for key, val in item['meta'].items():
                if isinstance(val,
                              str) and f'{{{key}}}' in self.meta_template_str:
                    template = self.meta_template_str
                    val = template.replace(f'{{{key}}}', val)
                    item['meta'][key] = val
        return item

    @staticmethod
    def get_config_template():
        para_dict = [{}]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)
