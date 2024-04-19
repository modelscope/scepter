# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import os
import sys
from collections.abc import Iterable

import numpy as np
import torchvision

from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.data.dataset.registry import DATASETS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import DATA_FS as FS


@DATASETS.register_class()
class ImageClassifyPublicDataset(BaseDataset):
    """
    Dataset for image classification wrapper

    Args:
        json_path (str): json file which contains all instances, should be a list of dict
            which contains img_path and gt_label
        image_dir (str or None): image directory, if None, img_path in json_path will be considered as absolute path
        classes (list[str] or None): image class description
    """
    para_dict = {
        'DATASET': {
            'value': 'cifar10',
            'description': 'the public dataset name'
        },
        'DATA_ROOT': {
            'value': '',
            'description': 'the download data save path'
        }
    }

    para_dict.update(BaseDataset.para_dict)

    def __init__(self, cfg, logger=None):

        super(ImageClassifyPublicDataset, self).__init__(cfg, logger=logger)

        self.dataset_name = cfg.DATASET
        self.data_root = cfg.DATA_ROOT
        self.phase = cfg.MODE
        if self.dataset_name == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(
                root=self.data_root,
                train=self.phase == 'train',
                download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get(self, index: int):
        img, target = self.dataset.__getitem__(index)
        ret = {
            'meta': {},
            'label': np.asarray(target, dtype=np.int64),
            'img': img
        }
        return ret

    def worker_init_fn(self, worker_id, num_workers=1):
        super(ImageClassifyPublicDataset,
              self).worker_init_fn(worker_id, num_workers=num_workers)

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
        return dict_to_yaml('modename_DATA',
                            __class__.__name__,
                            ImageClassifyPublicDataset.para_dict,
                            set_name=True)


@DATASETS.register_class()
class ImageTextPairDataset(BaseDataset):
    """
    Dataset for diffusion model training
    """
    para_dict = {
        'P_ZERO': {
            'value': 0.0,
            'description': '',
        },
        'NEGTIVE_PROMPT': {
            'value': '',
            'description': 'The default negtive prompt',
        },
        'DATA_NUM': {
            'value': '',
            'description': '',
        }
    }
    para_dict.update(BaseDataset.para_dict)

    def __init__(self, cfg, logger=None):
        super(ImageTextPairDataset, self).__init__(cfg, logger=logger)
        self.p_zero = cfg.get('P_ZERO', 0.0)
        self.real_number = cfg.get('DATA_NUM', None)
        self._default_item = {
            'meta': {},
            'prompt':
            'Plants in the Water, Nature, Lake, Horizontal, Reflection, Photography, Backgrounds, Swamp, No People'
        }

    def _get(self, index):
        meta = dict()
        # the last item is field_keys
        for key, value in zip(index[-1], index[:-1]):
            if key in ['oss_key', 'path', 'img_path', 'target_img_path']:
                meta['img_path'] = value
            elif key in ['prompt', 'caption', 'text']:
                meta['ori_prompt'] = value
            elif key in ['width', 'height']:
                meta[key] = int(value)
            else:
                meta[key] = value

        prompt = meta.get('prompt_prefix', '') + meta.get('ori_prompt', '')
        if self.mode == 'train' and np.random.uniform() < self.p_zero:
            prompt = ''

        item = {
            'meta': meta,
            'prompt': prompt,
        }
        return item

    def __getitem__(self, index):
        item = self._get(index)
        item = self.pipeline(item)
        return item

    def __len__(self) -> int:
        return sys.maxsize

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DATASETS',
                            __class__.__name__,
                            ImageTextPairDataset.para_dict,
                            set_name=True)


@DATASETS.register_class()
class Image2ImageDataset(BaseDataset):
    """
    Dataset for diffusion model training
    """
    para_dict = {}
    para_dict.update(BaseDataset.para_dict)

    def __init__(self, cfg, logger=None):
        super(Image2ImageDataset, self).__init__(cfg, logger=logger)
        self._default_item = {
            'meta': {},
            'prompt':
            'Plants in the Water, Nature, Lake, Horizontal, Reflection, Photography, Backgrounds, Swamp, No People'
        }

    def _get(self, index):
        meta = dict()
        # the last item is field_keys
        for key, value in zip(index[-1], index[:-1]):
            if key in ['oss_key', 'path', 'img_path']:
                meta['img_path'] = value
            elif key in ['prompt', 'caption', 'text']:
                meta['ori_prompt'] = value
            elif key in ['width', 'height']:
                meta[key] = int(value)
            else:
                meta[key] = value
        item = {'meta': meta}
        return item

    def __getitem__(self, index):
        item = self._get(index)
        item = self.pipeline(item)
        return item

    def __len__(self) -> int:
        return sys.maxsize

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DATASETS',
                            __class__.__name__,
                            Image2ImageDataset.para_dict,
                            set_name=True)


@DATASETS.register_class()
class Text2ImageDataset(BaseDataset):
    para_dict = {
        'PROMPT_FILE': {
            'value': '',
            'description': ''
        },
        'FIELDS': {
            'value': '',
            'description': ''
        },
        'DELIMITER': {
            'value': ',',
            'description': ''
        },
        'PROMPT_PREFIX': {
            'value': '',
            'description': ''
        },
        'IMAGE_SIZE': {
            'value': 512,
            'description': ''
        },
        'USE_NUM': {
            'value': -1,
            'description': ''
        },
    }
    para_dict.update(BaseDataset.para_dict)

    def __init__(self, cfg, logger=None):
        super(Text2ImageDataset, self).__init__(cfg, logger=logger)

        delimiter = cfg.get('DELIMITER', ',')
        fields = cfg.get('FIELDS', ['row_key', 'prompt'])
        prompt_prefix = cfg.get('PROMPT_PREFIX', '')
        path_prefix = cfg.get('PATH_PREFIX', '')
        use_num = cfg.get('USE_NUM', -1)

        image_size = cfg.get('IMAGE_SIZE', 1024)
        if isinstance(image_size, numbers.Number):
            image_size = [image_size, image_size]
        assert isinstance(image_size, Iterable) and len(image_size) == 2

        if cfg.PROMPT_FILE is not None and cfg.PROMPT_FILE != '':
            prompt_file = cfg.PROMPT_FILE
            with FS.get_object(prompt_file) as local_data:
                rows = [
                    i.split(delimiter,
                            len(fields) - 1)
                    for i in local_data.decode('utf-8').strip().split('\n')
                ]
        else:
            rows = [
                i.split(delimiter,
                        len(fields) - 1) for i in cfg.PROMPT_DATA
            ]

        self.items = list()
        for i, row in enumerate(rows):
            item = {'index': i, 'meta': {'image_size': image_size}}
            for key, value in zip(fields, row):
                if key in ['prompt', 'caption', 'text']:
                    item['ori_prompt'] = value
                    item['prompt'] = prompt_prefix + value
                elif key in ['oss_key', 'path', 'img_path', 'target_img_path']:
                    item['meta']['img_path'] = os.path.join(path_prefix, value)
                elif key in ['width', 'height']:
                    item['meta'][key] = int(value)
                else:
                    item['meta'][key] = value

            self.items.append(item)
        if use_num > 0:
            self.items = self.items[:use_num]
        if we.rank == 0:
            logger.info(f'eval prompt num: {len(self.items)}')
            logger.info('eval prompts: {}'.format(
                [k['prompt'] for k in self.items]))

    def _get(self, index: int):
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DATASETS',
                            __class__.__name__,
                            Text2ImageDataset.para_dict,
                            set_name=True)
