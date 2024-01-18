# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import os
import sys

from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.data.dataset.registry import DATASETS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


@DATASETS.register_class()
class ImageTextPairMSDataset(BaseDataset):
    para_dict = {
        'MS_DATASET_NAME': {
            'value': '',
            'description': 'Modelscope dataset name.'
        },
        'MS_DATASET_NAMESPACE': {
            'value': '',
            'description': 'Modelscope dataset namespace.'
        },
        'MS_DATASET_SUBNAME': {
            'value': '',
            'description': 'Modelscope dataset subname.'
        },
        'MS_DATASET_SPLIT': {
            'value': '',
            'description':
            'Modelscope dataset split set name, default is train.'
        },
        'MS_REMAP_KEYS': {
            'value':
            None,
            'description':
            'Modelscope dataset header of list file, the default is Target:FILE; '
            'If your file is not this header, please set this field, which is a map dict.'
            "For example, { 'Image:FILE': 'Target:FILE' } will replace the filed Image:FILE to Target:FILE"
        },
        'MS_REMAP_PATH': {
            'value':
            None,
            'description':
            'When modelscope dataset name is not None, that means you use the dataset from modelscope,'
            ' default is None. But if you want to use the datalist from modelscope and the file from '
            'local device, you can use this field to set the root path of your images. '
        },
        'TRIGGER_WORDS': {
            'value':
            '',
            'description':
            'The words used to describe the common features of your data, especially when you customize a '
            'tuner. Use these words you can get what you want.'
        },
        'REPLACE_STYLE': {
            'value':
            False,
            'description':
            'Whether use the MS_DATASET_SUBNAME to replace the word in your description, default is False.'
        },
        'HIGHLIGHT_KEYWORDS': {
            'value':
            '',
            'description':
            'The keywords you want to highlight in prompt, which will be replace by <HIGHLIGHT_KEYWORDS>.'
        },
        'KEYWORDS_SIGN': {
            'value':
            '',
            'description':
            'The keywords sign you want to add, which is like <{HIGHLIGHT_KEYWORDS}{KEYWORDS_SIGN}>'
        },
        'OUTPUT_SIZE': {
            'value':
            None,
            'description':
            'If you use the FlexibleResize transforms, this filed will output the image_size as [h, w],'
            'which will be used to set the output size of images used to train the model.'
        },
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg=cfg, logger=logger)
        from modelscope import MsDataset
        from modelscope.utils.constant import DownloadMode
        ms_dataset_name = cfg.get('MS_DATASET_NAME', None)
        ms_dataset_namespace = cfg.get('MS_DATASET_NAMESPACE', None)
        ms_dataset_subname = cfg.get('MS_DATASET_SUBNAME', None)
        ms_dataset_split = cfg.get('MS_DATASET_SPLIT', 'train')
        ms_remap_keys = cfg.get('MS_REMAP_KEYS', None)
        ms_remap_path = cfg.get('MS_REMAP_PATH', None)
        self.replace_style = cfg.get('REPLACE_STYLE', False)
        self.trigger_words = cfg.get('TRIGGER_WORDS', '')
        self.replace_keywords = cfg.get('HIGHLIGHT_KEYWORDS', '')
        self.keywords_sign = cfg.get('KEYWORDS_SIGN', '')
        self.output_size = cfg.get('OUTPUT_SIZE', None)
        if self.output_size is not None:
            if isinstance(self.output_size, numbers.Number):
                self.output_size = [self.output_size, self.output_size]
        # Use modelscope dataset

        if not ms_dataset_name:
            raise (
                'Your must set MS_DATASET_NAME as modelscope dataset or your local dataset orignized '
                'as modelscope dataset.')
        if FS.exists(ms_dataset_name):
            ms_dataset_name = FS.get_dir_to_local_dir(ms_dataset_name)
            ms_remap_path = ms_dataset_name
        try:
            self.data = MsDataset.load(str(ms_dataset_name),
                                       namespace=ms_dataset_namespace,
                                       subset_name=ms_dataset_subname,
                                       split=ms_dataset_split)
        except Exception:
            self.logger.info(
                "Load Modelscope dataset failed, retry with download_mode='force_redownload'."
            )
            try:
                self.data = MsDataset.load(
                    str(ms_dataset_name),
                    namespace=ms_dataset_namespace,
                    subset_name=ms_dataset_subname,
                    split=ms_dataset_split,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD)
            except Exception as sec_e:
                raise f'Load Modelscope dataset failed {sec_e}.'
        if ms_remap_keys:
            self.data = self.data.remap_columns(ms_remap_keys.get_dict())

        if ms_remap_path:

            def map_func(example):
                example['Target:FILE'] = os.path.join(ms_remap_path,
                                                      example['Target:FILE'])
                return example

            self.data = self.data.ds_instance.map(map_func)
        self.real_number = len(self.data)

    def __len__(self):
        if self.mode == 'train':
            return sys.maxsize
        else:
            return len(self.data)

    def _get(self, index: int):
        current_data = self.data[index % len(self.data)]
        # print(current_data.keys())
        image_path = current_data['Target:FILE']
        prompt = current_data['Prompt']
        style = current_data['Style'] if 'Style' in current_data else ''
        # print(prompt, style)
        if self.replace_style and not style == '':
            prompt = prompt.replace(style, f'<{self.keywords_sign}>')
        elif not self.replace_keywords.strip() == '':
            prompt = prompt.replace(
                self.replace_keywords,
                '<' + self.replace_keywords + f'{self.keywords_sign}>')
        if not self.trigger_words == '':
            prompt = self.trigger_words.strip() + ' ' + prompt
        if we.debug:
            print(prompt, self.replace_keywords.strip())
        ret_item = {
            'meta': {
                'img_path': image_path,
                'data_key': style,
                'data_num': self.real_number
            },
            'prompt': prompt
        }
        if self.output_size is not None:
            ret_item['meta']['image_size'] = self.output_size
        return ret_item

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DATASet',
                            __class__.__name__,
                            ImageTextPairMSDataset.para_dict,
                            set_name=True)


@DATASETS.register_class()
class ImageTextPairFolderDataset(BaseDataset):
    para_dict = {
        'DATA_FOLDER': {
            'value': '',
            'description': 'Dataset folder.'
        },
        'TRIGGER_WORDS': {
            'value':
            '',
            'description':
            'The words used to describe the common features of your data, especially when you customize a '
            'tuner. Use these words you can get what you want.'
        },
        'REPLACE_STYLE': {
            'value':
            False,
            'description':
            'Whether use the MS_DATASET_SUBNAME to replace the word in your description, default is False.'
        },
        'HIGHLIGHT_KEYWORDS': {
            'value':
            '',
            'description':
            'The keywords you want to highlight in prompt, which will be replace by <HIGHLIGHT_KEYWORDS>.'
        },
        'KEYWORDS_SIGN': {
            'value':
            '',
            'description':
            'The keywords sign you want to add, which is like <{HIGHLIGHT_KEYWORDS}{KEYWORDS_SIGN}>'
        },
        'OUTPUT_SIZE': {
            'value':
            None,
            'description':
            'If you use the FlexibleResize transforms, this filed will output the image_size as [h, w],'
            'which will be used to set the output size of images used to train the model.'
        },
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg=cfg, logger=logger)
        data_folder = cfg.get('DATA_FOLDER', None)
        self.replace_style = cfg.get('REPLACE_STYLE', False)
        self.trigger_words = cfg.get('TRIGGER_WORDS', '')
        self.replace_keywords = cfg.get('HIGHLIGHT_KEYWORDS', '')
        self.keywords_sign = cfg.get('KEYWORDS_SIGN', '')
        self.output_size = cfg.get('OUTPUT_SIZE', None)
        if self.output_size is not None:
            if isinstance(self.output_size, numbers.Number):
                self.output_size = [self.output_size, self.output_size]
        # Use modelscope dataset
        if not data_folder or not FS.exists(data_folder):
            raise ('Your must set datafolder for local dataset.')
        data_folder = FS.get_dir_to_local_dir(data_folder)
        all_lines = open(os.path.join(data_folder, 'train.csv'),
                         'r').read().split('\n')
        assert all_lines[0] == 'Target:FILE,Prompt'
        self.data = []
        for line in all_lines[1:]:
            line = line.strip()
            if line == '':
                continue
            self.data.append({
                'Target:FILE':
                os.path.join(data_folder,
                             line.split(',', 1)[0]),
                'Prompt':
                line.split(',', 1)[1]
            })
        self.real_number = len(self.data)

    def __len__(self):
        if self.mode == 'train':
            return sys.maxsize
        else:
            return len(self.data)

    def _get(self, index: int):
        current_data = self.data[index % len(self.data)]
        # print(current_data.keys())
        image_path = current_data['Target:FILE']
        prompt = current_data['Prompt']
        style = current_data['Style'] if 'Style' in current_data else ''
        # print(prompt, style)
        if self.replace_style and not style == '':
            prompt = prompt.replace(style, f'<{self.keywords_sign}>')
        elif not self.replace_keywords.strip() == '':
            prompt = prompt.replace(
                self.replace_keywords,
                '<' + self.replace_keywords + f'{self.keywords_sign}>')
        if not self.trigger_words == '':
            prompt = self.trigger_words.strip() + ' ' + prompt
        if we.debug:
            print(prompt, self.replace_keywords.strip())
        ret_item = {
            'meta': {
                'img_path': image_path,
                'data_key': style,
                'data_num': self.real_number
            },
            'prompt': prompt
        }
        if self.output_size is not None:
            ret_item['meta']['image_size'] = self.output_size
        return ret_item

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DATASet',
                            __class__.__name__,
                            ImageTextPairMSDataset.para_dict,
                            set_name=True)
