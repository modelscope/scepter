# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import math
import numbers
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.data.dataset.registry import DATASETS
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

Image.MAX_IMAGE_PIXELS = None


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
            raise ValueError(
                'Your must set MS_DATASET_NAME as modelscope dataset or your local dataset orignized '
                'as modelscope dataset.')
        if FS.exists(ms_dataset_name):
            ms_dataset_name = FS.get_dir_to_local_dir(ms_dataset_name)
            # ms_remap_path = ms_dataset_name
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
                raise ValueError(f'Load Modelscope dataset failed {sec_e}.')
        if ms_remap_keys:
            self.data = self.data.remap_columns(ms_remap_keys.get_dict())

        if ms_remap_path:

            def map_func(example):
                return {
                    k: os.path.join(ms_remap_path, v)
                    if k.endswith(':FILE') else v
                    for k, v in example.items()
                }

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
        image_path = current_data[
            'Target:FILE'] if 'Target:FILE' in current_data else ''
        prompt = current_data.get('Prompt', current_data.get('prompt', ''))
        style = current_data['Style'] if 'Style' in current_data else ''
        src_image_path = current_data[
            'Source:FILE'] if 'Source:FILE' in current_data else ''
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
                'src_path': src_image_path,
                'data_key': style,
                'data_num': self.real_number
            },
            'prompt': prompt
        }
        if self.output_size is not None:
            ret_item['meta']['image_size'] = self.output_size
        for key in current_data:
            if key not in ret_item['meta']:
                ret_item['meta'][key] = current_data[key]
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
            raise ValueError('Your must set datafolder for local dataset.')
        data_folder = FS.get_dir_to_local_dir(data_folder)
        all_lines = open(os.path.join(data_folder, 'train.csv'),
                         'r').read().split('\n')
        header = all_lines[0].split(',')
        self.data = []
        for line in all_lines[1:]:
            line = line.strip()
            if line == '':
                continue
            record = dict(zip(header, line.split(',', len(header) - 1)))
            record = {
                k: os.path.join(data_folder, v) if k.endswith(':FILE') else v
                for k, v in record.items()
            }
            self.data.append(record)
        self.real_number = len(self.data)

    def __len__(self):
        if self.mode == 'train':
            return sys.maxsize
        else:
            return len(self.data)

    def _get(self, index: int):
        current_data = self.data[index % len(self.data)]
        # print(current_data.keys())
        image_path = current_data[
            'Target:FILE'] if 'Target:FILE' in current_data else ''
        prompt = current_data.get('Prompt', current_data.get('prompt', ''))
        style = current_data['Style'] if 'Style' in current_data else ''
        src_image_path = current_data[
            'Source:FILE'] if 'Source:FILE' in current_data else ''
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
                'src_path': src_image_path,
                'data_key': style,
                'data_num': self.real_number
            },
            'prompt': prompt
        }
        if self.output_size is not None:
            ret_item['meta']['image_size'] = self.output_size
        for key in current_data:
            if key not in ret_item['meta']:
                ret_item['meta'][key] = current_data[key]
        return ret_item

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DATASet',
                            __class__.__name__,
                            ImageTextPairMSDataset.para_dict,
                            set_name=True)


@DATASETS.register_class()
class ImageTextPairMSDatasetForACE(BaseDataset):
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
        'ALIGN_SIZE': {
            'value': False,
            'description':
                'Whether ensure the size align between the source image and target image.'
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

        self.max_seq_len = cfg.get('MAX_SEQ_LEN', 1024)
        self.max_aspect_ratio = cfg.get('MAX_ASPECT_RATIO', 4)
        self.d = cfg.get('DOWNSAMPLE_RATIO', 16)
        self.replace_style = cfg.get('REPLACE_STYLE', False)
        self.trigger_words = cfg.get('TRIGGER_WORDS', '')
        self.replace_keywords = cfg.get('HIGHLIGHT_KEYWORDS', '')
        self.keywords_sign = cfg.get('KEYWORDS_SIGN', '')
        self.add_indicator = cfg.get('ADD_INDICATOR', False)

        self.align_size = cfg.get('ALIGN_SIZE', False)
        # Use modelscope dataset
        if not ms_dataset_name:
            raise ValueError(
                'Your must set MS_DATASET_NAME as modelscope dataset or your local dataset orignized '
                'as modelscope dataset.')
        if FS.exists(ms_dataset_name):
            ms_dataset_name = FS.get_dir_to_local_dir(ms_dataset_name)
            self.ms_dataset_name = ms_dataset_name
            # ms_remap_path = ms_dataset_name
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
                raise ValueError(f'Load Modelscope dataset failed {sec_e}.')
        if ms_remap_keys:
            self.data = self.data.remap_columns(ms_remap_keys.get_dict())

        if ms_remap_path:

            def map_func(example):
                return {
                    k: os.path.join(ms_remap_path, v)
                    if k.endswith(':FILE') else v
                    for k, v in example.items()
                }

            self.data = self.data.ds_instance.map(map_func)

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        if self.mode == 'train':
            return sys.maxsize
        else:
            return len(self.data)

    def _get(self, index: int):
        current_data = self.data[index % len(self.data)]

        tar_image_path = current_data.get('Target:FILE', '')
        src_image_path = current_data.get('Source:FILE', '')

        style = current_data.get('Style', '')
        prompt = current_data.get('Prompt', current_data.get('prompt', ''))
        if self.replace_style and not style == '':
            prompt = prompt.replace(style, f'<{self.keywords_sign}>')

        elif not self.replace_keywords.strip() == '':
            prompt = prompt.replace(
                self.replace_keywords,
                '<' + self.replace_keywords + f'{self.keywords_sign}>')

        if not self.trigger_words == '':
            prompt = self.trigger_words.strip() + ' ' + prompt

        src_image = self.load_image(self.ms_dataset_name,
                                    src_image_path,
                                    cvt_type='RGB')
        tar_image = self.load_image(self.ms_dataset_name,
                                    tar_image_path,
                                    cvt_type='RGB')
        src_image = self.image_preprocess(src_image)
        tar_image = self.image_preprocess(tar_image, size = src_image.shape[:2]  if self.align_size else None)

        tar_image = self.transforms(tar_image)
        src_image = self.transforms(src_image)
        src_mask = torch.ones_like(src_image[[0]])
        tar_mask = torch.ones_like(tar_image[[0]])
        if self.add_indicator:
            if '{image}' not in prompt:
                prompt = '{image}, ' + prompt
        return {
            'src_image_list': [src_image],
            'src_mask_list': [src_mask],
            'image': tar_image,
            'image_mask': tar_mask,
            'prompt': [prompt],
            'edit_id': [0]
        }

    def load_image(self, prefix, img_path, cvt_type=None):
        if img_path is None or img_path == '':
            return None
        img_path = os.path.join(prefix, img_path)
        with FS.get_object(img_path) as image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
            if cvt_type is not None:
                image = pillow_convert(image, cvt_type)
        return image

    def image_preprocess(self,
                         img,
                         size=None,
                         interpolation=InterpolationMode.BILINEAR):
        H, W = img.height, img.width
        if H / W > self.max_aspect_ratio:
            img = T.CenterCrop((self.max_aspect_ratio * W, W))(img)
        elif W / H > self.max_aspect_ratio:
            img = T.CenterCrop((H, self.max_aspect_ratio * H))(img)

        if size is None:
            # resize image for max_seq_len, while keep the aspect ratio
            H, W = img.height, img.width
            scale = min(
                1.0,
                math.sqrt(self.max_seq_len / ((H / self.d) * (W / self.d))))
            rH = int(
                H * scale) // self.d * self.d  # ensure divisible by self.d
            rW = int(W * scale) // self.d * self.d
        else:
            rH, rW = size
        img = T.Resize((rH, rW), interpolation=interpolation,
                       antialias=True)(img)
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DATASet',
                            __class__.__name__,
                            ImageTextPairMSDatasetForACE.para_dict,
                            set_name=True)

    @staticmethod
    def collate_fn(batch):
        collect = defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                collect[k].append(v)

        new_batch = dict()
        for k, v in collect.items():
            if all([i is None for i in v]):
                new_batch[k] = None
            else:
                new_batch[k] = v

        return new_batch
