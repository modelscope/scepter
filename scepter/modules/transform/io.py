# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile

from scepter.modules.transform.registry import TRANSFORMS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import DATA_FS as FS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pillow_convert(image, rgb_order):
    if image.mode != rgb_order:
        if image.mode == 'P':
            image = image.convert(f'{rgb_order}A')
        if image.mode == f'{rgb_order}A':
            bg = Image.new(rgb_order,
                           size=(image.width, image.height),
                           color=(255, 255, 255))
            bg.paste(image, (0, 0), mask=image)
            image = bg
        else:
            image = image.convert('RGB')
    return image


@TRANSFORMS.register_class()
class LoadImageFromFile(object):
    """ Load Image from file. We have multi ways to load image. Here we compose them into one transform.

    Args:
        rgb_order (str): 'RGB' or 'BGR'.
        backend (str): 'pillow', 'cv2' or 'torchvision'. Image should be read as uint8 dtype.
            - 'pillow': Read image file as PIL.Image object.
            - 'cv2': Read image file as numpy.ndarray object.
            - 'torchvision': Read image file as tensor object.
    """
    def __init__(self, cfg, logger=None):
        rgb_order = cfg.get('RGB_ORDER', 'RGB')
        backend = cfg.get('BACKEND', 'pillow')
        assert rgb_order in ('RGB', 'BGR')
        assert backend in ('pillow', 'cv2', 'torchvision')
        self.rgb_order = rgb_order
        self.backend = backend

    def read_file(self, img_path):
        if not we.data_online:
            with FS.get_from(img_path) as img_path:
                if self.backend == 'pillow':
                    try:
                        image = Image.open(img_path)
                        image = pillow_convert(image, self.rgb_order)
                    except Exception as e:
                        print(img_path, e)
                elif self.backend == 'cv2':
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if self.rgb_order == 'RGB':
                        try:
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                        except Exception as e:
                            print(img_path, e)
                else:
                    image = Image.open(img_path).convert(self.rgb_order)
                    image_np = np.asarray(image).transpose(
                        (2, 0, 1))  # Tensor type needs shape to be (C, H, W)
                    image = torch.from_numpy(image_np)
            return image
        else:
            with FS.get_object(img_path) as image_data:
                if self.backend == 'pillow':
                    try:
                        image = Image.open(io.BytesIO(image_data))
                        image = pillow_convert(image, self.rgb_order)
                    except Exception as e:
                        print(img_path, e)
                elif self.backend == 'cv2':
                    image = cv2.imdecode(
                        np.array(bytearray(image_data), dtype='uint8'),
                        cv2.IMREAD_COLOR)
                    if self.rgb_order == 'RGB':
                        try:
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                        except Exception as e:
                            print(img_path, e)
                else:
                    image = Image.open(img_path).convert(self.rgb_order)
                    image_np = np.asarray(image).transpose(
                        (2, 0, 1))  # Tensor type needs shape to be (C, H, W)
                    image = torch.from_numpy(image_np)
            return image

    def __call__(self, item):
        if 'prefix' in item['meta']:
            img_path = os.path.join(item['meta']['prefix'],
                                    item['meta']['img_path'])
        else:
            img_path = item['meta']['img_path']
        item['img'] = self.read_file(img_path)
        item['meta']['rgb_order'] = self.rgb_order
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
            'RGB_ORDER': {
                'value': 'RGB',
                'description': 'rgb order'
            },
            'BACKEND': {
                'value': 'pillow',
                'description': 'input backend'
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class LoadImageFromFileList(object):
    """ Load Image from file. We have multi ways to load image. Here we compose them into one transform.

    Args:
        rgb_order (str): 'RGB' or 'BGR'.
        backend (str): 'pillow', 'cv2' or 'torchvision'. Image should be read as uint8 dtype.
            - 'pillow': Read image file as PIL.Image object.
            - 'cv2': Read image file as numpy.ndarray object.
            - 'torchvision': Read image file as tensor object.
    """
    para_dict = [{
        'RGB_ORDER': {
            'value': 'RGB',
            'description': 'Rgb order!'
        },
        'BACKEND': {
            'value': 'pillow',
            'description': 'Input backend!'
        },
        'FILE_KEYS': {
            'value': [],
            'description':
            "The file keys for input, if key include '_path', "
            "the return results will be saved with key as key.replace('_path', '')!"
        }
    }]

    def __init__(self, cfg, logger=None):
        rgb_order = cfg.get('RGB_ORDER', 'RGB')
        backend = cfg.get('BACKEND', 'pillow')
        self.file_keys = cfg.get('FILE_KEYS', ['img_path'])
        if isinstance(self.file_keys, str):
            self.file_keys = [self.file_keys]
        assert rgb_order in ('RGB', 'BGR')
        assert backend in ('pillow', 'cv2', 'torchvision')
        self.rgb_order = rgb_order
        self.backend = backend

    def read_file(self, img_path):
        if not we.data_online:
            with FS.get_from(img_path) as img_path:
                if self.backend == 'pillow':
                    try:
                        image = Image.open(img_path)
                        image = pillow_convert(image, self.rgb_order)
                    except Exception as e:
                        print(img_path, e)
                elif self.backend == 'cv2':
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if self.rgb_order == 'RGB':
                        try:
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                        except Exception as e:
                            print(img_path, e)
                else:
                    image = Image.open(img_path).convert(self.rgb_order)
                    image_np = np.asarray(image).transpose(
                        (2, 0, 1))  # Tensor type needs shape to be (C, H, W)
                    image = torch.from_numpy(image_np)
            return image
        else:
            with FS.get_object(img_path) as image_data:
                if self.backend == 'pillow':
                    try:
                        image = Image.open(io.BytesIO(image_data))
                        image = pillow_convert(image, self.rgb_order)
                    except Exception as e:
                        print(img_path, e)
                elif self.backend == 'cv2':
                    image = cv2.imdecode(
                        np.array(bytearray(image_data), dtype='uint8'),
                        cv2.IMREAD_COLOR)
                    if self.rgb_order == 'RGB':
                        try:
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                        except Exception as e:
                            print(img_path, e)
                else:
                    image = Image.open(img_path).convert(self.rgb_order)
                    image_np = np.asarray(image).transpose(
                        (2, 0, 1))  # Tensor type needs shape to be (C, H, W)
                    image = torch.from_numpy(image_np)
            return image

    def __call__(self, item):
        for key in self.file_keys:
            if 'prefix' in item['meta']:
                img_path = os.path.join(item['meta']['prefix'],
                                        item['meta'][key])
            else:
                img_path = item['meta'][key]
            item[key.replace('_path', '')] = self.read_file(img_path)
            item['meta']['rgb_order'] = self.rgb_order
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

        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            LoadImageFromFileList.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class LoadPILImageFromFile(object):
    def __init__(self, cfg, logger=None):
        rgb_order = cfg.get('RGB_ORDER', 'RGB')
        assert rgb_order in ('RGB', 'BGR')
        self.rgb_order = rgb_order

    def __call__(self, item):
        if 'prefix' in item['meta']:
            img_path = os.path.join(item['meta']['prefix'],
                                    item['meta']['img_path'])
        else:
            img_path = item['meta']['img_path']

        with FS.get_from(img_path) as img_path:
            image = Image.open(img_path).convert(self.rgb_order)
            item['img'] = image
            item['meta']['rgb_order'] = self.rgb_order
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
            'RGB_ORDER': {
                'value': 'RGB',
                'description': 'rgb order'
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class LoadCvImageFromFile(object):
    def __init__(self, cfg, logger=None):
        rgb_order = cfg.get('RGB_ORDER', 'RGB')
        assert rgb_order in ('RGB', 'BGR')
        self.rgb_order = rgb_order

    def __call__(self, item):
        if 'prefix' in item['meta']:
            img_path = os.path.join(item['meta']['prefix'],
                                    item['meta']['img_path'])
        else:
            img_path = item['meta']['img_path']

        with FS.get_from(img_path) as img_path:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if self.rgb_order == 'RGB':
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        item['img'] = image
        item['meta']['rgb_order'] = self.rgb_order
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
            'RGB_ORDER': {
                'value': 'RGB',
                'description': 'rgb order'
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)
