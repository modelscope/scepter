# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import random

import numpy as np
import torch
import torchvision.transforms.functional as functional
import torchvision.transforms.transforms as transforms
from packaging import version
from torchvision.version import __version__ as tv_version

from scepter.modules.transform.registry import TRANSFORMS
from scepter.modules.transform.utils import (BACKEND_TORCHVISION,
                                             INTERPOLATION_STYLE, is_tensor)
# torchvision.transform._transforms_video is deprecated since torchvision 0.10.0, use transform instead
from scepter.modules.utils.config import dict_to_yaml

use_video_transforms = version.parse(tv_version) < version.parse('0.10.0')

BACKENDS = (BACKEND_TORCHVISION, )


class VideoTransform(object):
    para_dict = [{
        'INPUT_KEY': {
            'value': 'img',
            'description': 'input key'
        },
        'OUTPUT_KEY': {
            'value': 'img',
            'description': 'input key'
        },
        'BACKEND': {
            'value': 'pillow',
            'description': 'backend'
        }
    }]

    def __init__(self, cfg, logger=None):
        backend = cfg.get('BACKEND', BACKEND_TORCHVISION)
        self.input_key = cfg.get('INPUT_KEY', 'video')
        self.output_key = cfg.get('OUTPUT_KEY', 'video')
        self.backend = backend

    def check_video_type(self, input_video):
        if self.backend == BACKEND_TORCHVISION:
            assert is_tensor(input_video)

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
                            VideoTransform.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class RandomResizedCropVideo(VideoTransform):
    """Crop a random portion of video and resize it to a given size.

    Expect the video is a torch tensor with shape [..., H, W]

    Args:
    size (int or sequence): Desired output size.
        If size is a sequence like (h, w), the output size will be matched to this.
        If size is an int, the output size will be matched to (size, size).
    scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
        before resizing. The scale is defined with respect to the area of the original image.
    ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
        resizing.
    interpolation (str): Desired interpolation string, 'bilinear', 'nearest', 'bicubic' are supported.
    """
    para_dict = [{
        'SIZE': {
            'value': 0,
            'description': 'size'
        },
        'SCALE': {
            'value': [0.08, 1.0],
            'description': 'scale'
        },
        'RATIO': {
            'value': [3. / 4., 4. / 3.],
            'description': 'ratio'
        },
        'INTERPOLATION': {
            'value': 'bilinear',
            'description': 'interpolation'
        }
    }]
    para_dict[0].update(VideoTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        size, scale = cfg.SIZE, cfg.get('SCALE', [0.08, 1.0])
        ratio = cfg.get('RATIO', [3. / 4., 4. / 3.])
        interpolation = cfg.get('INTERPOLATION', 'bilinear')
        super(RandomResizedCropVideo, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS
        self.interpolation = interpolation
        if isinstance(size, (tuple, list)):
            assert len(size) == 2
            size = tuple(size)
        elif isinstance(size, int):
            size = (size, size)
        else:
            raise ValueError(
                f'Unexpected type {type(size)}, expected int or tuple or list')

        if use_video_transforms:
            from torchvision.transforms._transforms_video import \
                RandomResizedCropVideo as RandomResizedCropVideoOp
            self.callable = RandomResizedCropVideoOp(size, scale, ratio,
                                                     self.interpolation)
        else:
            self.callable = transforms.RandomResizedCrop(
                size, scale, ratio, INTERPOLATION_STYLE[self.interpolation])

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
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
                            RandomResizedCropVideo.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class CenterCropVideo(VideoTransform):
    """ Crops the given video at the center.

    Expect the video is a torch tensor with shape [..., H, W]

    Args:
        size (sequence or int): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the output size will be matched to (size, size).
    """
    para_dict = [{'SIZE': {'value': 0, 'description': 'size'}}]
    para_dict[0].update(VideoTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        size = cfg.SIZE
        super(CenterCropVideo, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS
        self.size = size

        if use_video_transforms:
            from torchvision.transforms._transforms_video import \
                CenterCropVideo as CenterCropVideoOp
            self.callable = CenterCropVideoOp(size)
        else:
            self.callable = transforms.CenterCrop(size)

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
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
                            CenterCropVideo.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class RandomHorizontalFlipVideo(VideoTransform):
    """ Horizontally flip the given video randomly with a given probability.

    Expect the video is a torch tensor with shape [..., H, W]

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    para_dict = [{'P': {'value': 0.5, 'description': 'P'}}]
    para_dict[0].update(VideoTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        p = cfg.get('P', 0.5)
        super(RandomHorizontalFlipVideo, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS

        if use_video_transforms:
            from torchvision.transforms._transforms_video import \
                RandomHorizontalFlipVideo as RandomHorizontalFlipVideoOp
            self.callable = RandomHorizontalFlipVideoOp(p)
        else:
            self.callable = transforms.RandomHorizontalFlip(p)

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.callable(item[self.input_key])
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
                            RandomHorizontalFlipVideo.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class NormalizeVideo(VideoTransform):
    """ Normalize a tensor video with mean and standard deviation.
    Expect the video is a torch tensor with shape [..., H, W]

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    para_dict = [{
        'MEAN': {
            'value': [],
            'description': 'mean'
        },
        'STD': {
            'value': [],
            'description': 'std'
        }
    }]
    para_dict[0].update(VideoTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        mean, std = cfg.MEAN, cfg.STD
        super(NormalizeVideo, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        if use_video_transforms:
            from torchvision.transforms._transforms_video import \
                NormalizeVideo as NormalizeVideoOp
            self.callable = NormalizeVideoOp(self.mean, self.std)
        else:
            self.callable = transforms.Normalize(self.mean, self.std)

    def __call__(self, item):
        video = item[self.input_key]
        if not use_video_transforms:
            video = video.permute(1, 0, 2, 3)
        video = self.callable(video)
        if not use_video_transforms:
            video = video.permute(1, 0, 2, 3)
        item[self.output_key] = video
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
                            NormalizeVideo.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class VideoToTensor(VideoTransform):
    """ Convert a uint8 type tensor to a float32 tensor, permute it and scale output to [0.0, 1.0].

    Expect the video is a uint8 torch tensor with shape [T, H, W, C]
    """
    para_dict = [{}]
    para_dict[0].update(VideoTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(VideoToTensor, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS

    def __call__(self, item):
        video = item[self.input_key]
        if isinstance(video, np.ndarray):
            video = torch.tensor(video)

        if not torch.is_tensor(video):
            raise TypeError('video should be Tensor. Got %s' % type(video))

        if not video.ndimension() == 4:
            raise ValueError('video should be 4D. Got %dD' % video.dim())

        if not video.dtype == torch.uint8:
            raise TypeError(
                'video tensor should have data type uint8. Got %s' %
                str(video.dtype))

        item[self.output_key] = video.float().permute(3, 0, 1, 2) / 255.0

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
                            VideoToTensor.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class AutoResizedCropVideo(VideoTransform):
    """ Crop the video with a given position and resize it to a given size.

    Expect the video is a torch tensor with shape [..., H, W].

    Input ``crop_mode`` supports values:
        - `cc`: center-center
        - `cl`: left-center
        - `cr`: right-center
        - `tl`: left-top
        - `tr`: right-top
        - `bl`: left-bottom
        - `br`: right-bottom

    Args:
        size (int or sequence): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the output size will be matched to (size, size).
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        interpolation (str): Desired interpolation string, 'bilinear', 'nearest', 'bicubic' are supported.
    """
    para_dict = [{'SCALE': {'value': [0.08, 1.0], 'description': 'scale'}}]
    para_dict[0].update(VideoTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        size, scale = cfg.SIZE, cfg.get('SCALE', [0.08, 1.0])
        interpolation = cfg.get('INTERPOLATION', 'bilinear')
        super(AutoResizedCropVideo, self).__init__(cfg, logger=logger)
        if isinstance(size, (tuple, list)):
            assert len(size) == 2
            size = tuple(size)
        elif isinstance(size, int):
            size = (size, size)
        else:
            raise ValueError(
                f'Unexpected type {type(size)}, expected int or tuple or list')
        self.size = size
        self.scale = scale
        self.interpolation_mode = interpolation

    def get_crop(self, clip, crop_mode='cc'):
        scale = random.uniform(*self.scale)
        _, _, video_height, video_width = clip.shape
        min_length = min(video_height, video_width)
        crop_size = int(min_length * scale)
        center_x = video_width // 2
        center_y = video_height // 2
        box_half = crop_size // 2

        # default is cc
        x0 = center_x - box_half
        y0 = center_y - box_half
        if crop_mode == 'cl':
            x0 = 0
            y0 = center_y - box_half
        elif crop_mode == 'cr':
            x0 = video_width - crop_size
            y0 = center_y - box_half
        elif crop_mode == 'tl':
            x0 = 0
            y0 = 0
        elif crop_mode == 'tr':
            x0 = video_width - crop_size
            y0 = 0
        elif crop_mode == 'bl':
            x0 = 0
            y0 = video_height - crop_size
        elif crop_mode == 'br':
            x0 = video_width - crop_size
            y0 = video_height - crop_size

        if use_video_transforms:
            from torchvision.transforms.functional import resized_crop
            return resized_crop(clip, y0, x0, crop_size, crop_size, self.size,
                                self.interpolation_mode)
        else:
            return functional.resized_crop(
                clip, y0, x0, crop_size, crop_size, self.size,
                INTERPOLATION_STYLE[self.interpolation_mode])

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        crop_mode = item['meta'].get('crop_mode') or 'cc'
        item[self.output_key] = self.get_crop(item[self.input_key], crop_mode)
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
                            AutoResizedCropVideo.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class ResizeVideo(VideoTransform):
    """Resize video to a given size.

    Expect the video is a torch tensor with shape [..., H, W].

    Args:
        size (int or sequence): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the smaller edge of the image will be matched to this
            number maintaining the aspect ratio.
        interpolation (str): Desired interpolation string, 'bilinear', 'nearest', 'bicubic' are supported.
    """
    para_dict = [{
        'SCALE': {
            'value': [0.08, 1.0],
            'description': 'scale'
        },
        'INTERPOLATION': {
            'value': 'bilinear',
            'description': 'interpolation'
        }
    }]
    para_dict[0].update(VideoTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        size = cfg.SIZE
        interpolation = cfg.get('INTERPOLATION', 'bilinear')
        super(ResizeVideo, self).__init__(cfg, logger=logger)
        self.size = size
        if isinstance(self.size, (tuple, list)):
            self.size = tuple(self.size)
            assert len(self.size) == 2
        else:
            if not isinstance(self.size, int):
                raise ValueError(
                    f'Expected size to be tuple or list or int, got {type(self.size)}'
                )
        self.interpolation_mode = interpolation

    def resize(self, clip):
        if use_video_transforms:
            from torchvision.transforms.functional import resize

            # resize function only takes a tuple size
            # so we need to compute scaled target size here
            if isinstance(self.size, int):
                h, w = clip.shape[-2], clip.shape[-1]
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    return clip
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                size = (oh, ow)
            else:
                size = self.size
            return resize(clip, size, self.interpolation_mode)
        else:
            return functional.resize(
                clip, self.size, INTERPOLATION_STYLE[self.interpolation_mode])

    def __call__(self, item):
        self.check_video_type(item[self.input_key])
        item[self.output_key] = self.resize(item[self.input_key])
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
                            ResizeVideo.para_dict,
                            set_name=True)
