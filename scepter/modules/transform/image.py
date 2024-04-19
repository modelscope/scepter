# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers

import numpy as np
import opencv_transforms.functional as cv2_TF
import opencv_transforms.transforms as cv2_transforms
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from scepter.modules.transform.registry import TRANSFORMS
from scepter.modules.transform.utils import (
    BACKEND_CV2, BACKEND_PILLOW, BACKEND_TORCHVISION, INPUT_CV2_TYPE_WARNING,
    INPUT_PIL_TYPE_WARNING, INPUT_TENSOR_TYPE_WARNING, INTERPOLATION_STYLE,
    INTERPOLATION_STYLE_CV2, TORCHVISION_CAPABILITY, is_cv2_image,
    is_pil_image, is_tensor)
from scepter.modules.utils.config import dict_to_yaml

if TORCHVISION_CAPABILITY:
    BACKENDS = (BACKEND_PILLOW, BACKEND_CV2, BACKEND_TORCHVISION)
else:
    BACKENDS = (BACKEND_PILLOW, BACKEND_CV2)


class ImageTransform(object):
    para_dict = [{
        'INPUT_KEY': {
            'value': 'img',
            'description': 'input key or key list.'
        },
        'OUTPUT_KEY': {
            'value': 'img',
            'description': 'input key or key list.'
        },
        'BACKEND': {
            'value': 'pillow',
            'description': 'backend, choose from pillow, cv2, torchvision'
        }
    }]

    def __init__(self, cfg, logger=None):
        self.input_key = cfg.get('INPUT_KEY', 'img')
        self.output_key = cfg.get('OUTPUT_KEY', 'img')
        self.backend = cfg.get('BACKEND', BACKEND_PILLOW)

    def check_image_type(self, input_img):
        if self.backend == BACKEND_PILLOW:
            assert is_pil_image(input_img), INPUT_PIL_TYPE_WARNING
            w, h = input_img.size
            return h, w
        elif self.backend == BACKEND_CV2:
            assert is_cv2_image(input_img), INPUT_CV2_TYPE_WARNING
            h, w, c = input_img.shape
            return h, w
        elif TORCHVISION_CAPABILITY:
            if self.backend == BACKEND_TORCHVISION:
                assert is_tensor(input_img), INPUT_TENSOR_TYPE_WARNING
                c, h, w = input_img.shape
                return h, w

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
                            ImageTransform.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class RandomCrop(ImageTransform):
    """ Crop a random portion of image.
    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        size (sequence or int): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the output size will be matched to (size, size).
        padding (sequence or int): Optional padding on each border of the image. Default is None.
        pad_if_needed (bool): It will pad the image if smaller than the desired size to avoid raising an exception.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
    """
    para_dict = [{
        'SIZE': {
            'value': 224,
            'description': 'crop size'
        },
        'PADDING': {
            'value': None,
            'description': 'padding'
        },
        'PAD_IF_NEEDED': {
            'value': False,
            'description': 'pad if needed'
        },
        'FILL': {
            'value': 0,
            'description': 'fill'
        },
        'PADDING_MODE': {
            'value': 'constant',
            'description': 'padding mode'
        }
    }]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        size = cfg.SIZE
        padding = cfg.get('PADDING', None)
        pad_if_needed = cfg.get('PAD_IF_NEEDED', False)
        fill = cfg.get('FILL', 0)
        padding_mode = cfg.get('PADDING_MODE', 'constant')
        super(RandomCrop, self).__init__(cfg)
        assert self.backend in BACKENDS
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            self.callable = transforms.RandomCrop(size,
                                                  padding=padding,
                                                  pad_if_needed=pad_if_needed,
                                                  fill=fill,
                                                  padding_mode=padding_mode)
        else:
            self.callable = cv2_transforms.RandomCrop(
                size,
                padding=padding,
                pad_if_needed=pad_if_needed,
                fill=fill,
                padding_mode=padding_mode)

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            self.check_image_type(item[key])
            item[self.output_key[idx]] = self.callable(item[key])
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
                            RandomCrop.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class RandomResizedCrop(ImageTransform):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

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
            'value': 224,
            'description': 'crop size'
        },
        'RATIO': {
            'value': [3. / 4., 4. / 3.],
            'description': 'ratio'
        },
        'SCALE': {
            'value': [0.08, 1.0],
            'description': 'scale'
        },
        'INTERPOLATION': {
            'value': 'bilinear',
            'description': 'interpolation'
        }
    }]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(RandomResizedCrop, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS
        self.interpolation = cfg.get('INTERPOLATION', 'bilinear')
        self.size = cfg.SIZE
        self.scale = tuple(cfg.get('SCALE', [0.08, 1.0]))
        self.ratio = tuple(cfg.get('RATIO', [3. / 4., 4. / 3.]))
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            assert self.interpolation in INTERPOLATION_STYLE
        else:
            assert self.interpolation in INTERPOLATION_STYLE_CV2
        self.callable = transforms.RandomResizedCrop(self.size,
             self.scale, self.ratio, INTERPOLATION_STYLE[self.interpolation]) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) \
            else cv2_transforms.RandomResizedCrop(self.size, self.scale, self.ratio, INTERPOLATION_STYLE_CV2[self.interpolation]) # noqa

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            self.check_image_type(item[key])
            item[self.output_key[idx]] = self.callable(item[key])
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
                            RandomResizedCrop.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class Resize(ImageTransform):
    """Resize image to a given size.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        size (int or sequence): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the smaller edge of the image will be matched to this number
            maintaining the aspect ratio.
        interpolation (str): Desired interpolation string, 'bilinear', 'nearest', 'bicubic' are supported.
    """
    para_dict = [{
        'INTERPOLATION': {
            'value': 'bilinear',
            'description': 'interpolation'
        },
        'SIZE': {
            'value': 224,
            'description': 'resize to size 224'
        }
    }]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(Resize, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS
        self.size = cfg.SIZE
        self.interpolation = cfg.get('INTERPOLATION', 'bilinear')
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            assert self.interpolation in INTERPOLATION_STYLE
        else:
            assert self.interpolation in INTERPOLATION_STYLE_CV2
        self.callable = transforms.Resize(self.size, INTERPOLATION_STYLE[self.interpolation]) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) \
            else cv2_transforms.Resize(self.size, INTERPOLATION_STYLE_CV2[self.interpolation])

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            self.check_image_type(item[key])
            item[self.output_key[idx]] = self.callable(item[key])
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
                            Resize.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class CenterCrop(ImageTransform):
    """ Crops the given image at the center.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        size (sequence or int): Desired output size.
            If size is a sequence like (h, w), the output size will be matched to this.
            If size is an int, the output size will be matched to (size, size).
    """
    para_dict = [{'SIZE': {'value': 224, 'description': 'resize to size 224'}}]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(CenterCrop, self).__init__(cfg, logger=None)
        assert self.backend in BACKENDS
        self.size = cfg.SIZE
        self.callable = transforms.CenterCrop(self.size) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.CenterCrop(self.size)

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            self.check_image_type(item[key])
            item[self.output_key[idx]] = self.callable(item[key])
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
                            CenterCrop.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class RandomHorizontalFlip(ImageTransform):
    """ Horizontally flip the given image randomly with a given probability.

    If the image is torch Tensor, it is expected to have [..., H, W] shape.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    para_dict = [{'P': {'value': 0.5, 'description': 'P'}}]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(RandomHorizontalFlip, self).__init__(cfg, logger=None)
        p = cfg.get('P', 0.5)
        assert self.backend in BACKENDS
        self.callable = transforms.RandomHorizontalFlip(p) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.RandomHorizontalFlip(p)

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            self.check_image_type(item[key])
            item[self.output_key[idx]] = self.callable(item[key])
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
                            RandomHorizontalFlip.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class Normalize(ImageTransform):
    """ Normalize a tensor image with mean and standard deviation.
    This transform only support tensor image.

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
        },
    }]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(Normalize, self).__init__(cfg, logger=None)
        assert self.backend in BACKENDS
        mean = cfg.MEAN
        std = cfg.STD
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.callable = transforms.Normalize(self.mean, self.std) \
            if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_transforms.Normalize(self.mean, self.std)

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            item[self.output_key[idx]] = self.callable(item[key])
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
                            Normalize.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class ImageToTensor(ImageTransform):
    """ Convert a ``PIL Image`` or ``numpy.ndarray`` or uint8 type tensor to a float32 tensor,
    and scale output to [0.0, 1.0].
    """
    para_dict = [{}]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(ImageToTensor, self).__init__(cfg, logger)
        assert self.backend in BACKENDS

        if self.backend == BACKEND_PILLOW:
            self.callable = transforms.ToTensor()
        elif self.backend == BACKEND_CV2:
            self.callable = cv2_transforms.ToTensor()
        else:
            self.callable = transforms.ConvertImageDtype(torch.float)

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        for idx, key in enumerate(self.input_key):
            item[self.output_key[idx]] = self.callable(item[key])
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
                            ImageToTensor.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class FlexibleResize(ImageTransform):
    para_dict = [{
        'INTERPOLATION': {
            'value': 'bilinear',
            'description': 'interpolation'
        },
    }]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(FlexibleResize, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS
        self.size = cfg.get('SIZE', None)
        if self.size is not None:
            if isinstance(self.size, numbers.Number):
                self.size = [self.size, self.size]

        interpolation = cfg.get('INTERPOLATION', 'bilinear')
        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            assert interpolation in INTERPOLATION_STYLE
        else:
            assert interpolation in INTERPOLATION_STYLE_CV2

        if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
            self.callable = TF.resize
            self.interpolation = INTERPOLATION_STYLE[interpolation]
        else:
            self.callable = cv2_TF.resize
            self.interpolation = INTERPOLATION_STYLE_CV2[interpolation]

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]

        for idx, key in enumerate(self.input_key):
            ih, iw = self.check_image_type(item[key])
            meta = item.get('meta', {})
            if 'image_size' in meta:
                iw, ih, ow, oh = iw, ih, meta['image_size'][1], meta[
                    'image_size'][0]
            elif self.size is not None:
                iw, ih, ow, oh = iw, ih, self.size[1], self.size[0]
                meta['image_size'] = [oh, ow]
            else:
                raise KeyError(
                    'The meta of input item must consists of '
                    "['width', 'height', 'image_size'], and at least one key is missing."
                )
            scale = max(ow / iw, oh / ih)
            new_size = (round(scale * ih), round(scale * iw))
            item[self.output_key[idx]] = self.callable(item[key], new_size,
                                                       self.interpolation)
        return item

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            FlexibleResize.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class FlexibleCenterCrop(ImageTransform):
    para_dict = [{}]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(FlexibleCenterCrop, self).__init__(cfg, logger=None)
        assert self.backend in BACKENDS
        self.size = cfg.get('SIZE', None)
        if self.size is not None:
            if isinstance(self.size, numbers.Number):
                self.size = [self.size, self.size]
        self.callable = TF.center_crop if self.backend in (
            BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_TF.center_crop

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]

        meta = item.get('meta', {})
        if 'image_size' in meta:
            oh, ow = meta['image_size']
            out_size = (oh, ow)
        else:
            out_size = self.size

        for idx, key in enumerate(self.input_key):
            self.check_image_type(item[key])
            item[self.output_key[idx]] = self.callable(item[key], out_size)
        return item

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            FlexibleCenterCrop.para_dict,
                            set_name=True)
