# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers
import random

# https://github.com/TengdaHan/DPC/blob/master/utils/augmentation.py
import torch
from torchvision.transforms import Compose, Lambda

from scepter.modules.transform.image import ImageTransform
from scepter.modules.transform.registry import TRANSFORMS
from scepter.modules.transform.utils import (BACKEND_CV2, BACKEND_PILLOW,
                                             BACKEND_TORCHVISION,
                                             TORCHVISION_CAPABILITY)
from scepter.modules.utils.config import dict_to_yaml

if TORCHVISION_CAPABILITY:
    BACKENDS = (BACKEND_PILLOW, BACKEND_CV2, BACKEND_TORCHVISION)
else:
    BACKENDS = (BACKEND_PILLOW, BACKEND_CV2)


def _is_tensor_a_torch_image(input):
    return input.ndim >= 2


def _blend(img1, img2, ratio):
    # type: (torch.Tensor, torch.Tensor, float) -> torch.Tensor
    bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64
                                ] else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def rgb_to_grayscale(img, split=False):
    # type: (torch.Tensor) -> torch.Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
        Args:
            clip (torch.tensor): Size is (T, H, W, C)
        Return:
            clip (torch.tensor): Size is (T, H, W, C)
    """
    orig_dtype = img.dtype
    rgb_convert = torch.tensor([0.299, 0.587, 0.114])
    if split:
        rgb_convert *= 0
        channel = random.randint(0, 2)
        rgb_convert[channel] = 1

    assert img.shape[0] == 3, 'First dimension need to be 3 Channels'
    if img.is_cuda:
        rgb_convert = rgb_convert.to(img.device)

    img = img.float().permute(1, 2, 3, 0).matmul(rgb_convert).to(orig_dtype)
    return torch.stack([img, img, img], 0)


def _rgb2hsv(img):
    r, g, b = img.unbind(0)

    maxc, _ = torch.max(img, dim=0)
    minc, _ = torch.min(img, dim=0)

    eqc = maxc == minc
    cr = maxc - minc
    s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
    cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc))


def _hsv2rgb(img):
    l = len(img.shape)  # noqa
    h, s, v = img.unbind(0)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    if l == 3:  # noqa
        tmp = torch.arange(6)[:, None, None]
    elif l == 4:  # noqa
        tmp = torch.arange(6)[:, None, None, None]

    if img.is_cuda:
        tmp = tmp.to(img.device)

    mask = i == tmp  # (H, W) == (6, H, W)

    a1 = torch.stack((v, q, p, p, t, v))
    a2 = torch.stack((t, v, v, q, p, p))
    a3 = torch.stack((p, p, t, v, v, q))
    a4 = torch.stack((a1, a2, a3))  # (3, 6, H, W)

    if l == 3:  # noqa
        return torch.einsum('ijk, xijk -> xjk', mask.to(dtype=img.dtype),
                            a4)  # (C, H, W)
    elif l == 4:  # noqa
        return torch.einsum('itjk, xitjk -> xtjk', mask.to(dtype=img.dtype),
                            a4)  # (C, T, H, W)


def adjust_brightness(img, brightness_factor):
    # type: (torch.Tensor, float) -> torch.Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img, contrast_factor):
    # type: (torch.Tensor, float) -> torch.Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    mean = torch.mean(rgb_to_grayscale(img).to(torch.float),
                      dim=(-4, -2, -1),
                      keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_saturation(img, saturation_factor):
    # type: (torch.Tensor, float) -> torch.Tensor
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (Tensor): Image to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
         Tensor: Hue adjusted image.
    """
    if isinstance(hue_factor, float) and not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(
            'hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
    elif (isinstance(hue_factor, torch.Tensor)
          and not ((-0.5 <= hue_factor).sum() == hue_factor.shape[0] and
                   (hue_factor <= 0.5).sum() == hue_factor.shape[0])):
        raise ValueError(
            'hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    img = _rgb2hsv(img)
    h, s, v = img.unbind(0)
    h += hue_factor
    h = h % 1.0
    img = torch.stack((h, s, v))
    img_hue_adj = _hsv2rgb(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)
    return img_hue_adj


# https://github.com/TengdaHan/DPC/blob/master/utils/augmentation.py
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        grayscale (probablitities for rgb-to-gray 0~1)
        consistent (for video input whether the augment scale is consistent or not!)
        shuffle (shuffle the transform's order when there are multiple transform)
        gray_first (whether use grayscale or not)
        is_split (whether randomly chance the channel as the gray results.)
    """
    def __init__(self,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 grayscale=0,
                 consistent=False,
                 shuffle=True,
                 gray_first=True,
                 is_split=False):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue,
                                     'hue',
                                     center=0,
                                     bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.grayscale = grayscale
        self.consistent = consistent
        self.shuffle = shuffle
        self.gray_first = gray_first
        self.is_split = is_split

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    'If {} is a single number, it must be non negative.'.
                    format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError('{} values should be between {}'.format(
                    name, bound))
        else:
            raise TypeError(
                '{} should be a single number or a list/tuple with lenght 2.'.
                format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_transform(self, T, device):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Arg:
            T (int): number of frames. Used when consistent = False.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if self.brightness is not None:
            if self.consistent:
                brightness_factor = random.uniform(self.brightness[0],
                                                   self.brightness[1])
            else:
                brightness_factor = torch.empty([1, T, 1, 1],
                                                device=device).uniform_(
                                                    self.brightness[0],
                                                    self.brightness[1])
            transforms.append(
                Lambda(
                    lambda frame: adjust_brightness(frame, brightness_factor)))

        if self.contrast is not None:
            if self.consistent:
                contrast_factor = random.uniform(self.contrast[0],
                                                 self.contrast[1])
            else:
                contrast_factor = torch.empty([1, T, 1, 1],
                                              device=device).uniform_(
                                                  self.contrast[0],
                                                  self.contrast[1])
            transforms.append(
                Lambda(lambda frame: adjust_contrast(frame, contrast_factor)))

        if self.saturation is not None:
            if self.consistent:
                saturation_factor = random.uniform(self.saturation[0],
                                                   self.saturation[1])
            else:
                saturation_factor = torch.empty([1, T, 1, 1],
                                                device=device).uniform_(
                                                    self.saturation[0],
                                                    self.saturation[1])
            transforms.append(
                Lambda(
                    lambda frame: adjust_saturation(frame, saturation_factor)))

        if self.hue is not None:
            if self.consistent:
                hue_factor = random.uniform(self.hue[0], self.hue[1])
            else:
                hue_factor = torch.empty([T, 1, 1], device=device).uniform_(
                    self.hue[0], self.hue[1])
            transforms.append(
                Lambda(lambda frame: adjust_hue(frame, hue_factor)))

        if self.shuffle:
            random.shuffle(transforms)

        if random.uniform(0, 1) < self.grayscale:
            gray_transform = Lambda(
                lambda frame: rgb_to_grayscale(frame, split=self.is_split))
            if self.gray_first:
                transforms.insert(0, gray_transform)
            else:
                transforms.append(gray_transform)

        transform = Compose(transforms)
        return transform

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """

        is_frame = False

        if len(clip.shape) == 3:
            # frame data, transfer to 4dim
            clip = torch.unsqueeze(clip, dim=1)
            is_frame = True
        # (C, T, H, W)
        raw_shape = clip.shape
        device = clip.device
        T = raw_shape[1]
        transform = self._get_transform(T, device)
        clip = transform(clip)
        assert clip.shape == raw_shape
        if is_frame:
            clip = torch.squeeze(clip)
        return clip  # (C, T, H, W)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', grayscale={0})'.format(self.grayscale)
        return format_string


@TRANSFORMS.register_class()
class ColorJitterGeneral(ImageTransform):
    '''
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        grayscale (probablitities for rgb-to-gray 0~1)
        consistent (for video input whether the augment scale is consistent or not!)
        shuffle (shuffle the transform's order when there are multiple transform)
        gray_first (whether use grayscale or not)
        is_split (whether randomly chance the channel as the gray results.)
    '''
    para_dict = [{
        'BRIGHTNESS': {
            'value':
            0,
            'description':
            '(float or tuple of float (min, max)): How much to jitter brightness.'
            'brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]'
            'or the given [min, max]. Should be non negative numbers.'
        },
        'CONTRAST': {
            'value':
            0,
            'description':
            '(float or tuple of float (min, max)): How much to jitter contrast.'
            'contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]'
            'or the given [min, max]. Should be non negative numbers.'
        },
        'SATURATION': {
            'value':
            0,
            'description':
            '(float or tuple of float (min, max)): How much to jitter saturation.'
            'saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]'
            'or the given [min, max]. Should be non negative numbers.'
        },
        'HUE': {
            'value':
            0,
            'description':
            '(float or tuple of float (min, max)): How much to jitter hue.'
            'hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].'
            'Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.'
        },
        'GRAYSCALE': {
            'value': 0,
            'description': '(probablitities for rgb-to-gray 0~1)'
        },
        'CONSISTENT': {
            'value':
            False,
            'description':
            '(for video input whether the augment scale is consistent or not!)'
        },
        'SHUFFLE': {
            'value':
            False,
            'description':
            "(shuffle the transform's order when there are multiple transform)"
        },
        'GRAY_FIRST': {
            'value': False,
            'description': '(whether use grayscale or not)'
        },
        'IS_SPLIT': {
            'value':
            False,
            'description':
            '(whether randomly chance the channel as the gray results.)'
        }
    }]
    para_dict[0].update(ImageTransform.para_dict[0])
    para_dict[0].pop('BACKEND')

    def __init__(self, cfg, logger=None):
        super(ColorJitterGeneral, self).__init__(cfg, logger=None)
        # brightness=0, contrast=0, saturation=0, hue=0, grayscale=0,
        # consistent=False, shuffle=True, gray_first=True, is_split=False
        brightness = cfg.get('BRIGHTNESS', 0)
        contrast = cfg.get('CONTRAST', 0)
        saturation = cfg.get('SATURATION', 0)
        hue = cfg.get('HUE', 0)
        grayscale = cfg.get('GRAYSCALE', 0)
        consistent = cfg.get('CONSISTENT', False)
        shuffle = cfg.get('SHUFFLE', False)
        gray_first = cfg.get('GRAY_FIRST', False)
        is_split = cfg.get('IS_SPLIT', False)

        cj_ins = ColorJitter(brightness=brightness,
                             contrast=contrast,
                             saturation=saturation,
                             hue=hue,
                             grayscale=grayscale,
                             consistent=consistent,
                             shuffle=shuffle,
                             gray_first=gray_first,
                             is_split=is_split)
        self.callable = cj_ins

    def __call__(self, item):
        item[self.output_key] = self.callable(item[self.input_key])
        # item["meta"]["normalize_params"] = dict(mean=self.mean, std=self.std)
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
                            ColorJitterGeneral.para_dict,
                            set_name=True)
