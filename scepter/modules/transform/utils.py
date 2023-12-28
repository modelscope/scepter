# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
import torch
from packaging import version
from PIL import Image
from torchvision.version import __version__ as tv_version

try:
    import accimage
except ImportError:
    accimage = None


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def is_cv2_image(img):
    return isinstance(img, np.ndarray) and img.dtype == np.uint8


def is_tensor(t):
    return isinstance(t, torch.Tensor)


INPUT_PIL_TYPE_WARNING = 'input should be PIL Image'
INPUT_CV2_TYPE_WARNING = 'input should be cv2 image(uint8 np.ndarray)'
INPUT_TENSOR_TYPE_WARNING = 'input should be tensor(uint8 np.ndarray)'

# Recommend to use nn.Module backend to transform
TORCHVISION_CAPABILITY = version.parse(tv_version) >= version.parse('0.8.0')

BACKEND_TORCHVISION = 'torchvision'
BACKEND_PILLOW = 'pillow'
BACKEND_CV2 = 'cv2'

# Recommend to use InterpolationMode since torchvision 0.9.0
INTERPOLATION_MODE_CAPABILITY = version.parse(tv_version) >= version.parse(
    '0.9.0')
if INTERPOLATION_MODE_CAPABILITY:
    from torchvision.transforms.functional import InterpolationMode
else:
    import warnings

    warnings.filterwarnings('ignore', message='Default upsampling behavior.*')
INTERPOLATION_STYLE = {
    'bilinear':
    Image.BILINEAR
    if not INTERPOLATION_MODE_CAPABILITY else InterpolationMode('bilinear'),
    'nearest':
    Image.NEAREST
    if not INTERPOLATION_MODE_CAPABILITY else InterpolationMode('nearest'),
    'bicubic':
    Image.BICUBIC
    if not INTERPOLATION_MODE_CAPABILITY else InterpolationMode('bicubic'),
}
INTERPOLATION_STYLE_CV2 = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'bicubic': cv2.INTER_CUBIC,
}
