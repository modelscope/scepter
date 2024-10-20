# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random
from abc import ABCMeta

import cv2
import numpy as np
import torch

from PIL import Image
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config, dict_to_yaml


def gaussian_noise_op(im, v):
    from basicsr.data.degradations import random_add_gaussian_noise
    noise_level = v.get('noise_level', [10, 20])
    out = random_add_gaussian_noise(
        im,
        sigma_range=noise_level,
        clip=True,
        rounds=False,
        gray_prob=0.4,
    )
    out = np.clip(out, 0.0, 1.0)
    return out


def resize_op(im, v):
    scale = v.get('scale', [0.5, 0.8])
    h, w = im.shape[:2]
    scale = random.uniform(scale[0], scale[1])
    h_, w_ = int(h * scale), int(w * scale)
    mode = v.get('mode', 'nearest')
    if mode == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif mode == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif mode == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_NEAREST
    im = cv2.resize(im, (w_, h_), interpolation=interpolation)
    out = cv2.resize(im, (w, h), interpolation=interpolation)
    out = np.clip(out, 0.0, 1.0)
    return out


def jpeg_op(im, v):
    from basicsr.data.degradations import add_jpg_compression
    jpeg_level = v.get('jpeg_level', [50, 75])
    v = int(random.uniform(jpeg_level[0], jpeg_level[1]))
    out = add_jpg_compression(im, v)
    out = np.clip(out, 0.0, 1.0)
    return out


def gaussian_blur_op(im, v):
    from basicsr.data.degradations import random_mixed_kernels
    kernel_range = v.get('kernel_size', [7, 9])
    kernel_size = random.choice(kernel_range)
    kernel_size = min(int(kernel_size) // 2 * 2 + 1, 21)
    blur_sigma = v.get('sigma', [0.9, 1.0])
    kernel = random_mixed_kernels(
        ('iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
         'plateau_aniso'), (0.45, 0.25, 0.12, 0.03, 0.12, 0.03),
        kernel_size,
        blur_sigma,
        blur_sigma, [-math.pi, math.pi], [0.5, 2.0], [1, 1.5],
        noise_range=None)

    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    out = cv2.filter2D(im, -1, kernel)
    out = np.clip(out, 0.0, 1.0)
    return out


@ANNOTATORS.register_class()
class DegradationAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.params = cfg.get('PARAMS', {
            'gaussian_noise': {},
            'resize': {},
            'jpeg': {},
            'gaussian_blur': {},
        })
        if not isinstance(self.params, dict):
            self.params = Config.get_dict(self.params)
        self.random_degradation = cfg.get('RANDOM_DEGRADATION', False)

    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        if np.max(image) > 1.0:
            image = (image / 255.).astype(np.float32)

        degradation_list = list(self.params.keys())
        if self.random_degradation:
            random.shuffle(degradation_list)

        for degradation_type in degradation_list:
            if degradation_type == 'gaussian_noise':
                image = gaussian_noise_op(image, self.params[degradation_type])
            elif degradation_type == 'resize':
                image = resize_op(image, self.params[degradation_type])
            elif degradation_type == 'jpeg':
                image = jpeg_op(image, self.params[degradation_type])
            elif degradation_type == 'gaussian_blur':
                image = gaussian_blur_op(image, self.params[degradation_type])
            else:
                raise NotImplementedError(
                    f'ERROR: degradation_type: {degradation_type} is invalid.')
        image = (image * 255.0).astype(np.uint8)

        assert len(image.shape) < 4
        return image

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            DegradationAnnotator.para_dict,
                            set_name=True)
