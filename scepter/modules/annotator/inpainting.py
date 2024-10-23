# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random
from abc import ABCMeta
from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config, dict_to_yaml


def invert_image(im):
    im_arr = np.array(im)
    mask_1 = im_arr == 0
    mask_2 = im_arr == 255
    im_arr[mask_1] = 255
    im_arr[mask_2] = 0
    new_im = im_arr
    return new_im


class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_irregular_mask(shape,
                               max_angle=4,
                               max_len=60,
                               max_width=20,
                               min_times=0,
                               max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip(
                (start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip(
                (start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0,
                         brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y),
                           radius=brush_w,
                           color=1.,
                           thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius,
                     start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    def __init__(self,
                 max_angle=4,
                 max_len=60,
                 max_width=20,
                 min_times=0,
                 max_times=10,
                 ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = None
        # self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (
            iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 +
                            (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(img.shape[1:],
                                          max_angle=self.max_angle,
                                          max_len=cur_max_len,
                                          max_width=cur_max_width,
                                          min_times=self.min_times,
                                          max_times=cur_max_times,
                                          draw_method=self.draw_method)


def make_random_rectangle_mask(shape,
                               margin=10,
                               bbox_min_size=30,
                               bbox_max_size=100,
                               min_times=0,
                               max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


class RandomRectangleMaskGenerator:
    def __init__(self,
                 margin=10,
                 bbox_min_size=30,
                 bbox_max_size=100,
                 min_times=0,
                 max_times=3,
                 ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = None
        # self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (
            iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 +
                                (self.bbox_max_size - self.bbox_min_size) *
                                coef)
        cur_max_times = int(self.min_times +
                            (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(img.shape[1:],
                                          margin=self.margin,
                                          bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size,
                                          min_times=self.min_times,
                                          max_times=cur_max_times)


class MixedMaskGenerator:
    def __init__(self,
                 irregular_proba=1 / 3,
                 irregular_kwargs=None,
                 box_proba=1 / 3,
                 box_kwargs=None,
                 invert_proba=0):
        self.probas = []
        self.gens = []

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    def __call__(self, img, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(img, iter_i=iter_i, raw_image=raw_image)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        return result


@ANNOTATORS.register_class()
class InpaintingAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.mask_cfg = cfg.get(
            'MASK_CFG', {
                'irregular_proba': 0.5,
                'irregular_kwargs': {
                    'min_times': 4,
                    'max_times': 10,
                    'max_width': 100,
                    'max_angle': 4,
                    'max_len': 200
                },
                'box_proba': 0.5,
                'box_kwargs': {
                    'margin': 0,
                    'bbox_min_size': 30,
                    'bbox_max_size': 150,
                    'max_times': 5,
                    'min_times': 1
                }
            })
        self.mask_cfg = Config.get_dict(self.mask_cfg) if not isinstance(
            self.mask_cfg, dict) else self.mask_cfg
        self.mask_generator = MixedMaskGenerator(**self.mask_cfg)
        self.return_mask = cfg.get('RETURN_MASK', False)
        self.return_invert = cfg.get('RETURN_INVERT', True)
        self.mask_color = cfg.get('MASK_COLOR', 0)

    def forward(self, image, mask=None, return_mask=None, return_invert=None, mask_color=None):
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_invert = return_invert if return_invert is not None else self.return_invert
        mask_color = mask_color if mask_color is not None else self.mask_color
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        if mask is not None:
            if mask_color:
                image[np.array(mask) == 255] = mask_color
            else:
                image[np.array(mask) == 255] = 0
        else:
            img = np.transpose(image, (2, 0, 1))
            mask = self.mask_generator(img)
            mask = (np.transpose(mask, (1, 2, 0)).squeeze(-1) * 255).astype(np.uint8)
            if return_invert:
                mask = invert_image(mask)
            colored_mask = np.zeros_like(image)
            if mask_color: colored_mask[:] = mask_color
            image = np.where(mask[:, :, np.newaxis] == 255, colored_mask, image)

        if return_mask:
            ret_data = {
                'image': np.array(image),
                'mask': np.array(mask)
            }
        else:
            ret_data = np.array(image)
        return ret_data

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            InpaintingAnnotator.para_dict,
                            set_name=True)
