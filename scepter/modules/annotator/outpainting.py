# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random
from abc import ABCMeta

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml


@ANNOTATORS.register_class()
class OutpaintingAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.mask_blur = cfg.get('MASK_BLUR', 0)
        self.random_cfg = cfg.get('RANDOM_CFG', None)
        self.return_mask = cfg.get('RETURN_MASK', False)
        self.return_source = cfg.get('RETURN_SOURCE', True)
        self.keep_padding_ratio = cfg.get('KEEP_PADDING_RATIO', 64)
        self.mask_color = cfg.get('MASK_COLOR', 0)

    def get_box(self, mask):
        locs = np.where(mask == 255)
        if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
            return None
        left, right = np.min(locs[1]), np.max(locs[1])
        top, bottom = np.min(locs[0]), np.max(locs[0])
        return [left, top, right, bottom]

    def forward(self,
                image,
                ratio=0.3,
                mask=None,
                direction=['left', 'right', 'up', 'down'],
                return_mask=None,
                return_source=None,
                mask_color=None):
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = return_source if return_source is not None else self.return_source
        mask_color = mask_color if mask_color is not None else self.mask_color
        if isinstance(image, Image.Image):
            image = image
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.detach().cpu().numpy())
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.copy())
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        if self.random_cfg:
            direction_range = self.random_cfg.get(
                'DIRECTION_RANGE', ['left', 'right', 'up', 'down'])
            ratio_range = self.random_cfg.get('RATIO_RANGE', [0.0, 1.0])
            direction = random.sample(
                direction_range,
                random.choice(list(range(1,
                                         len(direction_range) + 1))))
            ratio = random.uniform(ratio_range[0], ratio_range[1])

        if mask is None:
            init_image = image
            src_width, src_height = init_image.width, init_image.height
            left = int(ratio * src_width) if 'left' in direction else 0
            right = int(ratio * src_width) if 'right' in direction else 0
            up = int(ratio * src_height) if 'up' in direction else 0
            down = int(ratio * src_height) if 'down' in direction else 0
            # print(direction, ratio, left, right, up, down)
            tar_width = math.ceil(
                (src_width + left + right) /
                self.keep_padding_ratio) * self.keep_padding_ratio
            tar_height = math.ceil(
                (src_height + up + down) /
                self.keep_padding_ratio) * self.keep_padding_ratio
            if left > 0:
                left = left * (tar_width - src_width) // (left + right)
            if right > 0:
                right = tar_width - src_width - left
            if up > 0:
                up = up * (tar_height - src_height) // (up + down)
            if down > 0:
                down = tar_height - src_height - up
            if mask_color is not None:
                img = Image.new('RGB', (tar_width, tar_height), color=mask_color)
            else:
                img = Image.new('RGB', (tar_width, tar_height))
            img.paste(init_image, (left, up))
            mask = Image.new('L', (img.width, img.height), 'white')
            draw = ImageDraw.Draw(mask)

            draw.rectangle(
                (left + (self.mask_blur * 2 if left > 0 else 0), up +
                 (self.mask_blur * 2 if up > 0 else 0), mask.width - right -
                 (self.mask_blur * 2 if right > 0 else 0), mask.height - down -
                 (self.mask_blur * 2 if down > 0 else 0)),
                fill='black')
        else:
            bbox = self.get_box(np.array(mask))
            if bbox is None:
                img = image
                mask = mask
                init_image = image
            else:
                mask = Image.new('L', (image.width, image.height), 'white')
                mask_zero = Image.new('L', (bbox[2]-bbox[0], bbox[3]-bbox[1]), 'black')
                mask.paste(mask_zero, (bbox[0], bbox[1]))
                crop_image = image.crop(bbox)
                init_image = Image.new('RGB', (image.width, image.height), 'black')
                init_image.paste(crop_image, (bbox[0], bbox[1]))
                img = image
        if return_mask:
            if return_source:
                ret_data = {'src_image': np.array(init_image), 'image': np.array(img), 'mask': np.array(mask)}
            else:
                ret_data = {'image': np.array(img), 'mask': np.array(mask)}
        else:
            if return_source:
                ret_data = {'src_image': np.array(init_image), 'image': np.array(img)}
            else:
                ret_data = np.array(img)
        return ret_data

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            OutpaintingAnnotator.para_dict,
                            set_name=True)

@ANNOTATORS.register_class()
class OutpaintingResize(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

    def get_box(self, mask):
        locs = np.where(mask == 0)
        if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
            return None
        left, right = np.min(locs[1]), np.max(locs[1])
        top, bottom = np.min(locs[0]), np.max(locs[0])
        return [left, top, right, bottom]

    def forward(self,
                image,
                target_image,
                mask=None
                ):
        if isinstance(image, Image.Image):
            image = image
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.detach().cpu().numpy())
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.copy())
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        if isinstance(target_image, Image.Image):
            target_image = target_image
        elif isinstance(target_image, torch.Tensor):
            target_image = Image.fromarray(target_image.detach().cpu().numpy())
        elif isinstance(target_image, np.ndarray):
            target_image = Image.fromarray(target_image.copy())
        else:
            raise f'Unsurpport datatype{type(target_image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        bbox = self.get_box(np.array(mask))
        if bbox is None:
            init_image = image
        else:
            paste_img = image.resize((bbox[2]-bbox[0], bbox[3]-bbox[1]))
            init_image = Image.new('RGB', (target_image.width, target_image.height), 'black')
            init_image.paste(paste_img, (bbox[0], bbox[1]))
        ret_data = {'src_image': np.array(init_image)}
        return ret_data

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            OutpaintingResize.para_dict,
                            set_name=True)
