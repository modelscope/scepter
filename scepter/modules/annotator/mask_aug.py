# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from abc import ABCMeta
from functools import partial

import numpy as np
import torch
from PIL import Image, ImageDraw

import cv2
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.draw import polygon


@ANNOTATORS.register_class()
class MaskDrawAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.task_type = cfg.get('TASK_TYPE', 'input_box')

    def forward(self, mask=None, image=None, input_box=None, task_type=None):
        task_type = task_type if task_type is not None else self.task_type

        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            elif isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            elif isinstance(mask, np.ndarray):
                mask = mask.copy()
            else:
                raise f'Unsurpport datatype{type(mask)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        if image is not None:
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            elif isinstance(image, np.ndarray):
                image = image.copy()
            else:
                raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        mask_shape = mask.shape
        if task_type == 'mask_point':
            scribble = mask.transpose(1, 0)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            hull = ConvexHull(centers)
            hull_vertices = centers[hull.vertices]
            rr, cc = polygon(hull_vertices[:, 1], hull_vertices[:, 0],
                             mask_shape)
            out_mask[rr, cc] = 255
        elif task_type == 'mask_box':
            scribble = mask.transpose(1, 0)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            out_mask[int(y_min):int(y_max) + 1,
                     int(x_min):int(x_max) + 1] = 255
            if image is not None:
                out_image = image[int(y_min):int(y_max) + 1,
                                  int(x_min):int(x_max) + 1]
        elif task_type == 'input_box':
            if isinstance(input_box, list):
                input_box = np.array(input_box)
            x_min, y_min, x_max, y_max = input_box
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            out_mask[int(y_min):int(y_max) + 1,
                     int(x_min):int(x_max) + 1] = 255
            if image is not None:
                out_image = image[int(y_min):int(y_max) + 1,
                                  int(x_min):int(x_max) + 1]
        elif task_type == 'mask':
            out_mask = mask
        else:
            raise NotImplementedError

        if image is not None:
            return out_image, out_mask
        else:
            return out_mask


@ANNOTATORS.register_class()
class MaskAugAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        # original / original_expand / hull / hull_expand / bbox / bbox_expand
        self.mask_cfg = cfg.get('MASK_CFG', [{
            'mode': 'original',
            'proba': 0.1
        }, {
            'mode': 'original_expand',
            'proba': 0.1
        }, {
            'mode': 'hull',
            'proba': 0.1
        }, {
            'mode': 'hull_expand',
            'proba': 0.1,
            'kwargs': {
                'expand_rate': 0.2
            }
        }, {
            'mode': 'bbox',
            'proba': 0.1
        }, {
            'mode': 'bbox_expand',
            'proba': 0.1,
            'kwargs': {
                'min_expand_rate': 0.2,
                'max_expand_rate': 0.5
            }
        }])
        self.mask_cfg = Config.get_dict(self.mask_cfg) if isinstance(
            self.mask_cfg, Config) else self.mask_cfg

    def forward(self, mask, mask_cfg=None):
        mask_cfg = mask_cfg if mask_cfg is not None else self.mask_cfg
        if not isinstance(mask, list):
            is_batch = False
            masks = [mask]
        else:
            is_batch = True
            masks = mask

        mask_func = self.get_mask_func(mask_cfg)
        # print(mask_func)
        aug_masks = []
        for submask in masks:
            mask = self.get_mask(submask)
            valid, large, h, w, bbox = self.get_mask_info(mask)
            # print(valid, large, h, w, bbox)
            if valid:
                mask = mask_func(mask, bbox, h, w)
            else:
                mask = mask.astype(np.uint8)
            aug_masks.append(mask)
        return aug_masks if is_batch else aug_masks[0]

    def get_mask(self, mask):
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        elif isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask = mask.copy()
        else:
            raise f'Unsurpport datatype{type(mask)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        return mask

    def get_mask_info(self, mask):
        h, w = mask.shape
        locs = mask.nonzero()
        valid = True
        if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
            valid = False
            return valid, False, h, w, [0, 0, 0, 0]

        left, right = np.min(locs[1]), np.max(locs[1])
        top, bottom = np.min(locs[0]), np.max(locs[0])
        bbox = [left, top, right, bottom]

        large = False
        if (right - left + 1) * (bottom - top + 1) > 0.9 * h * w:
            large = True
        return valid, large, h, w, bbox

    def get_expand_params(self, mask_kwargs):
        if 'expand_rate' in mask_kwargs:
            expand_rate = mask_kwargs['expand_rate']
        elif 'min_expand_rate' in mask_kwargs and 'max_expand_rate' in mask_kwargs:
            expand_rate = random.uniform(mask_kwargs['min_expand_rate'],
                                         mask_kwargs['max_expand_rate'])
        else:
            expand_rate = 0.3

        if 'expand_iters' in mask_kwargs:
            expand_iters = mask_kwargs['expand_iters']
        else:
            expand_iters = random.randint(1, 10)

        if 'expand_lrtp' in mask_kwargs:
            expand_lrtp = mask_kwargs['expand_lrtp']
        else:
            expand_lrtp = [
                random.random(),
                random.random(),
                random.random(),
                random.random()
            ]

        return expand_rate, expand_iters, expand_lrtp

    def get_mask_func(self, mask_cfg):
        if not isinstance(mask_cfg, list):
            mask_cfg = [mask_cfg]
        probas = [
            item['proba'] if 'proba' in item else 1.0 / len(mask_cfg)
            for item in mask_cfg
        ]
        sel_mask_cfg = random.choices(mask_cfg, weights=probas, k=1)[0]
        mode = sel_mask_cfg['mode'] if 'mode' in sel_mask_cfg else 'original'
        mask_kwargs = sel_mask_cfg[
            'kwargs'] if 'kwargs' in sel_mask_cfg else {}

        if mode == 'random':
            mode = random.choice([
                'original', 'original_expand', 'hull', 'hull_expand', 'bbox',
                'bbox_expand'
            ])
        if mode == 'original':
            mask_func = partial(self.generate_mask)
        elif mode == 'original_expand':
            expand_rate, expand_iters, expand_lrtp = self.get_expand_params(
                mask_kwargs)
            mask_func = partial(self.generate_mask,
                                expand_rate=expand_rate,
                                expand_iters=expand_iters,
                                expand_lrtp=expand_lrtp)
        elif mode == 'hull':
            clockwise = random.choice([
                True, False
            ]) if 'clockwise' not in mask_kwargs else mask_kwargs['clockwise']
            mask_func = partial(self.generate_hull_mask, clockwise=clockwise)
        elif mode == 'hull_expand':
            expand_rate, expand_iters, expand_lrtp = self.get_expand_params(
                mask_kwargs)
            clockwise = random.choice([
                True, False
            ]) if 'clockwise' not in mask_kwargs else mask_kwargs['clockwise']
            mask_func = partial(self.generate_hull_mask,
                                clockwise=clockwise,
                                expand_rate=expand_rate,
                                expand_iters=expand_iters,
                                expand_lrtp=expand_lrtp)
        elif mode == 'bbox':
            mask_func = partial(self.generate_bbox_mask)
        elif mode == 'bbox_expand':
            expand_rate, expand_iters, expand_lrtp = self.get_expand_params(
                mask_kwargs)
            mask_func = partial(self.generate_bbox_mask,
                                expand_rate=expand_rate,
                                expand_iters=expand_iters,
                                expand_lrtp=expand_lrtp)
        else:
            raise NotImplementedError
        return mask_func

    def generate_mask(self,
                      mask,
                      bbox,
                      h,
                      w,
                      expand_rate=None,
                      expand_iters=None,
                      expand_lrtp=None):
        bin_mask = mask.astype(np.uint8)
        if expand_rate:
            bin_mask = self.rand_expand_mask(bin_mask, bbox, h, w, expand_rate,
                                             expand_iters, expand_lrtp)
        return bin_mask

    @staticmethod
    def rand_expand_mask(mask,
                         bbox,
                         h,
                         w,
                         expand_rate=None,
                         expand_iters=None,
                         expand_lrtp=None):
        expand_rate = 0.3 if expand_rate is None else expand_rate
        expand_iters = random.randint(
            1, 10) if expand_iters is None else expand_iters
        expand_lrtp = [
            random.random(),
            random.random(),
            random.random(),
            random.random()
        ] if expand_lrtp is None else expand_lrtp
        # print('iters', expand_iters, 'expand_rate', expand_rate, 'expand_lrtp', expand_lrtp)
        # mask = np.squeeze(mask)
        left, top, right, bottom = bbox
        # mask expansion
        box_w = (right - left + 1) * expand_rate
        box_h = (bottom - top + 1) * expand_rate
        left_, right_ = int(
            expand_lrtp[0] * min(box_w, left / 2) / expand_iters), int(
                expand_lrtp[1] * min(box_w, (w - right) / 2) / expand_iters)
        top_, bottom_ = int(
            expand_lrtp[2] * min(box_h, top / 2) / expand_iters), int(
                expand_lrtp[3] * min(box_h, (h - bottom) / 2) / expand_iters)
        kernel_size = max(left_, right_, top_, bottom_)
        if kernel_size > 0:
            kernel = np.zeros((kernel_size * 2, kernel_size * 2),
                              dtype=np.uint8)
            new_left, new_right = kernel_size - right_, kernel_size + left_
            new_top, new_bottom = kernel_size - bottom_, kernel_size + top_
            kernel[new_top:new_bottom + 1, new_left:new_right + 1] = 1
            mask = mask.astype(np.uint8)
            mask = cv2.dilate(mask, kernel,
                              iterations=expand_iters).astype(np.uint8)
            # mask = new_mask - (mask / 2).astype(np.uint8)
        # mask = np.expand_dims(mask, axis=-1)
        return mask

    @staticmethod
    def _convexhull(image, clockwise):
        # print('clockwise', clockwise)
        contours, hierarchy = cv2.findContours(image, 2, 1)
        cnt = np.concatenate(contours)  # merge all regions
        hull = cv2.convexHull(cnt, clockwise=clockwise)
        hull = np.squeeze(hull, axis=1).astype(np.float32).tolist()
        hull = [tuple(x) for x in hull]
        return hull  # b, 1, 2

    def generate_hull_mask(self,
                           mask,
                           bbox,
                           h,
                           w,
                           clockwise=None,
                           expand_rate=None,
                           expand_iters=None,
                           expand_lrtp=None):
        clockwise = random.choice([True, False
                                   ]) if clockwise is None else clockwise
        hull = self._convexhull(mask, clockwise)
        mask_img = Image.new('L', (w, h), 0)
        pt_list = hull
        mask_img_draw = ImageDraw.Draw(mask_img)
        mask_img_draw.polygon(pt_list, fill=255)
        bin_mask = np.array(mask_img).astype(np.uint8)
        if expand_rate:
            bin_mask = self.rand_expand_mask(bin_mask, bbox, h, w, expand_rate,
                                             expand_iters, expand_lrtp)
        return bin_mask

    def generate_bbox_mask(self,
                           mask,
                           bbox,
                           h,
                           w,
                           expand_rate=None,
                           expand_iters=None,
                           expand_lrtp=None):
        left, top, right, bottom = bbox
        bin_mask = np.zeros((h, w), dtype=np.uint8)
        bin_mask[top:bottom + 1, left:right + 1] = 255
        if expand_rate:
            bin_mask = self.rand_expand_mask(bin_mask, bbox, h, w, expand_rate,
                                             expand_iters, expand_lrtp)
        return bin_mask


@ANNOTATORS.register_class()
class MaskLayoutAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        ram_tag_color = cfg.get('RAM_TAG_COLOR', None)
        default_color = cfg.get('DEFAULT_COLOR', [0, 0, 0])
        self.use_aug = cfg.get('USE_AUG', False)
        self.color_dict = {'default': tuple(default_color)}
        if ram_tag_color is not None:
            with FS.get_object(ram_tag_color) as object:
                lines = object.decode('utf-8').strip().split('\n')
            lines = [id_name_color.split('#;#') for id_name_color in lines]
            self.color_dict.update({
                id_name_color[1]: tuple(eval(id_name_color[2]))
                for id_name_color in lines
            })
        if self.use_aug:
            mask_aug_dict = {'NAME': 'MaskAugAnnotator'}
            mask_aug_cfg = Config(cfg_dict=mask_aug_dict, load=False)
            self.mask_aug_anno = ANNOTATORS.build(mask_aug_cfg)

    def find_contours(self, mask):
        # @mask: gray cv2 image
        # contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, canvas, contour, color):
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        canvas = cv2.drawContours(canvas, contour, -1, color, thickness=3)
        return canvas

    def get_mask(self, mask):
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        elif isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask = mask.copy()
        else:
            raise f'Unsurpport datatype{type(mask)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        return mask

    def forward(self, mask=None, color=None, label=None, mask_cfg=None):
        if not isinstance(mask, list):
            is_batch = False
            mask = [mask]
        else:
            is_batch = True

        if label is not None and label in self.color_dict:
            color = self.color_dict[label]
        elif color is not None:
            color = color
        else:
            color = self.color_dict['default']

        ret_data = []
        for sub_mask in mask:
            sub_mask = self.get_mask(sub_mask)
            if self.use_aug:
                sub_mask = self.mask_aug_anno(sub_mask, mask_cfg)
            canvas = np.ones((sub_mask.shape[0], sub_mask.shape[1], 3)) * 255
            contour = self.find_contours(sub_mask)
            frame = self.draw_contours(canvas, contour, color)
            ret_data.append(frame)

        if is_batch:
            return ret_data
        else:
            return ret_data[0]
