# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers

import numpy as np
import opencv_transforms.functional as cv2_TF
import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image

from scepter.modules.transform import TRANSFORMS, ImageTransform
from scepter.modules.transform.image import BACKENDS
from scepter.modules.transform.utils import BACKEND_PILLOW, BACKEND_TORCHVISION
from scepter.modules.utils.config import dict_to_yaml


@TRANSFORMS.register_class()
class FlexibleCropXL(ImageTransform):
    para_dict = [{
        'IS_CENTER': {
            'value': False,
            'description': 'Use center crop or not.'
        }
    }]
    para_dict[0].update(ImageTransform.para_dict[0])

    def __init__(self, cfg, logger=None):
        super(FlexibleCropXL, self).__init__(cfg, logger=logger)
        assert self.backend in BACKENDS
        self.size = cfg.get('SIZE', None)
        self.is_center = cfg.get('IS_CENTER', False)
        if self.size is not None:
            if isinstance(self.size, numbers.Number):
                self.size = [self.size, self.size]
        self.callable = TF.crop if self.backend in (
            BACKEND_PILLOW, BACKEND_TORCHVISION) else cv2_TF.crop

    def __call__(self, item):
        if isinstance(self.input_key, str):
            self.input_key = [self.input_key]
        if isinstance(self.output_key, str):
            self.output_key = [self.output_key]
        meta = item.get('meta', {})
        for idx, key in enumerate(self.input_key):
            self.check_image_type(item[key])
            if isinstance(item[key], (torch.Tensor, np.ndarray)):
                if self.backend in (BACKEND_PILLOW, BACKEND_TORCHVISION):
                    w, h, c = item[key].shape
                else:
                    h, w, c = item[key].shape
            elif isinstance(item[key], Image):
                w, h = item[key].size
            if 'image_size' in meta:
                oh, ow = meta['image_size']
            else:
                assert self.size is not None
                oh, ow = self.size
                meta['image_size'] = [oh, ow]

            delta_h = h - oh
            delta_w = w - ow
            if not self.is_center:
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            else:
                top = delta_h // 2
                left = delta_w // 2

            item[self.output_key[idx]] = self.callable(item[key], top, left,
                                                       oh, ow)
            item[self.output_key[idx] + '_' +
                 'original_size_as_tuple'] = torch.tensor([h, w])
            item[self.output_key[idx] + '_' +
                 'target_size_as_tuple'] = torch.tensor([oh, ow])
            item[self.output_key[idx] + '_' +
                 'crop_coords_top_left'] = torch.tensor([top, left])
        return item

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            FlexibleCropXL.para_dict,
                            set_name=True)
