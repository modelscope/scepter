# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

import cv2
import numpy as np
import torch
from PIL import Image

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml


@ANNOTATORS.register_class()
class CannyAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.low_threshold = cfg.get('LOW_THRESHOLD', 100)
        self.high_threshold = cfg.get('HIGH_THRESHOLD', 200)
        self.random_cfg = cfg.get('RANDOM_CFG', None)

    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        assert len(image.shape) < 4

        if self.random_cfg is None:
            image = cv2.Canny(image, self.low_threshold, self.high_threshold)
        else:
            proba = self.random_cfg.get('PROBA', 1.0)
            if np.random.random() < proba:
                min_low_threshold = self.random_cfg.get(
                    'MIN_LOW_THRESHOLD', 50)
                max_low_threshold = self.random_cfg.get(
                    'MAX_LOW_THRESHOLD', 100)
                min_high_threshold = self.random_cfg.get(
                    'MIN_HIGH_THRESHOLD', 200)
                max_high_threshold = self.random_cfg.get(
                    'MAX_HIGH_THRESHOLD', 350)
                low_th = np.random.randint(min_low_threshold,
                                           max_low_threshold)
                high_th = np.random.randint(min_high_threshold,
                                            max_high_threshold)
            else:
                low_th, high_th = self.low_threshold, self.high_threshold
            image = cv2.Canny(image, low_th, high_th)
        return image[..., None].repeat(3, 2)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            CannyAnnotator.para_dict,
                            set_name=True)
