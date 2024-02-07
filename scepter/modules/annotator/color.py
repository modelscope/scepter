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
class ColorAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.ratio = cfg.get('RATIO', 64)
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
        h, w = image.shape[:2]

        if self.random_cfg is None:
            ratio = self.ratio
        else:
            proba = self.random_cfg.get('PROBA', 1.0)
            if np.random.random() < proba:
                if 'CHOICE_RATIO' in self.random_cfg:
                    ratio = np.random.choice(self.random_cfg['CHOICE_RATIO'])
                else:
                    min_ratio = self.random_cfg.get('MIN_RATIO', 48)
                    max_ratio = self.random_cfg.get('MAX_RATIO', 96)
                    ratio = np.random.randint(min_ratio, max_ratio)
            else:
                ratio = self.ratio
        image = cv2.resize(image, (int(w // ratio), int(h // ratio)),
                           interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        assert len(image.shape) < 4
        return image

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            ColorAnnotator.para_dict,
                            set_name=True)
