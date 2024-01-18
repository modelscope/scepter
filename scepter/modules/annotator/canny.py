# -*- coding: utf-8 -*-
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

    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.Canny(image, self.low_threshold, self.high_threshold)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            image = cv2.Canny(image, self.low_threshold, self.high_threshold)
        elif isinstance(image, np.ndarray):
            image = cv2.Canny(image.copy(), self.low_threshold,
                              self.high_threshold)
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        assert len(image.shape) < 4
        return image[..., None].repeat(3, 2)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            CannyAnnotator.para_dict,
                            set_name=True)
