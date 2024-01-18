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
class ColorAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.ratio = cfg.get('RATIO', 64)

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
        ratio = self.ratio
        image = cv2.resize(image, (w // ratio, h // ratio),
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
