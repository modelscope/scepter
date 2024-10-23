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
class GrayAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        gray_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_map[..., None].repeat(3, axis=2)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            GrayAnnotator.para_dict,
                            set_name=True)
