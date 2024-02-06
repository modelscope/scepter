# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# MLSD Line Detection
# From https://github.com/navervision/mlsd
# Apache-2.0 license

import warnings
from abc import ABCMeta

import cv2
import numpy as np
import torch
from PIL import Image

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.mlsd.mbv2_mlsd_large import MobileV2_MLSD_Large
from scepter.modules.annotator.mlsd.utils import pred_lines
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.annotator.utils import resize_image, resize_image_ori
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


@ANNOTATORS.register_class()
class MLSDdetector(BaseAnnotator, metaclass=ABCMeta):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        model = MobileV2_MLSD_Large()
        pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                model.load_state_dict(torch.load(local_path), strict=True)
        self.model = model.eval()
        self.thr_v = cfg.get('THR_V', 0.1)
        self.thr_d = cfg.get('THR_D', 0.1)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        h, w, c = image.shape
        image, k = resize_image(image, 1024 if min(h, w) > 1024 else min(h, w))
        img_output = np.zeros_like(image)
        try:
            lines = pred_lines(image,
                               self.model, [image.shape[0], image.shape[1]],
                               self.thr_v,
                               self.thr_d,
                               device=we.device_id)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end),
                         [255, 255, 255], 1)
        except Exception as e:
            warnings.warn(f'{e}')
            return None
        img_output = resize_image_ori(h, w, img_output, k)
        return img_output[:, :, 0]

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            MLSDdetector.para_dict,
                            set_name=True)
