# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Midas Depth Estimation
# From https://github.com/isl-org/MiDaS
# MIT LICENSE
from abc import ABCMeta

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.midas.api import MiDaSInference
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.annotator.utils import resize_image, resize_image_ori
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


@ANNOTATORS.register_class()
class MidasDetector(BaseAnnotator, metaclass=ABCMeta):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                self.model = MiDaSInference(model_type='dpt_hybrid',
                                            model_path=local_path)
        self.a = cfg.get('A', np.pi * 2.0)
        self.bg_th = cfg.get('BG_TH', 0.1)

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
        image_depth = image
        h, w, c = image.shape
        image_depth, k = resize_image(image_depth,
                                      1024 if min(h, w) > 1024 else min(h, w))
        image_depth = torch.from_numpy(image_depth).float().to(we.device_id)
        image_depth = image_depth / 127.5 - 1.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        depth = self.model(image_depth)[0]

        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        depth_image = depth_image[..., None].repeat(3, 2)

        # depth_np = depth.cpu().numpy()  # float16 error
        # x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        # y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
        # z = np.ones_like(x) * self.a
        # x[depth_pt < self.bg_th] = 0
        # y[depth_pt < self.bg_th] = 0
        # normal = np.stack([x, y, z], axis=2)
        # normal /= np.sum(normal**2.0, axis=2, keepdims=True)**0.5
        # normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        depth_image = resize_image_ori(h, w, depth_image, k)
        return depth_image

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            MidasDetector.para_dict,
                            set_name=True)
