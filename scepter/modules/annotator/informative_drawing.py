# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from torchvision.transforms import InterpolationMode

norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ContourInference(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(ContourInference, self).__init__()

        # Initial convolution block
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features,
                                   out_features,
                                   3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


@ANNOTATORS.register_class()
class InfoDrawContourAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        input_nc = cfg.get('INPUT_NC', 3)
        output_nc = cfg.get('OUTPUT_NC', 1)
        n_residual_blocks = cfg.get('N_RESIDUAL_BLOCKS', 3)
        sigmoid = cfg.get('SIGMOID', True)
        pretrained_model = cfg.get('PRETRAINED_MODEL', None)

        self.model = ContourInference(input_nc, output_nc, n_residual_blocks,
                                      sigmoid)
        with FS.get_from(pretrained_model, wait_finish=True) as local_path:
            self.model.load_state_dict(torch.load(local_path))
        self.model = self.model.eval().requires_grad_(False).to(we.device_id)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        is_batch = False if len(image.shape) == 3 else True
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = rearrange(image, 'h w c -> 1 c h w')
                B, C, H, W = image.shape
            elif len(image.shape) == 4:
                B, C, H, W = image.shape
            else:
                raise "Unsurpport input image's shape"
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image.copy()).float()
            if len(image.shape) == 3:
                image = rearrange(image, 'h w c -> 1 c h w')
                B, C, H, W = image.shape
            elif len(image.shape) == 4:
                B, C, H, W = image.shape
            else:
                raise "Unsurpport input image's shape"
        else:
            raise "Unsurpport input image's type"

        image = image.float().div(255).to(we.device_id)
        contour_map = self.model(image)
        contour_map = (contour_map.squeeze(dim=1) * 255.0).clip(
            0, 255).cpu().numpy().astype(np.uint8)
        contour_map = contour_map[..., None].repeat(3, -1)
        if not is_batch:
            contour_map = contour_map.squeeze()
        return contour_map

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            InfoDrawContourAnnotator.para_dict,
                            set_name=True)


@ANNOTATORS.register_class()
class InfoDrawAnimeAnnotator(InfoDrawContourAnnotator):
    pass


@ANNOTATORS.register_class()
class InfoDrawOpenSketchAnnotator(InfoDrawContourAnnotator):
    pass
