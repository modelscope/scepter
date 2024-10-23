# -*- coding: utf-8 -*-

import math
from abc import ABCMeta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
from einops import rearrange
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


class SketchNet(nn.Module):
    def __init__(self, mean, std):
        assert isinstance(mean, float) and isinstance(std, float)
        super().__init__()
        self.mean = mean
        self.std = std

        # layers
        self.layers = nn.Sequential(nn.Conv2d(1, 48, 5, 2, 2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(48, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 2, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, 2, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 1024, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, 1024, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, 1024, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, 1024, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, 512, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(256, 256, 4, 2, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(128, 128, 4, 2, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 48, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(48, 48, 4, 2, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(48, 24, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(24, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        """x: [B, 1, H, W] within range [0, 1]. Sketch pixels in dark color.
        """
        x = (x - self.mean) / self.std
        return self.layers(x)


@ANNOTATORS.register_class()
class SketchAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        self.model = SketchNet(mean=0.9664114577640158,
                               std=0.0858381272736797).eval()
        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                state = torch.load(local_path, map_location='cpu')
                self.model.load_state_dict(state)

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        is_batch = False if len(image.shape) == 3 else True
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                if torch.equal(image[:, :, 0], image[:, :, 1]) and torch.equal(
                        image[:, :, 1], image[:, :, 2]):
                    image = image[:, :, 0]
                else:
                    raise "Unsurpport input image's shape and each channel is different"
            elif len(image.shape) == 4:
                if (torch.equal(image[:, :, :, 0], image[:, :, :, 1])
                        and torch.equal(image[:, :, :, 1], image[:, :, :, 2])):
                    image = image[:, :, :, 0]
                else:
                    raise "Unsurpport input image's shape and each channel is different"
            if len(image.shape) == 2:
                image = rearrange(image, 'h w -> 1 h w')
                B, H, W = image.shape
            elif len(image.shape) == 3:
                B, H, W = image.shape
            else:
                raise "Unsurpport input image's shape"
        elif isinstance(image, np.ndarray):
            image = image.copy()
            if len(image.shape) == 3:
                if np.array_equal(image[:, :, 0],
                                  image[:, :, 1]) and np.array_equal(
                                      image[:, :, 1], image[:, :, 2]):
                    image = image[:, :, 0]
                else:
                    raise "Unsurpport input image's shape and each channel is different"
            elif len(image.shape) == 4:
                if (np.array_equal(image[:, :, :, 0], image[:, :, :, 1]) and
                        np.array_equal(image[:, :, :, 1], image[:, :, :, 2])):
                    image = image[:, :, :, 0]
                else:
                    raise "Unsurpport input image's shape and each channel is different"
            image = torch.from_numpy(image).float()
            if len(image.shape) == 2:
                image = rearrange(image, 'h w -> 1 1 h w')
            elif len(image.shape) == 3:
                image = rearrange(image, 'b h w -> b 1 h w')
            else:
                raise "Unsurpport input image's shape"
        else:
            raise "Unsurpport input image's type"
        image = image.float().div(255)
        image = image.to(we.device_id)
        edge = self.model(image)
        edge = edge.squeeze(dim=1)
        edge = (edge * 255.0).clip(0, 255)
        edge = edge.cpu().numpy()
        edge = edge.astype(np.uint8)
        if not is_batch:
            edge = edge.squeeze()
        return edge[..., None].repeat(3, -1)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            SketchAnnotator.para_dict,
                            set_name=True)
