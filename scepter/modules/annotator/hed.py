# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

from abc import ABCMeta

import cv2
import numpy as np
import torch
from einops import rearrange

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(in_channels=input_channel,
                            out_channels=output_channel,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1))
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(in_channels=output_channel,
                                out_channels=output_channel,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel,
                                          out_channels=1,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h,
                                               kernel_size=(2, 2),
                                               stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3,
                                      output_channel=64,
                                      layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64,
                                      output_channel=128,
                                      layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128,
                                      output_channel=256,
                                      layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256,
                                      output_channel=512,
                                      layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512,
                                      output_channel=512,
                                      layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


@ANNOTATORS.register_class()
class HedAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.netNetwork = ControlNetHED_Apache2().float().eval()
        pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                self.netNetwork.load_state_dict(torch.load(local_path))

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = rearrange(image, 'h w c -> 1 c h w')
                B, C, H, W = image.shape
            else:
                raise "Unsurpport input image's shape"
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image.copy()).float()
            if len(image.shape) == 3:
                image = rearrange(image, 'h w c -> 1 c h w')
                B, C, H, W = image.shape
            else:
                raise "Unsurpport input image's shape"
        else:
            raise "Unsurpport input image's type"
        edges = self.netNetwork(image.to(we.device_id))
        edges = [
            e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges
        ]
        edges = [
            cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR)
            for e in edges
        ]
        edges = np.stack(edges, axis=2)
        edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        edge = 255 - (edge * 255.0).clip(0, 255).astype(np.uint8)
        return edge[..., None].repeat(3, 2)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            HedAnnotator.para_dict,
                            set_name=True)
