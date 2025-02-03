# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import Image
import numpy as np
import math
import torch
import torchvision.transforms as T

from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.annotator.registry import ANNOTATORS
from .constant import WORKFLOW_CONFIG


class ACEPlusProcessorNode:
    def __init__(self,
                 max_aspect_ratio=4,
                 d=16,
                 processor=WORKFLOW_CONFIG.ace_plus_processor_config):
        self.max_aspect_ratio = max_aspect_ratio
        self.processor_cfg = processor
        self.task_list = {}
        self.d = d
        self.max_seq_len = processor.DEFAULT_PARAS.MAX_SEQ_LENGTH
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        for task in self.processor_cfg.PROCESSORS:
            self.task_list[task.TYPE] = task

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'ref_image': ('IMAGE',),
                'task_type': (list(s().task_list.keys()),),
                'repainting_scale': ('FLOAT', {
                    'default': 1,
                    'min': 0,
                    'max': 1,
                    'step': 0.01
                }),
            },
            'optional': {
                'edit_mask': ('MASK',),
                'edit_image': ('IMAGE',)
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('IMAGE', 'MASK')
    FUNCTION = 'execute'

    def execute(self,
                ref_image,
                task_type,
                edit_mask=None,
                edit_image=None,
                repainting_scale=1):
        if task_type != 'image_processor':
            edit_image = self.edit_preprocess(self.task_list[task_type], we.device_id,
                                              ref_image, edit_mask)
        return self.preprocess(ref_image, edit_image,
                               edit_mask, repainting_scale)

    def edit_preprocess(self, processor, device, edit_image, edit_mask):
        if edit_image is None or processor is None:
            return edit_image
        processor = Config(cfg_dict=processor, load=False)
        processor = ANNOTATORS.build(processor).to(device)
        edit_image = self.trans_tensor_pil(edit_image)
        new_edit_image = processor(np.asarray(edit_image))

        del processor
        new_edit_image = Image.fromarray(new_edit_image)
        to_pil = T.ToPILImage()
        edit_mask = to_pil(edit_mask)

        if new_edit_image.size != edit_image.size:
            edit_image = T.Resize((edit_image.size[1], edit_image.size[0]),
                                  interpolation=T.InterpolationMode.BILINEAR,
                                  antialias=True)(new_edit_image)
        image = Image.composite(new_edit_image, edit_image, edit_mask)
        return self.trans_pil_tensor(image)

    def trans_tensor_pil(self, tensor_image):
        image = tensor_image.squeeze(0).permute(2, 0, 1)
        to_pil = T.ToPILImage()
        return to_pil(image)

    def trans_pil_tensor(self, pil_image):
        transform = T.Compose([
            T.ToTensor()
        ])
        tensor_image = transform(pil_image)
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image = tensor_image.permute(0, 2, 3, 1)

        return tensor_image

    def image_check(self, image):
        if image is None:
            return image
        W, H = image.size
        if H / W > self.max_aspect_ratio:
            image = T.CenterCrop([int(self.max_aspect_ratio * W), W])(image)
        elif W / H > self.max_aspect_ratio:
            image = T.CenterCrop([H, int(self.max_aspect_ratio * H)])(image)
        return self.transforms(image)

    def denormalize(self, t):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        return t * std + mean

    def preprocess(self,
                   reference_image=None,
                   edit_image=None,
                   edit_mask=None,
                   repainting_scale=1.0):
        reference_image = self.trans_tensor_pil(reference_image) \
            if reference_image is not None else None
        width, height = reference_image.size
        edit_image = edit_image.squeeze(0).permute(2, 0, 1) \
            if edit_image is not None else None
        to_pil = T.ToPILImage()
        edit_mask = to_pil(edit_mask) if edit_mask is not None else None

        reference_image = self.image_check(reference_image)
        if edit_image is None:
            edit_image = torch.zeros([3, height, width])
            edit_mask = torch.ones([1, height, width])
        else:
            edit_mask = np.asarray(edit_mask)
            edit_mask = np.where(edit_mask > 128, 1, 0)
            edit_mask = edit_mask.astype(
                np.float32) if np.any(edit_mask) else np.ones_like(edit_mask).astype(
                np.float32)
            edit_mask = torch.tensor(edit_mask).unsqueeze(0)

        edit_image = edit_image * (1 - edit_mask * repainting_scale)

        assert edit_mask is not None
        if reference_image is not None:
            _, H, W = reference_image.shape
            _, eH, eW = edit_image.shape
            scale = eH / H
            tH, tW = eH, int(W * scale)
            reference_image = T.Resize((tH, tW),
                                       interpolation=T.InterpolationMode.BILINEAR,
                                       antialias=True)(reference_image)
            if repainting_scale == 1:
                reference_image = self.denormalize(reference_image)
            edit_image = torch.cat([reference_image, edit_image], dim=-1)
            edit_mask = torch.cat([torch.zeros([1, reference_image.shape[1],
                                                reference_image.shape[2]]), edit_mask], dim=-1)

        H, W = edit_image.shape[-2:]
        scale = min(1.0, math.sqrt(self.max_seq_len * 2 / ((H / self.d) * (W / self.d))))
        rH = int(H * scale) // self.d * self.d
        rW = int(W * scale) // self.d * self.d

        edit_image = T.Resize((rH, rW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(edit_image)
        edit_mask = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_mask)
        edit_image = edit_image.unsqueeze(0).permute(0, 2, 3, 1)

        return edit_image, edit_mask
