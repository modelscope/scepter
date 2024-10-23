# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import numpy as np
from PIL import Image
import torchvision.transforms as TT
import torch

from .constant import WORKFLOW_CONFIG


class ControlNode:
    def __init__(self):
        self.annotators = {}
        self.anno_info = WORKFLOW_CONFIG.anno_info
        for tp, anno in self.anno_info.items():
            self.annotators[tp] = {
                'cfg': anno,
                'device': 'offline',
                'model': None
            }
        self.control_info = WORKFLOW_CONFIG.control_info

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'source_image': ('IMAGE', ),
                'control_model': (list(s().control_info.keys()), ),
                'control_preprocessor': (list(s().annotators.keys()), ),
                'crop_type': (['CenterCrop', 'NoCrop'], ),
            },
            'optional': {
                'control_scale': ('FLOAT', {
                    'default': 1,
                    'min': 0,
                    'max': 1,
                    'step': 0.05
                }),
                'output_height': ('INT', {
                    'default': 1024,
                    'min': 256,
                    'max': 2048,
                }),
                'output_width': ('INT', {
                    'default': 1024,
                    'min': 256,
                    'max': 2048,
                }),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ('CONDITIONING', 'IMAGE')
    RETURN_NAMES = ('Result', 'Control Image')
    FUNCTION = 'execute'

    def execute(self, source_image, control_model, control_preprocessor,
                crop_type, control_scale, output_height, output_width):
        source_image = TT.ToPILImage()(source_image.squeeze(0).permute(2, 0, 1))
        cond_image = self.extract_condition(source_image, control_preprocessor,
                                            crop_type, output_height, output_width)
        cond_image_pil = Image.fromarray(cond_image)
        cond_image_show = torch.from_numpy(cond_image).float().unsqueeze(0)
        ctr_model = self.control_info[control_model]
        out = {
            "control_model": ctr_model,
            "crop_type": crop_type,
            "control_scale": control_scale,
            "control_cond_image": cond_image_pil
        }
        return (out, cond_image_show, )

    def extract_condition(self, source_image, control_mode, crop_type,
                          output_height, output_width):
        annotator = self.annotators[control_mode]
        annotator = self.load_annotator(annotator)

        if crop_type == 'CenterCrop':
            source_image = TT.Resize(max(output_height,
                                         output_width))(source_image)
            source_image = TT.CenterCrop(
                (output_height, output_width))(source_image)
        cond_image = annotator['model'](np.array(source_image))
        self.annotators[control_mode] = self.unload_annotator(annotator)

        if cond_image is None:
            raise RuntimeError('Pre-process error!')
        return cond_image

    def load_annotator(self, annotator):
        from scepter.modules.annotator.registry import ANNOTATORS
        from scepter.modules.utils.distribute import we

        if annotator['device'] == 'offline':
            annotator['model'] = ANNOTATORS.build(annotator['cfg'])
            annotator['device'] = 'cpu'
        if annotator['device'] == 'cpu':
            annotator['model'] = annotator['model'].to(we.device_id)
            annotator['device'] = we.device_id
        return annotator

    def unload_annotator(self, annotator):
        if not annotator['device'] == 'offline' and not annotator[
                'device'] == 'cpu':
            annotator['model'] = annotator['model'].to('cpu')
            annotator['device'] = 'cpu'
        return annotator