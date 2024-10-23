# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from .constant import WORKFLOW_CONFIG

class ParameterNode:
    def __init__(self):
        self.cfg = WORKFLOW_CONFIG.workflow_config

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {},
            'optional': {
                'sample': (s().cfg['BASE_PARAMETERS']['SAMPLER'], ),
                'sample_steps': ('INT', {
                    'default': 50,
                    'min': 1,
                    'max': 100,
                    'step': 1
                }),
                'guide_scale': ('FLOAT', {
                    'default': 5,
                    'min': 0,
                    'max': 10,
                    'step': 0.5
                }),
                'guide_rescale': ('FLOAT', {
                    'default': 0.5,
                    'min': 0,
                    'max': 1,
                    'step': 0.1
                }),
                'discretization': (s().cfg['BASE_PARAMETERS']['DISCRETIZATION'], ),
                'output_height': (s().cfg['BASE_PARAMETERS']['OUTPUT_HEIGHT'], ),
                'output_width': (s().cfg['BASE_PARAMETERS']['OUTPUT_WIDTH'], ),
                'random_seed': ('INT', {
                    'default': -1,
                    'min': -1000000,
                    'max': 1000000
                }),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ('CONDITIONING', )
    RETURN_NAMES = ('Result', )
    FUNCTION = 'execute'

    def execute(self, sample, sample_steps, guide_scale, guide_rescale,
                discretization, output_height, output_width, random_seed):
        out = {
            'sample': sample,
            'sample_steps': sample_steps,
            'guide_scale': guide_scale,
            'guide_rescale': guide_rescale,
            'discretization': discretization,
            'target_size_as_tuple': [output_height, output_width],
            'seed': random_seed
        }
        return (out, )
