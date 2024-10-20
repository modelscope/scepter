# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from .node_utils import load_example_image

from .constant import WORKFLOW_CONFIG

class TunerNode:
    def __init__(self):
        self.tuner_info = WORKFLOW_CONFIG.tuner_info

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        tuner_name = list(s().tuner_info.keys())
        return {
            'required': {
                'tuner': (tuner_name, ),
            },
            'optional': {
                'tuner_scale': (
                    'FLOAT',
                    {
                        'default': 1,
                        'min': 0,
                        'max': 1,
                        'step': 0.05
                    },
                ),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ('CONDITIONING', )
    RETURN_NAMES = ('Result', )
    FUNCTION = 'execute'

    def execute(self, tuner, tuner_scale):
        out = {
            "tuner_info": self.tuner_info[tuner],
            "tuner_scale": tuner_scale
        }
        return (out, )
