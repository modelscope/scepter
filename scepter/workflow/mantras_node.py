# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from .node_utils import load_example_image

from .constant import WORKFLOW_CONFIG

class MantrasNode:
    def __init__(self):
        self.mantra_info = WORKFLOW_CONFIG.mantra_info

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        mantras_styles = list(s().mantra_info.keys())
        return {'required': {'mantra_styles': (mantras_styles, )}}

    OUTPUT_NODE = True
    RETURN_TYPES = ('CONDITIONING', )
    RETURN_NAMES = ('Result', )
    FUNCTION = 'execute'

    def execute(self, mantra_styles):
        info = self.mantra_info[mantra_styles]
        out = {
            'prompt_template': info['PROMPT'],
            'negative_prompt_template': info['NEGATIVE_PROMPT'],
        }
        return (out, )
