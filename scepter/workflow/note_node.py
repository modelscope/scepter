# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

class NoteNode:
    def __init__(self):
        pass

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'Notebook': ('STRING', {'multiline': True})}}

    OUTPUT_NODE = False
    RETURN_TYPES = ()
    RETURN_NAMES = ()
