# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from .model_node import ModelNode
from .note_node import NoteNode
from .parameter_node import ParameterNode
from .mantras_node import MantrasNode
from .tuner_node import TunerNode
from .control_node import ControlNode

NODE_MAPPINGS = {
    'ModelNode': ('🪄 ScepterModel~', ModelNode),
    'NoteNode': ('🪄 ScepterNote~', NoteNode),
    'ParameterNode': ('🪄 ScepterParameter~', ParameterNode),
    'MantrasNode': ('🪄 ScepterMantra~', MantrasNode),
    'TunerNode': ('🪄 ScepterTuner~', TunerNode),
    'ControlNode': ('🪄 ScepterControl~', ControlNode)
}

NODE_CLASS_MAPPINGS = {k : v[1] for k, v in NODE_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k : v[0] for k, v in NODE_MAPPINGS.items()}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
