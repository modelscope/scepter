# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from scepter.modules.solver.hooks.backward import BackwardHook
from scepter.modules.solver.hooks.checkpoint import CheckpointHook
from scepter.modules.solver.hooks.data_probe import ProbeDataHook
from scepter.modules.solver.hooks.ema import ModelEmaHook
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.log import LogHook, TensorboardLogHook
from scepter.modules.solver.hooks.lr import LrHook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.solver.hooks.safetensors import SafetensorsHook
from scepter.modules.solver.hooks.sampler import DistSamplerHook
"""
Normally, hooks have priorities, below we recommend priority that runs fine (low score MEANS high priority)
BackwardHook: 0
LogHook: 100
LrHook: 200
CheckpointHook: 300
SamplerHook: 400

Recommend sequences in training are:
before solve:
    TensorboardLogHook: prepare file handler
    CheckpointHook: resume checkpoint

before epoch:
    LogHook: clear epoch variables
    DistSamplerHook: change sampler seed

before iter:
    LogHook: record data time

after iter:
    BackwardHook: network backward
    LogHook: log
    TensorboardLogHook: log
    CheckpointHook: save checkpoint
    SafetensorsHook: save checkpoint

after epoch:
    LrHook: reset learning rate
    CheckpointHook: save checkpoint

after solve:
    TensorboardLogHook: close file handler
"""

__all__ = [
    'HOOKS', 'BackwardHook', 'CheckpointHook', 'Hook', 'LrHook', 'LogHook',
    'TensorboardLogHook', 'DistSamplerHook', 'ProbeDataHook',
    'SafetensorsHook', 'ModelEmaHook'
]
