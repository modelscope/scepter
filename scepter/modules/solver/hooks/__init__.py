# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule
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


if TYPE_CHECKING:
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
    from scepter.modules.solver.hooks.val_loss import ValLossHook
else:
    _import_structure = {
        'backward': ['BackwardHook'],
        'checkpoint': ['CheckpointHook'],
        'data_probe': ['ProbeDataHook'],
        'ema': ['ModelEmaHook'],
        'hook': ['Hook'],
        'log': ['LogHook', 'TensorboardLogHook'],
        'lr': ['LrHook'],
        'registry': ['HOOKS'],
        'safetensors': ['SafetensorsHook'],
        'sampler': ['DistSamplerHook'],
        'val_loss': ['ValLossHook']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
