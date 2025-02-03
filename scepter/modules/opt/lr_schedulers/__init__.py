# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.opt.lr_schedulers.define_schedulers import LinoPolyLR
    from scepter.modules.opt.lr_schedulers.official_schedulers import *  # noqa
    from scepter.modules.opt.lr_schedulers.warmup import (StepAnnealingLR,
                                                          WarmupToConstantLR)
else:
    _import_structure = {
        'define_schedulers': ['LinoPolyLR'],
        'official_schedulers': ['StepLR', 'CyclicLR', 'LambdaLR', 'MultiStepLR',
                                'ExponentialLR', 'CosineAnnealingLR',
                                'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau'],
        'warmup': ['StepAnnealingLR', 'WarmupToConstantLR'],
        'registry': ['LR_SCHEDULERS']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
