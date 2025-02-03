# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from .diffusions import BaseDiffusion, DiffusionFluxRF
    from .samplers import BaseDiffusionSampler, DDIMSampler, FlowEluerSampler
    from .schedules import (BaseNoiseScheduler, FlowMatchShiftScheduler,
                            ScaledLinearScheduler)
else:
    _import_structure = {
        'diffusions': ['BaseDiffusion', 'DiffusionFluxRF'],
        'samplers': ['BaseDiffusionSampler', 'DDIMSampler', 'FlowEluerSampler'],
        'schedules': ['BaseNoiseScheduler', 'FlowMatchShiftScheduler',
                      'ScaledLinearScheduler']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
