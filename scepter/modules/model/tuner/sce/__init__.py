# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.tuner.sce.scetuning import CSCTuners, SCTuner
    from scepter.modules.model.tuner.sce.scetuning_component import SCEAdapter
else:
    _import_structure = {
        'scetuning': ['CSCTuners', 'SCTuner'],
        'scetuning_component': ['SCEAdapter']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
