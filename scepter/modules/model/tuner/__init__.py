# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.tuner import sce
    from scepter.modules.model.tuner.swift_tuner import (SwiftPart, SwiftAdapter, SwiftFull,
                                                         SwiftLoRA, SwiftSCETuning)
else:
    _import_structure = {
        'tuner': ['sce'],
        'swift_tuner': ['SwiftPart', 'SwiftAdapter', 'SwiftFull',
                        'SwiftLoRA', 'SwiftSCETuning']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
