# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.opt.optimizers.official_optimizers import (
        ASGD, LBFGS, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW, RMSprop, Rprop,
        SparseAdam)
    from scepter.modules.opt.optimizers.registry import OPTIMIZERS
else:
    _import_structure = {
        'official_optimizers': ['ASGD', 'LBFGS', 'SGD', 'Adadelta',
                                'Adagrad', 'Adam', 'Adamax', 'AdamW',
                                'RMSprop', 'Rprop', 'SparseAdam'],
        'registry': ['OPTIMIZERS']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
