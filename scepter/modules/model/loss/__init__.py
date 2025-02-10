# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.loss.base_losses import CrossEntropy
    from scepter.modules.model.loss.rec_loss import MinSNRLoss, ReconstructLoss
else:
    _import_structure = {
        'base_losses': ['CrossEntropy'],
        'rec_loss': ['MinSNRLoss', 'ReconstructLoss']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
