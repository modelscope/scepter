# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.metric.classification import (AccuracyMetric,
                                                             EnsembleAccuracyMetric
                                                             )
else:
    _import_structure = {
        'classification': ['AccuracyMetric', 'EnsembleAccuracyMetric']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
