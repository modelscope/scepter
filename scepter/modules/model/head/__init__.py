# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.head.classifier_head import (
        ClassifierHead, CosineLinearHead, TransformerHead, TransformerHeadx2,
        VideoClassifierHead, VideoClassifierHeadx2)
else:
    _import_structure = {
        'classifier_head': ['ClassifierHead', 'CosineLinearHead',
                            'TransformerHead', 'TransformerHeadx2',
                            'VideoClassifierHead', 'VideoClassifierHeadx2']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
