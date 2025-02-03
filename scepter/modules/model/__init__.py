# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model import (backbone, embedder, head, loss, metric,
                                       neck, network, tokenizer, tuner, diffusion)
else:
    _import_structure = {
        'model': ['backbone', 'embedder', 'head', 'loss', 'metric',
                  'neck', 'network', 'tokenizer', 'tuner', 'diffusion']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
