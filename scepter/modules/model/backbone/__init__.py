# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.backbone import (ace, autoencoder, flux, image, cogvideox,
                                                mmdit, pixart, unet, utils, video)
else:
    _import_structure = {
        'backbone': ['ace', 'autoencoder', 'flux', 'image', 'cogvideox',
                     'mmdit', 'pixart', 'unet', 'utils', 'video']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
