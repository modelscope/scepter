# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.network.autoencoder.ae_kl import AutoencoderKL
    from scepter.modules.model.network.autoencoder.ae_kl_cogvideox import AutoencoderKLCogVideoX
else:
    _import_structure = {
        'ae_kl': ['AutoencoderKL'],
        'ae_kl_cogvideox': ['AutoencoderKLCogVideoX']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
