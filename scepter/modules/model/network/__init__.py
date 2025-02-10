# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.network.autoencoder import ae_kl
    from scepter.modules.model.network.classifier import Classifier
    from scepter.modules.model.network.diffusion import (diffusion, schedules,
                                                         solvers)
    from scepter.modules.model.network.ldm import (ldm, ldm_edit, ldm_pixart,
                                                   ldm_sce, ldm_sd3, ldm_xl,
                                                   ldm_flux)
else:
    _import_structure = {
        'autoencoder': ['ae_kl'],
        'classifier': ['Classifier'],
        'diffusion': ['diffusion', 'schedules', 'solvers'],
        'ldm': ['ldm', 'ldm_edit', 'ldm_pixart',
                'ldm_sce', 'ldm_sd3', 'ldm_xl',
                'ldm_flux']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
