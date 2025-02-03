# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.network.ldm.ldm import LatentDiffusion
    from scepter.modules.model.network.ldm.ldm_ace import (LatentDiffusionACE,
                                                           LatentDiffusionACERefiner)
    from scepter.modules.model.network.ldm.ldm_edit import LatentDiffusionEdit
    from scepter.modules.model.network.ldm.ldm_pixart import LatentDiffusionPixart
    from scepter.modules.model.network.ldm.ldm_sce import (
        LatentDiffusionSCEControl, LatentDiffusionSCETuning,
        LatentDiffusionXLSCEControl, LatentDiffusionXLSCETuning)
    from scepter.modules.model.network.ldm.ldm_sd3 import LatentDiffusionSD3
    from scepter.modules.model.network.ldm.ldm_xl import LatentDiffusionXL
    from scepter.modules.model.network.ldm.ldm_cogvideox import LatentDiffusionCogVideoX
    from scepter.modules.model.network.ldm.ldm_flux import (LatentDiffusionFlux,
                                                            LatentDiffusionFluxMR)
    from scepter.modules.model.network.ldm.ldm_ace_plus import LatentDiffusionACEPlus
else:
    _import_structure = {
        'ldm': ['LatentDiffusion'],
        'ldm_ace': ['LatentDiffusionACE', 'LatentDiffusionACERefiner'],
        'ldm_edit': ['LatentDiffusionEdit'],
        'ldm_pixart': ['LatentDiffusionPixart'],
        'ldm_sce': ['LatentDiffusionSCEControl', 'LatentDiffusionSCETuning',
                    'LatentDiffusionXLSCEControl', 'LatentDiffusionXLSCETuning'],
        'ldm_sd3': ['LatentDiffusionSD3'],
        'ldm_xl': ['LatentDiffusionXL'],
        'ldm_cogvideox': ['LatentDiffusionCogVideoX'],
        'ldm_flux': ['LatentDiffusionFlux', 'LatentDiffusionFluxMR'],
        'ldm_ace_plus': ['LatentDiffusionACEPlus'],
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
