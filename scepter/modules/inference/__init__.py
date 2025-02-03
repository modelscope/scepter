# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.inference.diffusion_inference import DiffusionInference
    from scepter.modules.inference.ace_inference import ACEInference
    from scepter.modules.inference.cogvideox_inference import CogVideoXInference
    from scepter.modules.inference.control_inference import ControlInference
    from scepter.modules.inference.flux_inference import FluxInference
    from scepter.modules.inference.largen_inference import LargenInference
    from scepter.modules.inference.pixart_inference import PixArtInference
    from scepter.modules.inference.sd3_inference import SD3Inference
    from scepter.modules.inference.stylebooth_inference import StyleboothInference
    from scepter.modules.inference.tuner_inference import TunerInference
else:
    _import_structure = {
        'diffusion_inference': ['DiffusionInference'],
        'ace_inference': ['ACEInference'],
        'cogvideox_inference': ['CogVideoXInference'],
        'control_inference': ['ControlInference'],
        'flux_inference': ['FluxInference'],
        'largen_inference': ['LargenInference'],
        'pixart_inference': ['PixArtInference'],
        'sd3_inference': ['SD3Inference'],
        'stylebooth_inference': ['StyleboothInference'],
        'tuner_inference': ['TunerInference']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
