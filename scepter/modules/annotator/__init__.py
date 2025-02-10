# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from scepter.modules.annotator.base_annotator import GeneralAnnotator
    from scepter.modules.annotator.canny import CannyAnnotator
    from scepter.modules.annotator.color import ColorAnnotator
    from scepter.modules.annotator.degradation import DegradationAnnotator
    from scepter.modules.annotator.doodle import DoodleAnnotator
    from scepter.modules.annotator.gray import GrayAnnotator
    from scepter.modules.annotator.hed import HedAnnotator
    from scepter.modules.annotator.identity import IdentityAnnotator
    from scepter.modules.annotator.informative_drawing import (
        InfoDrawAnimeAnnotator, InfoDrawContourAnnotator,
        InfoDrawOpenSketchAnnotator)
    from scepter.modules.annotator.inpainting import InpaintingAnnotator
    from scepter.modules.annotator.invert import InvertAnnotator
    from scepter.modules.annotator.midas_op import MidasDetector
    from scepter.modules.annotator.mlsd_op import MLSDdetector
    from scepter.modules.annotator.openpose import OpenposeAnnotator
    from scepter.modules.annotator.outpainting import OutpaintingAnnotator, OutpaintingResize
    from scepter.modules.annotator.pidinet import PiDiAnnotator
    from scepter.modules.annotator.segmentation import ESAMAnnotator
    from scepter.modules.annotator.sketch import SketchAnnotator
    from scepter.modules.annotator.lama import LamaAnnotator
else:
    _import_structure = {
        'base_annotator': ['GeneralAnnotator'],
        'canny': ['CannyAnnotator'],
        'color': ['ColorAnnotator'],
        'degradation': ['DegradationAnnotator'],
        'doodle': ['DoodleAnnotator'],
        'gray': ['GrayAnnotator'],
        'hed': ['HedAnnotator'],
        'identity': ['IdentityAnnotator'],
        'informative_drawing': ['InfoDrawAnimeAnnotator',
                                'InfoDrawContourAnnotator',
                                'InfoDrawOpenSketchAnnotator'],
        'inpainting': ['InpaintingAnnotator'],
        'invert': ['InvertAnnotator'],
        'midas_op': ['MidasDetector'],
        'mlsd_op': ['MLSDdetector'],
        'openpose': ['OpenposeAnnotator'],
        'outpainting': ['OutpaintingAnnotator', 'OutpaintingResize'],
        'pidinet': ['PiDiAnnotator'],
        'segmentation': ['ESAMAnnotator'],
        'sketch': ['SketchAnnotator'],
        'lama': ['LamaAnnotator'],
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
