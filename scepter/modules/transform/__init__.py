# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.transform.augmention import ColorJitterGeneral
    from scepter.modules.transform.compose import Compose
    from scepter.modules.transform.identity import Identity
    from scepter.modules.transform.image import (CenterCrop, FlexibleCenterCrop,
                                                 FlexibleResize, ImageToTensor,
                                                 ImageTransform, Normalize,
                                                 RandomHorizontalFlip,
                                                 RandomResizedCrop, Resize)
    from scepter.modules.transform.io import (LoadCvImageFromFile,
                                              LoadImageFromFile,
                                              LoadImageFromFileList,
                                              LoadPILImageFromFile)
    from scepter.modules.transform.io_video import (DecodeVideoToTensor,
                                                    LoadVideoFromFile)
    from scepter.modules.transform.registry import TRANSFORMS, build_pipeline
    from scepter.modules.transform.tensor import (Rename, RenameMeta, Select,
                                                  TemplateStr, ToNumpy, ToTensor)
    from scepter.modules.transform.transform_xl import FlexibleCropXL
    from scepter.modules.transform.video import (AutoResizedCropVideo,
                                                 CenterCropVideo, NormalizeVideo,
                                                 RandomHorizontalFlipVideo,
                                                 RandomResizedCropVideo,
                                                 ResizeVideo, VideoToTensor,
                                                 VideoTransform)
else:
    _import_structure = {
        'augmention': ['ColorJitterGeneral'],
        'compose': ['Compose'],
        'identity': ['Identity'],
        'image': ['CenterCrop', 'FlexibleCenterCrop', 'FlexibleResize',
                  'ImageToTensor', 'ImageTransform', 'Normalize',
                  'RandomHorizontalFlip', 'RandomResizedCrop', 'Resize'],
        'io': ['LoadCvImageFromFile', 'LoadImageFromFile',
               'LoadImageFromFileList', 'LoadPILImageFromFile'],
        'io_video': ['DecodeVideoToTensor', 'LoadVideoFromFile'],
        'registry': ['TRANSFORMS', 'build_pipeline'],
        'tensor': ['Rename', 'RenameMeta', 'Select', 'TemplateStr',
                   'ToNumpy', 'ToTensor'],
        'transform_xl': ['FlexibleCropXL'],
        'video': ['AutoResizedCropVideo', 'CenterCropVideo', 'NormalizeVideo',
                  'RandomHorizontalFlipVideo', 'RandomResizedCropVideo',
                  'ResizeVideo', 'VideoToTensor', 'VideoTransform']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
