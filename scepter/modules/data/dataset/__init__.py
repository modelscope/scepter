# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.data.dataset.base_dataset import BaseDataset
    from scepter.modules.data.dataset.dataset import (Image2ImageDataset,
                                                      ImageClassifyPublicDataset,
                                                      ImageTextPairDataset,
                                                      Text2ImageDataset)
    from scepter.modules.data.dataset.ms_dataset import (
        ImageTextPairFolderDataset, ImageTextPairMSDataset)
    from scepter.modules.data.dataset.registry import DATASETS
    from scepter.modules.data.dataset.video_gen_dataset import VideoGenDataset
else:
    _import_structure = {
        'base_dataset': ['BaseDataset'],
        'dataset': ['Image2ImageDataset', 'ImageClassifyPublicDataset',
                    'ImageTextPairDataset', 'Text2ImageDataset'],
        'ms_dataset': ['ImageTextPairFolderDataset',
                       'ImageTextPairMSDataset'],
        'registry': ['DATASETS'],
        'video_gen_dataset': ['VideoGenDataset']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
