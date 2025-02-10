# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from .frame_sampler import (FRAME_SAMPLERS, IntervalSampler, SegmentSampler,
                                UniformSampler, do_frame_sample)
    from .video_reader import (EasyVideoReader, FramesReaderWrapper,
                               VideoReaderWrapper)
else:
    _import_structure = {
        'frame_sampler': ['FRAME_SAMPLERS', 'IntervalSampler', 'SegmentSampler',
                          'UniformSampler', 'do_frame_sample'],
        'video_reader': ['EasyVideoReader', 'FramesReaderWrapper', 'VideoReaderWrapper']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
