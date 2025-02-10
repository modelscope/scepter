# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.data.sampler.base_sampler import BaseSampler
    from scepter.modules.data.sampler.registry import SAMPLERS
    from scepter.modules.data.sampler.sampler import (
        EvalDistributedSampler, LoopSampler, MixtureOfSamplers,
        MultiFoldDistributedSampler, MultiLevelBatchSampler,
        MultiLevelBatchSamplerMultiSource, ResolutionBatchSampler)
else:
    _import_structure = {
        'base_sampler': ['BaseSampler'],
        'registry': ['SAMPLERS'],
        'sampler': ['EvalDistributedSampler', 'LoopSampler',
                    'MixtureOfSamplers', 'MultiFoldDistributedSampler',
                    'MultiLevelBatchSampler', 'MultiLevelBatchSamplerMultiSource',
                    'ResolutionBatchSampler']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
