# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.utils.file_clients.aliyun_oss_fs import AliyunOssFs
    from scepter.modules.utils.file_clients.http_fs import HttpFs
    from scepter.modules.utils.file_clients.huggingface_fs import HuggingfaceFs
    from scepter.modules.utils.file_clients.local_fs import LocalFs
    from scepter.modules.utils.file_clients.modelscope_fs import ModelscopeFs
else:
    _import_structure = {
        'aliyun_oss_fs': ['AliyunOssFs'],
        'http_fs': ['HttpFs'],
        'huggingface_fs': ['HuggingfaceFs'],
        'local_fs': ['LocalFs'],
        'modelscope_fs': ['ModelscopeFs']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
