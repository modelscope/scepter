# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules.model.tokenizer.base_tokenizer import BaseTokenizer
    from scepter.modules.model.tokenizer.tokenizer import (ClipTokenizer,
                                                           HuggingfaceTokenizer,
                                                           OpenClipTokenizer)
else:
    _import_structure = {
        'base_tokenizer': ['BaseTokenizer'],
        'tokenizer': ['ClipTokenizer', 'HuggingfaceTokenizer', 'OpenClipTokenizer']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
