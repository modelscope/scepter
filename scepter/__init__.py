# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule


if TYPE_CHECKING:
    from scepter.modules import data, model, opt, solver, transform, utils
    from scepter.tools.helper import get_module_list as module_list
    from scepter.tools.helper import \
        get_module_object_config as configures_by_objects
    from scepter.tools.helper import get_module_objects as objects_by_module
    from scepter.version import __version__, version_info
else:
    _import_structure = {
        'modules': ['data', 'model', 'opt', 'solver', 'transform', 'utils'],
        'helper': ['get_module_list', 'get_module_object_config', 'get_module_objects'],
        'version': ['__version__', 'version_info']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
