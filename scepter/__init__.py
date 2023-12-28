# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import scepter
from scepter.modules import data, model, opt, solver, transform, utils
from scepter.tools.helper import get_module_list as module_list
from scepter.tools.helper import \
    get_module_object_config as configures_by_objects
from scepter.tools.helper import get_module_objects as objects_by_module
from scepter.version import __version__, version_info

dirname = os.path.dirname(scepter.__file__)

__all__ = [
    utils, transform, data, model, solver, version_info, opt, '__version__',
    'dirname'
]
