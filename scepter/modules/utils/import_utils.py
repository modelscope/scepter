# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import importlib
import logging
import os
import sys
from collections import OrderedDict
from importlib import import_module
from itertools import chain
from types import ModuleType
from typing import Any
import scepter
from scepter.modules.utils.ast_utils import (INDEX_KEY,
                                             MODULE_KEY,
                                             REQUIREMENT_KEY,
                                             load_index)
from scepter.modules.utils.error import *
from scepter.modules.utils.logger import get_logger


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

logger = get_logger()


def get_dirname():
    return os.path.dirname(scepter.__file__)


def import_modules(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                logger.warning(f'{imp} failed to import and is ignored.')
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


# following code borrows implementation from huggingface/transformers
ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({'AUTO'})
USE_TORCH = os.environ.get('USE_TORCH', 'AUTO').upper()

_torch_version = 'N/A'
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _torch_available = importlib.util.find_spec('torch') is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version('torch')
            logger.info(f'PyTorch version {_torch_version} Found.')
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info('Disabling PyTorch because USE_TF is set')
    _torch_available = False


def is_torchvision_available():
    return importlib.util.find_spec('torchvision') is not None


def is_sentencepiece_available():
    return importlib.util.find_spec('sentencepiece') is not None


def is_scepter_available():
    return importlib.util.find_spec('scepter') is not None


def is_torch_available():
    return _torch_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch
        return torch.cuda.is_available()
    else:
        return False


def is_swift_available():
    return importlib.util.find_spec('swift') is not None


def is_opencv_available():
    return importlib.util.find_spec('cv2') is not None


def is_pillow_available():
    return importlib.util.find_spec('PIL.Image') is not None


def _is_package_available_fn(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


def is_package_available(pkg_name):
    return functools.partial(_is_package_available_fn, pkg_name)


def is_flash_attn_available():
    return importlib.util.find_spec('flash-attn') is not None


def is_transformers_available():
    return importlib.util.find_spec('transformers') is not None


REQUIREMENTS_MAAPING = OrderedDict([
    ('scepter', (is_scepter_available(), SCEPTER_IMPORT_ERROR)),
    ('torch', (is_torch_available, PYTORCH_IMPORT_ERROR)),
    ('torchvision', (is_torchvision_available(), TORCHVISION_IMPORT_ERROR)),
    ('cv2', (is_opencv_available, OPENCV_IMPORT_ERROR)),
    ('PIL', (is_pillow_available, PILLOW_IMPORT_ERROR)),
    ('modelscope', (is_package_available('modelscope'), MODELSCOPE_IMPORT_ERROR)),
    ('flash-attn', (is_flash_attn_available, FLASH_ATTN_IMPORT_ERROR)),
    ('xformers', (is_package_available('funasr'), XFORMERS_IMPORT_ERROR)),
    ('albumentations', (is_package_available('albumentations'), ALBUMENTATIONS_IMPORT_ERROR)),
    ('decord', (is_package_available('decord'), DECORD_IMPORT_ERROR)),
    ('beautifulsoup4', (is_package_available('beautifulsoup4'), BEAUTIFULSOUP4_IMPORT_ERROR)),
    ('bezier', (is_package_available('bezier'), BEZIER_IMPORT_ERROR)),
    ('einops', (is_package_available('einops'), EINOPS_IMPORT_ERROR)),
    ('numpy', (is_package_available('numpy'), NUMPY_IMPORT_ERROR)),
    ('oss2', (is_package_available('oss2'), OSS2_IMPORT_ERROR)),
    ('pycocotools', (is_package_available('pycocotools'), PYCOCOTOOLS_IMPORT_ERROR)),
    ('open_clip', (is_package_available('open_clip'), OPENCLIP_IMPORT_ERROR)),
    ('pyyaml', (is_package_available('pyyaml'), PYYAML_IMPORT_ERROR)),
    ('transformers', (is_package_available('transformers'), TRANSFORMERS_IMPORT_ERROR)),
    ('ms-swift', (is_package_available('ms-swift'), SWIFT_IMPORT_ERROR)),
    ('gradio', (is_package_available('gradio'), SWIFT_IMPORT_ERROR)),
    ('scikit-image', (is_package_available('scikit-image'), SCIKIT_IMAGE_IMPORT_ERROR)),
    ('scikit-learn', (is_package_available('scikit-learn'), SCIKIT_LEARN_IMPORT_ERROR)),
    ('sentencepiece', (is_package_available('sentencepiece'), SENTENCEPIECE_IMPORT_ERROR)),
    ('torchsde', (is_package_available('torchsde'), TORCHSDE_IMPORT_ERROR)),
    ('bitsandbytes', (is_package_available('bitsandbytes'), BITSANDBYTES_IMPORT_ERROR)),
    ('gradio_imageslider', (is_package_available('gradio_imageslider'), GRADIO_IMAGESLIDER_IMPORT_ERROR)),
    ('imagehash', (is_package_available('imagehash'), IMAGEHASH_IMPORT_ERROR)),
    ('psutil', (is_package_available('psutil'), PSUTIL_IMPORT_ERROR)),
    ('tiktoken', (is_package_available('tiktoken'), TIKTOKEN_IMPORT_ERROR))
])

SYSTEM_PACKAGE = set(['os', 'sys', 'typing'])


def requires(obj, requirements):
    if not isinstance(requirements, (list, tuple)):
        requirements = [requirements]
    if isinstance(obj, str):
        name = obj
    else:
        name = obj.__name__ if hasattr(obj,
                                       '__name__') else obj.__class__.__name__

    checks = []
    for req in requirements:
        if req == '' or req in SYSTEM_PACKAGE:
            continue
        if req in REQUIREMENTS_MAAPING:
            check = REQUIREMENTS_MAAPING[req]
        else:
            check_fn = is_package_available(req)
            err_msg = GENERAL_IMPORT_ERROR.replace('REQ', req)
            check = (check_fn, err_msg)
        checks.append(check)

    failed = [msg.format(name) for available, msg in checks if not available]
    if failed:
        raise ImportError(''.join(failed))


def torch_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_torch_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f'Method `{func.__name__}` requires PyTorch.')
    return wrapper


class LazyImportModule(ModuleType):
    _AST_INDEX = None

    def __init__(self,
                 name,
                 module_file,
                 import_structure,
                 module_spec=None,
                 extra_objects=None,
                 try_to_pre_import=False):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(
            chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure
        if try_to_pre_import:
            self._try_to_import()

    def _try_to_import(self):
        for sub_module in self._class_to_module.keys():
            try:
                getattr(self, sub_module)
            except Exception as e:
                logger.warning(
                    f'pre load module {sub_module} error, please check {e}')

    def __dir__(self):
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(
                f'module {self.__name__} has no attribute {name}')

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            module_name_full = self.__name__ + '.' + module_name
            if not any(
                    module_name_full.startswith(f'scepter.{prefix}')
                    for prefix in ['modules', 'studio', 'version', 'tools', 'workflow']):
                # check requirements before module import
                requirements = self.get_requirements()
                if module_name_full in requirements:
                    requires(module_name_full, requirements)
            return importlib.import_module('.' + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f'Failed to import {self.__name__}.{module_name} because of the following error '
                f'(look up to see its traceback):\n{e}') from e

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__,
                                self._import_structure)

    @staticmethod
    def get_ast_index():
        if LazyImportModule._AST_INDEX is None:
            LazyImportModule._AST_INDEX = load_index()
        return LazyImportModule._AST_INDEX

    @staticmethod
    def import_module(signature):
        """ import a lazy import module using signature

        Args:
            signature (tuple): a tuple of str, (registry_name, class_name)
        """
        ast_index = LazyImportModule.get_ast_index()
        if signature in ast_index[INDEX_KEY]:
            mod_index = ast_index[INDEX_KEY][signature]
            module_name = mod_index[MODULE_KEY]
            if module_name in ast_index[REQUIREMENT_KEY]:
                requirements = ast_index[REQUIREMENT_KEY][module_name]
                requires(module_name, requirements)
            importlib.import_module(module_name)
        else:
            logger.warning(f'{signature} not found in ast index file')

    @staticmethod
    def get_module_type(module):
        ast_index = LazyImportModule.get_ast_index()
        if module in ast_index[INDEX_KEY]:
            return True
        else:
            return False
