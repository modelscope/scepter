# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
from itertools import chain
from unittest.mock import patch

from scepter.modules.utils.import_utils import LazyImportModule

# Mock data for testing
mock_import_structure = {'utils': ['config'], 'aliyun_oss': ['AliyunOssFs']}
mock_module_file = globals()['__file__']
mock_module_spec = __spec__
mock_extra_objects = {}

mock_modules = list(chain(*mock_import_structure.values()))


class TestLazyImportModule(unittest.TestCase):
    def test_init(self):
        lazy_module = LazyImportModule(__name__,
                                       mock_module_file,
                                       mock_import_structure,
                                       module_spec=mock_module_spec,
                                       extra_objects=mock_extra_objects,
                                       try_to_pre_import=False)
        self.assertEqual(lazy_module._name, __name__)
        self.assertEqual(lazy_module.__file__, mock_module_file)
        self.assertEqual(lazy_module.__spec__, mock_module_spec)
        self.assertListEqual(lazy_module.__all__,
                             list(mock_import_structure.keys()) + mock_modules)

    @patch('import_utils.importlib.import_module')
    def test_getattr_submodule(self, mock_import_module):
        lazy_module = LazyImportModule(__name__, mock_module_file,
                                       mock_import_structure)
        mock_import_module.return_value = 'mock_submodule'
        result = lazy_module.__getattr__('utils')
        self.assertEqual(result, 'mock_submodule')

    @patch('import_utils.importlib.import_module')
    def test_getattr_class(self, mock_import_module):
        lazy_module = LazyImportModule(__name__, mock_module_file,
                                       mock_import_structure)
        mock_import_module.return_value = 'mock_class'
        result = lazy_module.__getattr__('aliyun_oss')
        self.assertEqual(result, 'mock_class')

    def test_get_requirements(self):
        requirements = LazyImportModule.get_requirements()
        print('requirements is:', requirements)


if __name__ == '__main__':
    unittest.main()
