# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS


class FSTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def test_modelscope(self):
        fs_info = {'NAME': 'ModelscopeFs', 'TEMP_DIR': 'cache/data'}
        config = Config(load=False, cfg_dict=fs_info)
        FS.init_fs_client(config)

        path = 'ms://AI-ModelScope/stable-diffusion-v1-5'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

        path = 'ms://AI-ModelScope/stable-diffusion-v1-5:v1.0.8'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

        path = 'ms://AI-ModelScope/stable-diffusion-v1-5@configuration.json'
        with FS.get_from(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

        path = 'ms://AI-ModelScope/stable-diffusion-v1-5:v1.0.8@v1-5-pruned-emaonly.ckpt'
        with FS.get_from(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

        path = 'ms://AI-ModelScope/stable-diffusion-v1-5:v1.0.8@text_encoder'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))


if __name__ == '__main__':
    unittest.main()
