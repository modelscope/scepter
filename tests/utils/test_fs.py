# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
import unittest

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS


class FSTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    @unittest.skip('')
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

        path = 'ms://AI-ModelScope/clip-vit-large-patch14'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

    @unittest.skip('')
    def test_modelscope_token(self):
        fs_info = {'NAME': 'ModelscopeFs', 'TEMP_DIR': 'cache/data'}
        config = Config(load=False, cfg_dict=fs_info)
        FS.init_fs_client(config)

        # path = 'ms://group_name/model_id:revision@file_path'
        path = 'ms://group_name/model_id:revision@file#token'
        with FS.get_from(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

        path = 'ms://group_name/model_id@file_dir#token'
        # path = 'ms://group_name/model_id#token'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

    @unittest.skip('')
    def test_huggingface(self):
        fs_info = {'NAME': 'HuggingfaceFs', 'TEMP_DIR': 'cache/data'}
        config = Config(load=False, cfg_dict=fs_info)
        FS.init_fs_client(config)

        path = 'hf://runwayml/stable-diffusion-v1-5'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

        path = 'hf://stabilityai/stable-diffusion-2-1-base@text_encoder'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

        path = 'hf://stabilityai/stable-diffusion-xl-base-1.0@README.md'
        with FS.get_from(path, wait_finish=True) as local_path:
            print(f'Download from {path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))

    @unittest.skip('')
    def test_scedit(self):
        fs_info = {'NAME': 'ModelscopeFs', 'TEMP_DIR': 'cache/cache_data'}
        config = Config(load=False, cfg_dict=fs_info)
        FS.init_fs_client(config)

        st_time = time.time()
        path = 'ms://iic/scepter_scedit'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(
                f'Download from {path} to {local_path}, take {time.time()-st_time}'
            )
            self.assertTrue(os.path.exists(local_path))

        st_time = time.time()
        path = 'ms://iic/scepter_scedit'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(
                f'Download from {path} to {local_path}, take {time.time()-st_time}'
            )
            self.assertTrue(os.path.exists(local_path))

        st_time = time.time()
        path = 'ms://iic/scepter_scedit@controllable_model/SD2.1/canny_control/'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(
                f'Download from {path} to {local_path}, take {time.time()-st_time}'
            )
            self.assertTrue(os.path.exists(local_path))

        st_time = time.time()
        path = 'ms://iic/scepter_scedit@controllable_model/SD2.1/canny_control/'
        with FS.get_dir_to_local_dir(path, wait_finish=True) as local_path:
            print(
                f'Download from {path} to {local_path}, take {time.time()-st_time}'
            )
            self.assertTrue(os.path.exists(local_path))

        st_time = time.time()
        path = 'ms://iic/scepter@mantra_images_jpg/SD_XL1.0/894f40ed44b37c3372e6a22b8ae577a4.jpg'
        with FS.get_from(path, wait_finish=True) as local_path:
            print(
                f'Download from {path} to {local_path}, take {time.time()-st_time}'
            )
            self.assertTrue(os.path.exists(local_path))

        st_time = time.time()
        path = 'ms://iic/scepter@mantra_images_jpg/SD2.1/894f40ed44b37c3372e6a22b8ae577a4.jpg'
        with FS.get_from(path, wait_finish=True) as local_path:
            print(
                f'Download from {path} to {local_path}, take {time.time()-st_time}'
            )
            self.assertTrue(os.path.exists(local_path))

        st_time = time.time()
        path = 'ms://iic/scepter@mantra_images_jpg/SD2.1/894f40ed44b37c3372e6a22b8ae577a4.jpg'
        with FS.get_from(path, wait_finish=True) as local_path:
            print(
                f'Download from {path} to {local_path}, take {time.time()-st_time}'
            )
            self.assertTrue(os.path.exists(local_path))


if __name__ == '__main__':
    unittest.main()
