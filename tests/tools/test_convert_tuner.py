# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS


class ConvertTunerTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.convert_path = './cache/save_data/module_transform'
        self.convert_source_path = './cache/save_data/module_transform/source'
        self.convert_target_path = './cache/save_data/module_transform/target'
        if not os.path.exists(self.convert_target_path):
            os.makedirs(self.convert_target_path)
        fs_info = {
            'NAME': 'ModelscopeFs',
            'TEMP_DIR': self.convert_source_path
        }
        config = Config(load=False, cfg_dict=fs_info)
        FS.init_fs_client(config)

    def tearDown(self):
        super().tearDown()

    # @unittest.skip('')
    def test_scepter_to_civitai(self):
        save_path = os.path.join(self.convert_target_path, 'civitai_tuner',
                                 'tuner.safetensors')
        scepter_tuner_path = 'ms://sd_lora/SD15-LoRA-3DStyle-Scepter-Test'
        with FS.get_dir_to_local_dir(scepter_tuner_path,
                                     wait_finish=True) as local_path:
            print(f'Download from {scepter_tuner_path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))
        os.system('python scepter/tools/convert_tuner.py '
                  f'--source {local_path} '
                  f'--target {save_path} '
                  f'--export true')

    # @unittest.skip('')
    def test_civitai_to_scepter(self):
        save_path = os.path.join(self.convert_target_path, 'scepter_tuner')
        civitai_tuner_path = 'ms://sd_lora/SD15-LoRA-AnimeLineartStyle@animeoutlineV4_16.safetensors'
        with FS.get_dir_to_local_dir(civitai_tuner_path,
                                     wait_finish=True) as local_path:
            print(f'Download from {civitai_tuner_path} to {local_path}')
            self.assertTrue(os.path.exists(local_path))
        os.system('python scepter/tools/convert_tuner.py '
                  f'--source {local_path} '
                  f'--target {save_path} '
                  f'--export false')


if __name__ == '__main__':
    unittest.main()
