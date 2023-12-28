# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest


class TrainTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    # @unittest.skip('')
    def test_generation_example_full(self):
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml ")
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768.yaml ")
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024.yaml ")
    
    # @unittest.skip('')
    def test_generation_example_lora(self):
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512_lora.yaml ")
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768_lora.yaml ")
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024_lora.yaml ")
        
    # @unittest.skip('')
    def test_generation_example_scedit(self):
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/SCEdit/t2i_sd15_512_sce.yaml ")
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/SCEdit/t2i_sd21_768_sce.yaml ")
        os.system("python scepter/tools/run_train.py "
          "--cfg scepter/methods/SCEdit/t2i_sdxl_1024_sce.yaml ")
        


if __name__ == '__main__':
    unittest.main()
