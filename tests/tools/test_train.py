# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest


class TrainTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = './cache/save_data'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.data_dir = './cache/datasets'
        if not os.path.exists(self.data_dir):
            data_cmd = """mkdir -p cache/datasets/ && wget 'https://modelscope.cn/api/v1/models
                          /damo/scepter_scedit/repo?Revision=master&FilePath=dataset/3D_example_txt.zip'
                          -O cache/datasets/3D_example_txt.zip &&
                          unzip cache/datasets/3D_example_txt.zip
                          -d cache/datasets/ && rm cache/datasets/3D_example_txt.zip"""
            os.system(data_cmd)

    def tearDown(self):
        super().tearDown()

    @unittest.skip('')
    def test_generation_example_full(self):
        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd15_512_full/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_2.1_512.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd21_512_full/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd21_768_full/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sdxl_1024_full/checkpoints')))

    @unittest.skip('')
    def test_generation_example_lora(self):
        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512_lora.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd15_512_lora/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_2.1_512_lora.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd21_512_lora/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768_lora.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd21_768_lora/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024_lora.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sdxl_1024_lora/checkpoints')))

    @unittest.skip('')
    def test_generation_example_scedit_t2i_swift(self):
        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/t2i/sd15_512_sce_t2i_swift.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sd15_512_sce_t2i_swift/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/t2i/sd21_768_sce_t2i_swift.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sd21_768_sce_t2i_swift/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/t2i/sdxl_1024_sce_t2i_swift.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sdxl_1024_sce_t2i_swift/checkpoints')))

    @unittest.skip('')
    def test_generation_example_scedit_t2i(self):
        os.system('python scepter/tools/run_train.py '
                  '--cfg scepter/methods/scedit/t2i/sd15_512_sce_t2i.yaml '
                  '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd15_512_sce_t2i/checkpoints')))

        os.system('python scepter/tools/run_train.py '
                  '--cfg scepter/methods/scedit/t2i/sd21_768_sce_t2i.yaml '
                  '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sd21_768_sce_t2i/checkpoints')))

        os.system('python scepter/tools/run_train.py '
                  '--cfg scepter/methods/scedit/t2i/sdxl_1024_sce_t2i.yaml '
                  '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir, 'sdxl_1024_sce_t2i/checkpoints')))

    @unittest.skip('')
    def test_generation_example_scedit_ctr(self):
        os.system('python scepter/tools/run_train.py '
                  '--cfg scepter/methods/scedit/ctr/sd15_512_sce_ctr_hed.yaml '
                  '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sd15_512_sce_ctr_hed/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_canny.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sd21_768_sce_ctr_canny/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_pose.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sd21_768_sce_ctr_pose/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/ctr/sdxl_1024_sce_ctr_depth.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sdxl_1024_sce_ctr_depth/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/ctr/sdxl_1024_sce_ctr_color.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sdxl_1024_sce_ctr_color/checkpoints')))

    @unittest.skip('')
    def test_generation_example_datatxt(self):
        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/t2i/sdxl_1024_sce_t2i_datatxt.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sdxl_1024_sce_t2i_datatxt/checkpoints')))

        os.system(
            'python scepter/tools/run_train.py '
            '--cfg scepter/methods/scedit/ctr/sdxl_1024_sce_ctr_color_datatxt.yaml '
            '--max_steps 100')
        self.assertTrue(
            os.path.exists(
                os.path.join(self.tmp_dir,
                             'sdxl_1024_sce_ctr_color_datatxt/checkpoints')))


if __name__ == '__main__':
    unittest.main()
