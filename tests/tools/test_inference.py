# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest


class InferenceTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    # @unittest.skip('')
    def test_infer_args(self):
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_2023'")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_2024' --seed 2024")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_size768' "
          "--image_size '768'")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_1280_720' "
          "--image_size '1280,720'")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_720_1280_step10' "
          "--image_size '720,1280' --sample_steps 10")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_num2' --num_samples 2")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_dpmpp_2s_ancestral' "
          "--sampler 'dpmpp_2s_ancestral'")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_scale5' "
          "--guide_scale 5.0")
        os.system("python scepter/tools/run_inference.py "
          "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
          "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_rescale0_1' "
          "--guide_rescale 0.1")

    # @unittest.skip('')
    def test_example_infer(self):
          os.system("python scepter/tools/run_inference.py "
            "--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml "
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog'")
          os.system("python scepter/tools/run_inference.py "
            "--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768.yaml "
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog'")
          os.system("python scepter/tools/run_inference.py "
            "--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024.yaml "
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog'")


if __name__ == '__main__':
    unittest.main()
