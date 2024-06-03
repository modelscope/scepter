# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest


class InferenceTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    @unittest.skip('')
    def test_infer_args(self):
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_2023'"
        )
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_2024' --seed 2024"
        )
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_size768' "
            "--image_size '768'")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_1280_720' "
            "--image_size '1280,720'")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_720_1280_step10' "
            "--image_size '720,1280' --sample_steps 10")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_num2' --num_samples 2"
        )
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_dpmpp_2s_ancestral' "
            "--sampler 'dpmpp_2s_ancestral'")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_scale5' "
            '--guide_scale 5.0')
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog_rescale0_1' "
            '--guide_rescale 0.1')

    @unittest.skip('')
    def test_example_infer(self):
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog'")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog'")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024.yaml '
            "--prompt 'a cute dog' --save_folder 'test_prompt_a_cute_dog'")

    @unittest.skip('')
    def test_trained_infer(self):
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml '
            "--pretrained_model 'cache/save_data/sd15_512_full/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768.yaml '
            "--pretrained_model 'cache/save_data/sd21_768_full/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024.yaml --guide_scale 5.0 '
            "--pretrained_model 'cache/save_data/sdxl_1024_full/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")

    @unittest.skip('')
    def test_trained_tuning_infer(self):
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_1.5_512_lora.yaml '
            "--pretrained_model 'cache/save_data/sd15_512_lora/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_2.1_768_lora.yaml '
            "--pretrained_model 'cache/save_data/sd21_768_lora/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/examples/generation/stable_diffusion_xl_1024_lora.yaml '
            "--pretrained_model 'cache/save_data/sdxl_1024_lora/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")

    @unittest.skip('')
    def test_trained_scedit_infer(self):
        # swift
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/t2i/sd15_512_sce_t2i_swift.yaml '
            "--pretrained_model 'cache/save_data/sd15_512_sce_t2i_swift/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/t2i/sd21_768_sce_t2i_swift.yaml '
            "--pretrained_model 'cache/save_data/sd21_768_sce_t2i_swift/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/t2i/sdxl_1024_sce_t2i_swift.yaml '
            "--pretrained_model 'cache/save_data/sdxl_1024_sce_t2i_swift/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        # original
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/t2i/sd15_512_sce_t2i.yaml '
            "--pretrained_model 'cache/save_data/sd15_512_sce_t2i/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/t2i/sd21_768_sce_t2i.yaml '
            "--pretrained_model 'cache/save_data/sd21_768_sce_t2i/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/t2i/sdxl_1024_sce_t2i.yaml '
            "--pretrained_model 'cache/save_data/sdxl_1024_sce_t2i/checkpoints/ldm_step-100.pth' "
            "--prompt 'A close up of a small rabbit wearing a hat and scarf' "
            "--save_folder 'trained_test_prompt_rabbit' ")

    # @unittest.skip('')
    def test_pretrained_scedit_control_infer(self):
        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_canny.yaml --num_samples 1 '
            "--prompt 'a single flower is shown in front of a tree' --save_folder 'test_flower_canny' "
            "--image_size 768 --task control --image 'asset/images/flower.jpg' --control_mode canny "
            '--pretrained_model '
            'ms://iic/scepter_scedit@controllable_model/SD2.1/canny_control/0_SwiftSCETuning/pytorch_model.bin'
        )

        os.system(
            'python scepter/tools/run_inference.py '
            '--cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_pose.yaml --num_samples 1 '
            "--prompt 'super mario' --save_folder 'test_mario_pose' "
            "--image_size 768 --task control --image 'asset/images/pose_source.png' --control_mode source "
            '--pretrained_model '
            'ms://iic/scepter_scedit@controllable_model/SD2.1/pose_control/0_SwiftSCETuning/pytorch_model.bin'
        )


if __name__ == '__main__':
    unittest.main()
