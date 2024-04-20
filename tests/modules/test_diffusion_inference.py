# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

import numpy as np
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.inference.diffusion_inference import DiffusionInference
from scepter.modules.inference.stylebooth_inference import StyleboothInference
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger


class DiffusionInferenceTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.logger = get_logger(name='scepter')
        config_file = 'scepter/methods/studio/scepter_ui.yaml'
        cfg = Config(cfg_file=config_file)
        if 'FILE_SYSTEM' in cfg:
            for fs_info in cfg['FILE_SYSTEM']:
                FS.init_fs_client(fs_info)
        self.tmp_dir = './cache/save_data/diffusion_inference'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()

    @unittest.skip('')
    def test_sd15(self):
        config_file = 'scepter/methods/studio/inference/stable_diffusion/sd15_pro.yaml'
        cfg = Config(cfg_file=config_file)
        diff_infer = DiffusionInference(logger=self.logger)
        diff_infer.init_from_cfg(cfg)
        output = diff_infer({'prompt': 'a cute dog'})
        save_path = os.path.join(self.tmp_dir,
                                 'sd15_test_prompt_a_cute_dog.png')
        save_image(output['images'], save_path)

    @unittest.skip('')
    def test_sd21(self):
        config_file = 'scepter/methods/studio/inference/stable_diffusion/sd21_pro.yaml'
        cfg = Config(cfg_file=config_file)
        diff_infer = DiffusionInference(logger=self.logger)
        diff_infer.init_from_cfg(cfg)
        output = diff_infer({'prompt': 'a cute dog'})
        save_path = os.path.join(self.tmp_dir,
                                 'sd21_test_prompt_a_cute_dog.png')
        save_image(output['images'], save_path)

    @unittest.skip('')
    def test_sdxl(self):
        config_file = 'scepter/methods/studio/inference/sdxl/sdxl1.0_pro.yaml'
        cfg = Config(cfg_file=config_file)
        diff_infer = DiffusionInference(logger=self.logger)
        diff_infer.init_from_cfg(cfg)
        output = diff_infer({'prompt': 'a cute dog'})
        save_path = os.path.join(self.tmp_dir,
                                 'sdxl_test_prompt_a_cute_dog.png')
        save_image(output['images'], save_path)

    # @unittest.skip('')
    def test_sd15_scedit_t2i_2D(self):
        # init model
        config_file = 'scepter/methods/studio/inference/stable_diffusion/sd15_pro.yaml'
        cfg = Config(cfg_file=config_file)
        diff_infer = DiffusionInference(logger=self.logger)
        diff_infer.init_from_cfg(cfg)
        # load tuner model
        tuner_model = {
            'NAME': 'Flat 2D Art',
            'NAME_ZH': None,
            'DESCRIPTION': None,
            'BASE_MODEL': 'SD1.5',
            'IMAGE_PATH': None,
            'TUNER_TYPE': 'SwiftSCE',
            'MODEL_PATH':
            'ms://damo/scepter_scedit@tuners_model/SD1.5/Flat2DArt',
            'PROMPT_EXAMPLE': None
        }
        tuner_model = Config(cfg_dict=tuner_model, load=False)
        # prepare data
        input_data = {'prompt': 'a single flower is shown in front of a tree'}
        input_params = {
            'tuner_model': tuner_model,
            'tuner_scale': 1.0,
            'seed': 2024
        }
        output = diff_infer(input_data, **input_params)
        save_path = os.path.join(self.tmp_dir, 'sd15_flower_2d.png')
        save_image(output['images'], save_path)

    # @unittest.skip('')
    def test_sdxl_scedit_ctr_canny(self):
        # init model
        config_file = 'scepter/methods/studio/inference/sdxl/sdxl1.0_pro.yaml'
        cfg = Config(cfg_file=config_file)
        diff_infer = DiffusionInference(logger=self.logger)
        diff_infer.init_from_cfg(cfg)
        # extract condition
        canny_dict = {
            'NAME': 'CannyAnnotator',
            'LOW_THRESHOLD': 100,
            'HIGH_THRESHOLD': 200
        }
        canny_anno = Config(cfg_dict=canny_dict, load=False)
        canny_ins = ANNOTATORS.build(canny_anno).to(we.device_id)
        output_height, output_width = 1024, 1024
        control_cond_image = Image.open('asset/images/flower.jpg')
        control_cond_image = TT.Resize(min(output_height,
                                           output_width))(control_cond_image)
        control_cond_image = TT.CenterCrop(
            (output_height, output_width))(control_cond_image)
        control_cond_image = canny_ins(np.array(control_cond_image))
        control_save_path = os.path.join(self.tmp_dir,
                                         'sdxl_flower_canny_preproccess.png')
        save_image(TF.to_tensor(control_cond_image), control_save_path)
        control_cond_image = Image.open(control_save_path)
        # load control model
        control_model = {
            'NAME':
            'canny',
            'NAME_ZH':
            None,
            'DESCRIPTION':
            None,
            'BASE_MODEL':
            'SD_XL1.0',
            'TYPE':
            'Canny',
            'MODEL_PATH':
            'ms://damo/scepter_scedit@controllable_model/SD_XL1.0/canny_control'
        }
        control_model = Config(cfg_dict=control_model, load=False)
        # prepare data
        input_data = {'prompt': 'a single flower is shown in front of a tree'}
        input_params = {
            'control_model': control_model,
            'control_cond_image': control_cond_image,
            'control_scale': 1.0,
            'crop_type': 'CenterCrop',
            'seed': 2024
        }
        output = diff_infer(input_data, **input_params)
        save_path = os.path.join(self.tmp_dir, 'sdxl_flower_canny.png')
        save_image(output['images'], save_path)

    # @unittest.skip('')
    def test_stylebooth(self):
        config_file = 'scepter/methods/studio/inference/edit/stylebooth_tb_pro.yaml'
        cfg = Config(cfg_file=config_file)
        diff_infer = StyleboothInference(logger=self.logger)
        diff_infer.init_from_cfg(cfg)

        output = diff_infer({'prompt': 'Let this image be in the style of sai-lowpoly'},
                            style_edit_image=Image.open('asset/images/inpainting_text_ref/ex4_scene_im.jpg'),
                            style_guide_scale_text=7.5,
                            style_guide_scale_image=0.5)
        save_path = os.path.join(self.tmp_dir,
                                 'stylebooth_test_lowpoly_cute_dog.png')
        save_image(output['images'], save_path)


if __name__ == '__main__':
    unittest.main()
