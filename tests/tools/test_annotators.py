# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

import numpy as np
from PIL import Image

from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


class AnnotatorTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        _ = FS.init_fs_client(Config(cfg_dict={
            'NAME': 'ModelscopeFs',
            'TEMP_DIR': './cache/data'
        },
                                     load=False),
                              overwrite=False)
        image_path = 'asset/images/sunflower.jpeg'
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        self.image = np.array(image)

        self.save_dir = './cache/save_data/images'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def tearDown(self):
        super().tearDown()

    @unittest.skip('')
    def test_annotator_canny(self):
        # canny
        canny_dict = {
            'NAME': 'CannyAnnotator',
            'LOW_THRESHOLD': 100,
            'HIGH_THRESHOLD': 200
        }
        canny_anno = Config(cfg_dict=canny_dict, load=False)
        canny_ins = ANNOTATORS.build(canny_anno).to(we.device_id)
        canny_image = canny_ins(self.image)
        print("canny's shape:", canny_image.shape)
        Image.fromarray(canny_image).save(
            os.path.join(self.save_dir, 'sunflower_canny.png'))

    @unittest.skip('')
    def test_annotator_canny_random(self):
        # canny
        canny_dict = {
            'NAME': 'CannyAnnotator',
            'LOW_THRESHOLD': 100,
            'HIGH_THRESHOLD': 200,
            'RANDOM_CFG': {
                'PROBA': 1.0,
                'MIN_LOW_THRESHOLD': 50,
                'MAX_LOW_THRESHOLD': 100,
                'MIN_HIGH_THRESHOLD': 200,
                'MAX_HIGH_THRESHOLD': 350
            }
        }
        canny_anno = Config(cfg_dict=canny_dict, load=False)
        canny_ins = ANNOTATORS.build(canny_anno).to(we.device_id)
        canny_image = canny_ins(self.image)
        print("canny's shape:", canny_image.shape)
        Image.fromarray(canny_image).save(
            os.path.join(self.save_dir, 'sunflower_canny_random.png'))

    @unittest.skip('')
    def test_annotator_hed(self):
        # hed
        hed_dict = {
            'NAME':
            'HedAnnotator',
            'PRETRAINED_MODEL':
            'ms://damo/scepter_scedit@annotator/ckpts/ControlNetHED.pth'
        }
        hed_anno = Config(cfg_dict=hed_dict, load=False)
        hed_ins = ANNOTATORS.build(hed_anno).to(we.device_id)
        hed_image = hed_ins(self.image)
        print("hed's shape:", hed_image.shape)
        Image.fromarray(hed_image).save(
            os.path.join(self.save_dir, 'sunflower_hed.png'))

    @unittest.skip('')
    def test_annotator_openpose(self):
        # openpose
        openpose_dict = {
            'NAME':
            'OpenposeAnnotator',
            'BODY_MODEL_PATH':
            'ms://damo/scepter_scedit@annotator/ckpts/body_pose_model.pth',
            'HAND_MODEL_PATH':
            'ms://damo/scepter_scedit@annotator/ckpts/hand_pose_model.pth'
        }
        openpose_anno = Config(cfg_dict=openpose_dict, load=False)
        openpose_ins = ANNOTATORS.build(openpose_anno).to(we.device_id)
        openpose_image = openpose_ins(self.image)
        print("openpose's shape:", openpose_image.shape)
        Image.fromarray(openpose_image).save(
            os.path.join(self.save_dir, 'sunflower_openpose.png'))

    @unittest.skip('')
    def test_annotator_midas(self):
        # midas
        midas_dict = {
            'NAME': 'MidasDetector',
            'PRETRAINED_MODEL':
            'ms://damo/scepter_scedit@annotator/ckpts/dpt_hybrid-midas-501f0c75.pt',
            'A': 6.2,
            'BG_TH': 0.1
        }
        midas_anno = Config(cfg_dict=midas_dict, load=False)
        midas_ins = ANNOTATORS.build(midas_anno).to(we.device_id)
        midas_image = midas_ins(self.image)
        print("midas's shape:", midas_image.shape)
        Image.fromarray(midas_image).save(
            os.path.join(self.save_dir, 'sunflower_midas.png'))

    @unittest.skip('')
    def test_annotator_mlsd(self):
        # mlsd
        mlsd_dict = {
            'NAME': 'MLSDdetector',
            'PRETRAINED_MODEL':
            'ms://damo/scepter_scedit@annotator/ckpts/mlsd_large_512_fp32.pth',
            'THR_V': 0.1,
            'THR_D': 0.1
        }
        mlsd_anno = Config(cfg_dict=mlsd_dict, load=False)
        mlsd_ins = ANNOTATORS.build(mlsd_anno).to(we.device_id)
        mlsd_image = mlsd_ins(self.image)
        print("mlsd's shape:", mlsd_image.shape)
        Image.fromarray(mlsd_image).save(
            os.path.join(self.save_dir, 'sunflower_mlsd.png'))

    @unittest.skip('')
    def test_annotator_color(self):
        # color
        color_dict = {'NAME': 'ColorAnnotator', 'RATIO': 64}
        color_anno = Config(cfg_dict=color_dict, load=False)
        color_ins = ANNOTATORS.build(color_anno).to(we.device_id)
        color_image = color_ins(self.image)
        print("color's shape:", color_image.shape)
        Image.fromarray(color_image).save(
            os.path.join(self.save_dir, 'sunflower_color.png'))

    @unittest.skip('')
    def test_annotator_color_random(self):
        # color
        color_dict = {
            'NAME': 'ColorAnnotator',
            'RATIO': 64,
            'RANDOM_CFG': {
                'PROBA': 1.0,
                # 'MIN_RATIO': 64,
                # 'MAX_RATIO': 128
                'CHOICE_RATIO': [32, 64, 128]
            }
        }
        color_anno = Config(cfg_dict=color_dict, load=False)
        color_ins = ANNOTATORS.build(color_anno).to(we.device_id)
        color_image = color_ins(self.image)
        print("color's shape:", color_image.shape)
        Image.fromarray(color_image).save(
            os.path.join(self.save_dir, 'sunflower_color_random.png'))

    @unittest.skip('')
    def test_annotator_multi(self):
        # multi annotators
        canny_dict = {
            'NAME': 'CannyAnnotator',
            'LOW_THRESHOLD': 100,
            'HIGH_THRESHOLD': 200,
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['canny_img']
        }
        hed_dict = {
            'NAME': 'HedAnnotator',
            'PRETRAINED_MODEL':
            'ms://damo/scepter_scedit@annotator/ckpts/ControlNetHED.pth',
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['hed_img']
        }
        openpose_dict = {
            'NAME': 'OpenposeAnnotator',
            'BODY_MODEL_PATH':
            'ms://damo/scepter_scedit@annotator/ckpts/body_pose_model.pth',
            'HAND_MODEL_PATH':
            'ms://damo/scepter_scedit@annotator/ckpts/hand_pose_model.pth',
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['openpose_img']
        }
        midas_dict = {
            'NAME': 'MidasDetector',
            'PRETRAINED_MODEL':
            'ms://damo/scepter_scedit@annotator/ckpts/dpt_hybrid-midas-501f0c75.pt',
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['midas_img']
        }
        mlsd_dict = {
            'NAME': 'MLSDdetector',
            'PRETRAINED_MODEL':
            'ms://damo/scepter_scedit@annotator/ckpts/mlsd_large_512_fp32.pth',
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['mlsd_img']
        }
        color_dict = {'NAME': 'ColorAnnotator', 'RATIO': 64}
        general_dict = {
            'NAME':
            'GeneralAnnotator',
            'ANNOTATORS': [
                canny_dict, hed_dict, openpose_dict, midas_dict, mlsd_dict,
                color_dict
            ]
        }
        general_anno = Config(cfg_dict=general_dict, load=False)
        general_ins = ANNOTATORS.build(general_anno).to(we.device_id)
        output_image = general_ins({'img': self.image})
        for key, save_image in output_image.items():
            Image.fromarray(save_image).save(
                os.path.join(self.save_dir, f'sunflower_multi_{key}.png'))

    @unittest.skip('')
    def test_annotator_processor(self):
        from scepter.modules.annotator.utils import AnnotatorProcessor
        anno_processor = AnnotatorProcessor(anno_type='hed')
        output_image = anno_processor.run(self.image, 'hed')
        Image.fromarray(output_image).save(
            os.path.join(self.save_dir, 'sunflower_processor_hed.png'))

        anno_processor = AnnotatorProcessor(
            anno_type=['canny', 'color', 'depth'])
        output_image = anno_processor.run(self.image, 'color')
        Image.fromarray(output_image).save(
            os.path.join(self.save_dir, 'sunflower_processor_color.png'))

        output_image = anno_processor.run(self.image, ['canny', 'depth'])
        for key, save_image in output_image.items():
            Image.fromarray(save_image).save(
                os.path.join(self.save_dir, f'sunflower_processor_{key}.png'))


if __name__ == '__main__':
    unittest.main()
