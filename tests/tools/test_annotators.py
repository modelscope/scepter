# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

import cv2
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
            'ms://iic/scepter_scedit@annotator/ckpts/ControlNetHED.pth'
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
            'ms://iic/scepter_scedit@annotator/ckpts/body_pose_model.pth',
            'HAND_MODEL_PATH':
            'ms://iic/scepter_scedit@annotator/ckpts/hand_pose_model.pth'
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
            'ms://iic/scepter_scedit@annotator/ckpts/dpt_hybrid-midas-501f0c75.pt',
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
            'ms://iic/scepter_scedit@annotator/ckpts/mlsd_large_512_fp32.pth',
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
            'ms://iic/scepter_scedit@annotator/ckpts/ControlNetHED.pth',
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['hed_img']
        }
        openpose_dict = {
            'NAME': 'OpenposeAnnotator',
            'BODY_MODEL_PATH':
            'ms://iic/scepter_scedit@annotator/ckpts/body_pose_model.pth',
            'HAND_MODEL_PATH':
            'ms://iic/scepter_scedit@annotator/ckpts/hand_pose_model.pth',
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['openpose_img']
        }
        midas_dict = {
            'NAME': 'MidasDetector',
            'PRETRAINED_MODEL':
            'ms://iic/scepter_scedit@annotator/ckpts/dpt_hybrid-midas-501f0c75.pt',
            'INPUT_KEYS': ['img'],
            'OUTPUT_KEYS': ['midas_img']
        }
        mlsd_dict = {
            'NAME': 'MLSDdetector',
            'PRETRAINED_MODEL':
            'ms://iic/scepter_scedit@annotator/ckpts/mlsd_large_512_fp32.pth',
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

    @unittest.skip('')
    def test_annotator_doodle(self):
        doodle_dict = {
            'NAME': 'DoodleAnnotator', 'PROCESSOR_TYPE': 'pidinet_sketch',
            'PROCESSOR_CFG': [
                {'NAME': 'PiDiAnnotator',
                 'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/table5_pidinet.pth'},
                {'NAME': 'SketchAnnotator',
                 'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/sketch_simplification_gan.pth'}
            ]
        }
        doodle_anno = Config(cfg_dict=doodle_dict, load=False)
        doodle_ins = ANNOTATORS.build(doodle_anno).to(we.device_id)
        doodle_image = doodle_ins(self.image)
        print("doodle's shape:", doodle_image.shape)
        Image.fromarray(doodle_image).save(
            os.path.join(self.save_dir, 'sunflower_doodle.png'))

    @unittest.skip('')
    def test_annotator_gray(self):
        gray_dict = {'NAME': 'GrayAnnotator'}
        gray_anno = Config(cfg_dict=gray_dict, load=False)
        gray_ins = ANNOTATORS.build(gray_anno).to(we.device_id)
        gray_image = gray_ins(self.image)
        print("gray's shape:", gray_image.shape)
        Image.fromarray(gray_image).save(
            os.path.join(self.save_dir, 'sunflower_gray.png'))

    @unittest.skip('')
    def test_annotator_drawing(self):
        cont_dict = {'NAME': 'InfoDrawContourAnnotator', 'INPUT_NC': 3, 'OUTPUT_NC': 1, 'N_RESIDUAL_BLOCKS': 3,
                     'SIGMOID': True,
                     'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/informative_drawing_contour_style.pth'}
        cont_anno = Config(cfg_dict=cont_dict, load=False)
        cont_ins = ANNOTATORS.build(cont_anno).to(we.device_id)
        cont_image = cont_ins(self.image)
        print("cont's shape:", cont_image.shape)
        Image.fromarray(cont_image).save(
            os.path.join(self.save_dir, 'sunflower_drawing_contour_style.png'))

        cont_dict = {'NAME': 'InfoDrawAnimeAnnotator', 'INPUT_NC': 3, 'OUTPUT_NC': 1, 'N_RESIDUAL_BLOCKS': 3,
                     'SIGMOID': True,
                     'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/informative_drawing_anime_style.pth'}
        cont_anno = Config(cfg_dict=cont_dict, load=False)
        cont_ins = ANNOTATORS.build(cont_anno).to(we.device_id)
        cont_image = cont_ins(self.image)
        print("cont's shape:", cont_image.shape)
        Image.fromarray(cont_image).save(
            os.path.join(self.save_dir, 'sunflower_drawing_anime_style.png'))

        cont_dict = {'NAME': 'InfoDrawOpenSketchAnnotator', 'INPUT_NC': 3, 'OUTPUT_NC': 1, 'N_RESIDUAL_BLOCKS': 3,
                     'SIGMOID': True,
                     'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/informative_drawing_opensketch_style.pth'}
        cont_anno = Config(cfg_dict=cont_dict, load=False)
        cont_ins = ANNOTATORS.build(cont_anno).to(we.device_id)
        cont_image = cont_ins(self.image)
        print("cont's shape:", cont_image.shape)
        Image.fromarray(cont_image).save(
            os.path.join(self.save_dir, 'sunflower_drawing_opensketch_style.png'))

    @unittest.skip('')
    def test_annotator_outpainting(self):
        outpaint_dict = {'NAME': 'OutpaintingAnnotator', 'RETURN_SOURCE': False}
        outpaint_anno = Config(cfg_dict=outpaint_dict, load=False)
        outpaint_ins = ANNOTATORS.build(outpaint_anno).to(we.device_id)
        outpaint_image = outpaint_ins(self.image)
        print("outpaint's shape:", outpaint_image.shape)
        Image.fromarray(outpaint_image).save(
            os.path.join(self.save_dir, 'sunflower_outpaint.png'))

        outpaint_dict = {'NAME': 'OutpaintingAnnotator',
                         'RANDOM_CFG': {'DIRECTION_RANGE': ['left', 'right', 'up'], 'RATIO_RANGE': [0.2, 0.8]}}
        outpaint_anno = Config(cfg_dict=outpaint_dict, load=False)
        outpaint_ins = ANNOTATORS.build(outpaint_anno).to(we.device_id)
        outpaint_image = outpaint_ins(self.image, return_mask=True)
        print("outpaint image's shape:", outpaint_image['image'].shape)
        print("outpaint mask's shape:", outpaint_image['mask'].shape)
        Image.fromarray(outpaint_image['image']).save(
            os.path.join(self.save_dir, 'sunflower_outpaint_rand_image.png'))
        Image.fromarray(outpaint_image['mask']).save(
            os.path.join(self.save_dir, 'sunflower_outpaint_rand_mask.png'))

    @unittest.skip('')
    def test_annotator_inpainting(self):
        inpaint_dict = {'NAME': 'InpaintingAnnotator'}
        inpaint_anno = Config(cfg_dict=inpaint_dict, load=False)
        inpaint_ins = ANNOTATORS.build(inpaint_anno).to(we.device_id)
        mask = np.zeros_like(self.image)
        mask = cv2.rectangle(mask, (0, 0), (150, 150), (255, 255, 255), -1)
        mask = mask[:, :, 0]  # one channel format
        inpaint_image = inpaint_ins(self.image, mask=mask, return_mask=True)
        print("inpaint image's shape:", inpaint_image['image'].shape)
        print("inpaint mask's shape:", inpaint_image['mask'].shape)
        Image.fromarray(inpaint_image['image']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_image.png'))
        Image.fromarray(inpaint_image['mask']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_mask.png'))

        inpaint_dict = {'NAME': 'InpaintingAnnotator'}
        inpaint_anno = Config(cfg_dict=inpaint_dict, load=False)
        inpaint_ins = ANNOTATORS.build(inpaint_anno).to(we.device_id)
        inpaint_image = inpaint_ins(self.image, return_mask=True)
        print("inpaint image's shape:", inpaint_image['image'].shape)
        print("inpaint mask's shape:", inpaint_image['mask'].shape)
        Image.fromarray(inpaint_image['image']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_image_2.png'))
        Image.fromarray(inpaint_image['mask']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_mask_2.png'))

        inpaint_dict = {'NAME': 'InpaintingAnnotator',
                        'MASK_CFG': {"irregular_proba": 0.5,
                                     "irregular_kwargs": {"min_times": 4,
                                                          "max_times": 10,
                                                          "max_width": 150,
                                                          "max_angle": 4,
                                                          "max_len": 200},
                                     "box_proba": 0.5,
                                     "box_kwargs": {"margin": 0,
                                                    "bbox_min_size": 50,
                                                    "bbox_max_size": 150,
                                                    "max_times": 5,
                                                    "min_times": 1}
                                     }
                        }
        inpaint_anno = Config(cfg_dict=inpaint_dict, load=False)
        inpaint_ins = ANNOTATORS.build(inpaint_anno).to(we.device_id)
        inpaint_image = inpaint_ins(self.image, return_mask=True)
        print("inpaint image's shape:", inpaint_image['image'].shape)
        print("inpaint mask's shape:", inpaint_image['mask'].shape)
        Image.fromarray(inpaint_image['image']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_image_3.png'))
        Image.fromarray(inpaint_image['mask']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_mask_3.png'))

        inpaint_image = inpaint_ins(self.image, return_mask=True, mask_color=255)
        print("inpaint image's shape:", inpaint_image['image'].shape)
        print("inpaint mask's shape:", inpaint_image['mask'].shape)
        Image.fromarray(inpaint_image['image']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_image_4.png'))
        Image.fromarray(inpaint_image['mask']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_mask_4.png'))

        inpaint_image = inpaint_ins(self.image, return_mask=True, mask_color=255, return_invert=False)
        print("inpaint image's shape:", inpaint_image['image'].shape)
        print("inpaint mask's shape:", inpaint_image['mask'].shape)
        Image.fromarray(inpaint_image['image']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_image_5.png'))
        Image.fromarray(inpaint_image['mask']).save(
            os.path.join(self.save_dir, 'sunflower_inpaint_mask_5.png'))

    @unittest.skip('')
    def test_annotator_deg(self):
        deg_dict = {'NAME': 'DegradationAnnotator'}
        deg_anno = Config(cfg_dict=deg_dict, load=False)
        deg_ins = ANNOTATORS.build(deg_anno).to(we.device_id)
        deg_image = deg_ins(self.image)
        print("deg's shape:", deg_image.shape)
        Image.fromarray(deg_image).save(
            os.path.join(self.save_dir, 'sunflower_deg.png'))

        deg_dict = {
            'NAME': 'DegradationAnnotator',
            'RANDOM_DEGRADATION': True,
            'PARAMS': {
                'gaussian_noise': {},
                'resize': {'scale': [0.4, 0.8]},
                'jpeg': {'jpeg_level': [25, 75]},
                'gaussian_blur': {'kernel_size': [7, 9, 11, 13, 15], 'sigma': [0.9, 1.8]}
            }
        }
        deg_anno = Config(cfg_dict=deg_dict, load=False)
        deg_ins = ANNOTATORS.build(deg_anno).to(we.device_id)
        deg_image = deg_ins(self.image)
        print("deg's shape:", deg_image.shape)
        Image.fromarray(deg_image).save(
            os.path.join(self.save_dir, 'sunflower_deg_2.png'))

    @unittest.skip('')
    def test_annotator_seg(self):
        seg_dict = {
            'NAME': 'ESAMAnnotator',
            'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/efficient_sam_vits.pt',
            'SAVE_MODE': 'P',
            'GRID_SIZE': 32,
        }
        seg_anno = Config(cfg_dict=seg_dict, load=False)
        seg_ins = ANNOTATORS.build(seg_anno).to(we.device_id)
        seg_image = seg_ins(self.image)
        print("seg's shape:", seg_image.shape)
        Image.fromarray(seg_image).save(
            os.path.join(self.save_dir, 'sunflower_esam_seg.png'))

        seg_dict = {
            'NAME': 'ESAMAnnotator',
            'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/efficient_sam_vits.pt',
            'SAVE_MODE': 'P',
            'GRID_SIZE': 32,
            'USE_DOMINANT_COLOR': True,
            'RETURN_MASK': True
        }
        seg_anno = Config(cfg_dict=seg_dict, load=False)
        seg_ins = ANNOTATORS.build(seg_anno).to(we.device_id)
        seg_image = seg_ins(self.image)
        print("seg image's shape:", seg_image['image'].shape)
        Image.fromarray(seg_image['image']).save(
            os.path.join(self.save_dir, 'sunflower_esam_seg_dominant_image.png'))
        print("seg mask's shape:", seg_image['mask'].shape)
        Image.fromarray(seg_image['mask']).save(
            os.path.join(self.save_dir, 'sunflower_esam_seg_dominant_mask.png'))

    @unittest.skip('')
    def test_annotator_samdraw(self):
        sam_dict = {
            'NAME': 'SAMAnnotatorDraw',
            'TASK_TYPE': 'input_box',
            'SAM_MODEL': 'vit_b',
            'PRETRAINED_MODEL': 'ms://iic/scepter_annotator@annotator/ckpts/sam_vit_b_01ec64.pth'
        }
        sam_anno = Config(cfg_dict=sam_dict, load=False)
        sam_ins = ANNOTATORS.build(sam_anno).to(we.device_id)
        sam_res = sam_ins(self.image, input_box=[0, 0, 200, 200], task_type='input_box', multimask_output=False)
        Image.fromarray(sam_res['mask']).save(os.path.join(self.save_dir, f'sunflower_sam_mask.png'))

    @unittest.skip('')
    def test_annotator_lama(self):
        lama_dict = {
            'NAME': 'LamaAnnotator',
            'PRETRAINED_MODEL': 'ms://iic/cv_fft_inpainting_lama/'
        }
        lama_anno = Config(cfg_dict=lama_dict, load=False)
        lama_ins = ANNOTATORS.build(lama_anno).to(we.device_id)
        mask = np.zeros_like(self.image)
        mask = cv2.rectangle(mask, (0, 0), (150, 150), (255, 255, 255), -1)
        mask = mask[:, :, 0]
        lama_res = lama_ins(self.image, mask)
        print("lama's shape:", lama_res.shape)
        Image.fromarray(lama_res).save(os.path.join(self.save_dir, f'sunflower_lama_mask2.png'))


if __name__ == '__main__':
    unittest.main()
