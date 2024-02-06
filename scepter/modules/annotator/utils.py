# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image, (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img, k


def resize_image_ori(h, w, image, k):
    img = cv2.resize(
        image, (w, h),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


class AnnotatorProcessor():
    canny_cfg = {
        'NAME': 'CannyAnnotator',
        'LOW_THRESHOLD': 100,
        'HIGH_THRESHOLD': 200,
        'INPUT_KEYS': ['img'],
        'OUTPUT_KEYS': ['canny']
    }
    hed_cfg = {
        'NAME': 'HedAnnotator',
        'PRETRAINED_MODEL':
        'ms://damo/scepter_scedit@annotator/ckpts/ControlNetHED.pth',
        'INPUT_KEYS': ['img'],
        'OUTPUT_KEYS': ['hed']
    }
    openpose_cfg = {
        'NAME': 'OpenposeAnnotator',
        'BODY_MODEL_PATH':
        'ms://damo/scepter_scedit@annotator/ckpts/body_pose_model.pth',
        'HAND_MODEL_PATH':
        'ms://damo/scepter_scedit@annotator/ckpts/hand_pose_model.pth',
        'INPUT_KEYS': ['img'],
        'OUTPUT_KEYS': ['openpose']
    }
    midas_cfg = {
        'NAME': 'MidasDetector',
        'PRETRAINED_MODEL':
        'ms://damo/scepter_scedit@annotator/ckpts/dpt_hybrid-midas-501f0c75.pt',
        'INPUT_KEYS': ['img'],
        'OUTPUT_KEYS': ['depth']
    }
    mlsd_cfg = {
        'NAME': 'MLSDdetector',
        'PRETRAINED_MODEL':
        'ms://damo/scepter_scedit@annotator/ckpts/mlsd_large_512_fp32.pth',
        'INPUT_KEYS': ['img'],
        'OUTPUT_KEYS': ['mlsd']
    }
    color_cfg = {
        'NAME': 'ColorAnnotator',
        'RATIO': 64,
        'INPUT_KEYS': ['img'],
        'OUTPUT_KEYS': ['color']
    }

    anno_type_map = {
        'canny': canny_cfg,
        'hed': hed_cfg,
        'pose': openpose_cfg,
        'depth': midas_cfg,
        'mlsd': mlsd_cfg,
        'color': color_cfg
    }

    def __init__(self, anno_type):
        from scepter.modules.annotator.registry import ANNOTATORS
        from scepter.modules.utils.config import Config
        from scepter.modules.utils.distribute import we

        if isinstance(anno_type, str):
            assert anno_type in self.anno_type_map.keys()
            anno_type = [anno_type]
        elif isinstance(anno_type, (list, tuple)):
            assert all(tp in self.anno_type_map.keys() for tp in anno_type)
        else:
            raise Exception(f'Error anno_type: {anno_type}')

        general_dict = {
            'NAME': 'GeneralAnnotator',
            'ANNOTATORS': [self.anno_type_map[tp] for tp in anno_type]
        }
        general_anno = Config(cfg_dict=general_dict, load=False)
        self.general_ins = ANNOTATORS.build(general_anno).to(we.device_id)

    def run(self, image, anno_type=None):
        output_image = self.general_ins({'img': image})
        if anno_type is not None:
            if isinstance(anno_type, str) and anno_type in output_image:
                return output_image[anno_type]
            else:
                return {
                    tp: output_image[tp]
                    for tp in anno_type if tp in output_image
                }
        else:
            return output_image
