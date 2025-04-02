# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

# ``` requirements for cuda 12.1:
#     onnxruntime==1.19
#     onnxruntime-gpu==1.19
# ```

import os

import numpy as np
import torch
from PIL import Image

import cv2
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.dwpose import util
from scepter.modules.annotator.dwpose.wholebody import (HWC3, Wholebody,
                                                        resize_image)
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


@ANNOTATORS.register_class()
class DWposeAnnotator(BaseAnnotator):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        with FS.get_from(cfg['DETECTION_MODEL'],
                         wait_finish=True) as onnx_det, FS.get_from(
                             cfg['POSE_MODEL'], wait_finish=True) as onnx_pose:
            self.pose_estimation = Wholebody(onnx_det,
                                             onnx_pose,
                                             device=f'cuda:{we.device_id}')
        self.resize_size = cfg.get('RESIZE_SIZE', 1024)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        input_image = HWC3(image[..., ::-1])
        return self.process(resize_image(input_image, self.resize_size),
                            image.shape[:2])

    def process(self, ori_img, ori_shape):
        ori_h, ori_w = ori_shape
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(ori_img)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            ret_data = {}
            if self.use_body:
                detected_map_body = draw_pose(pose, H, W, use_body=True)
                detected_map_body = cv2.resize(
                    detected_map_body[..., ::-1], (ori_w, ori_h),
                    interpolation=cv2.INTER_LANCZOS4
                    if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data['detected_map_body'] = detected_map_body

            if self.use_face:
                detected_map_face = draw_pose(pose, H, W, use_face=True)
                detected_map_face = cv2.resize(
                    detected_map_face[..., ::-1], (ori_w, ori_h),
                    interpolation=cv2.INTER_LANCZOS4
                    if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data['detected_map_face'] = detected_map_face

            if self.use_body and self.use_face:
                detected_map_bodyface = draw_pose(pose,
                                                  H,
                                                  W,
                                                  use_body=True,
                                                  use_face=True)
                detected_map_bodyface = cv2.resize(
                    detected_map_bodyface[..., ::-1], (ori_w, ori_h),
                    interpolation=cv2.INTER_LANCZOS4
                    if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data['detected_map_bodyface'] = detected_map_bodyface

            if self.use_hand and self.use_body and self.use_face:
                detected_map_handbodyface = draw_pose(pose,
                                                      H,
                                                      W,
                                                      use_hand=True,
                                                      use_body=True,
                                                      use_face=True)
                detected_map_handbodyface = cv2.resize(
                    detected_map_handbodyface[..., ::-1], (ori_w, ori_h),
                    interpolation=cv2.INTER_LANCZOS4
                    if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data[
                    'detected_map_handbodyface'] = detected_map_handbodyface

            # convert_size
            if det_result.shape[0] > 0:
                w_ratio, h_ratio = ori_w / W, ori_h / H
                det_result[..., ::2] *= h_ratio
                det_result[..., 1::2] *= w_ratio
                det_result = det_result.astype(np.int32)
            # for det_tup in det_result:
            #     cv2.rectangle(detected_map, det_tup[2:].tolist(), det_tup[:2].tolist(), color=(255, 0, 0), thickness=3)
            return ret_data, det_result


@ANNOTATORS.register_class()
class DWposeBodyAnnotator(DWposeAnnotator):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.use_body, self.use_face, self.use_hand = True, False, False

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_body']


@ANNOTATORS.register_class()
class DWposeFaceAnnotator(DWposeAnnotator):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.use_body, self.use_face, self.use_hand = False, True, False

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_face']


@ANNOTATORS.register_class()
class DWposeBodyFaceAnnotator(DWposeAnnotator):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.use_body, self.use_face, self.use_hand = True, True, False

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_bodyface']
