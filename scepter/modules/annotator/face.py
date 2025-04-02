# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


@ANNOTATORS.register_class()
class FaceAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        from insightface.app import FaceAnalysis
        local_path = FS.map_to_local(cfg.PRETRAINED_MODEL)[0]
        local_model_path = os.path.join(local_path, 'models', cfg.MODEL_NAME)
        FS.get_dir_to_local_dir(cfg.PRETRAINED_MODEL, local_model_path)
        self.model = FaceAnalysis(name=cfg.MODEL_NAME, root=local_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=we.device_id, det_size=(640, 640))

    def forward(self, image=None):

        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        # [dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])]
        faces = self.model.get(image)
        return faces


@ANNOTATORS.register_class()
class FaceMaskAnnotator(FaceAnnotator):

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.multi_face = cfg.get('MULTI_FACE', True)

    def forward(self, image=None):
        faces = super().forward(image)
        if len(faces) > 0:
            if not self.multi_face:
                faces = faces[:1]
            mask = np.zeros_like(image[:, :, 0])
            for face in faces:
                x_min, y_min, x_max, y_max = face['bbox'].tolist()
                mask[int(y_min): int(y_max) + 1, int(x_min): int(x_max) + 1] = 255
            return mask
        else:
            return np.zeros_like(image[:, :, 0])
