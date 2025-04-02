# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import numpy as np

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config


@ANNOTATORS.register_class()
class FrameReferenceAnnotator(BaseAnnotator):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        # first / last / firstlast / random
        self.ref_cfg = cfg.get('REF_CFG', [{"mode": "first", "proba": 0.1},
                                           {"mode": "last", "proba": 0.1},
                                           {"mode": "firstlast", "proba": 0.1},
                                           {"mode": "random", "proba": 0.1}])
        self.ref_num = cfg.get('REF_NUM', 1)
        self.ref_cfg = Config.get_dict(self.ref_cfg) if isinstance(
            self.ref_cfg, Config) else self.ref_cfg
        self.ref_color = cfg.get('REF_COLOR', 127.5)

    def forward(self, frames, ref_cfg=None, ref_num=None):
        ref_cfg = ref_cfg if ref_cfg is not None else self.ref_cfg
        ref_cfg = [ref_cfg] if not isinstance(ref_cfg, list) else ref_cfg
        probas = [item['proba'] if 'proba' in item else 1.0 / len(ref_cfg) for item in ref_cfg]
        sel_ref_cfg = random.choices(ref_cfg, weights=probas, k=1)[0]
        mode = sel_ref_cfg['mode'] if 'mode' in sel_ref_cfg else 'original'
        ref_num = int(ref_num) if ref_num is not None else self.ref_num

        frame_num = len(frames)
        frame_num_range = list(range(frame_num))
        if mode == "first":
            sel_idx = frame_num_range[:ref_num]
        elif mode == "last":
            sel_idx = frame_num_range[-ref_num:]
        elif mode == "firstlast":
            sel_idx = frame_num_range[:ref_num] + frame_num_range[-ref_num:]
        elif mode == "random":
            sel_idx = random.sample(frame_num_range, ref_num)
        else:
            raise NotImplementedError

        out_frames, out_masks = [], []
        for i in range(frame_num):
            if i in sel_idx:
                out_frame = frames[i]
                out_mask = np.zeros_like(frames[i][:, :, 0])
            else:
                out_frame = np.ones_like(frames[i]) * self.ref_color
                out_mask = np.ones_like(frames[i][:, :, 0]) * 255
            out_frames.append(out_frame)
            out_masks.append(out_mask)
        return out_frames, out_masks
