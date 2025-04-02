# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import random
import numpy as np
import argparse

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

try:
    from raft import RAFT
    from raft.utils.utils import InputPadder
    from raft.utils import flow_viz
except:
    import warnings
    warnings.warn("ignore raft import, please pip install raft.")


@ANNOTATORS.register_class()
class RAFTAnnotator(BaseAnnotator):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        params = {
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        }
        params = argparse.Namespace(**params)
        model = RAFT(params)
        if cfg.PRETRAINED_MODEL is not None:
            with FS.get_from(cfg.PRETRAINED_MODEL,
                             wait_finish=True) as local_path:
                model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(local_path, map_location="cpu", weights_only=True).items()})
        self.model = model.to(we.device_id).eval()

    def forward(self, frames):
        # frames / RGB
        frames = [torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()[None].to(we.device_id) for frame in frames]
        flow_up_list, flow_up_vis_list = [], []
        with torch.no_grad():
            for i, (image1, image2) in enumerate(zip(frames[:-1], frames[1:])):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flow_up_vis = flow_viz.flow_to_image(flow_up)
                flow_up_list.append(flow_up)
                flow_up_vis_list.append(flow_up_vis)
        return flow_up_list, flow_up_vis_list  # RGB


@ANNOTATORS.register_class()
class RAFTVisAnnotator(RAFTAnnotator):
    def forward(self, frames):
        flow_up_list, flow_up_vis_list = super().forward(frames)
        return flow_up_vis_list[:1] + flow_up_vis_list
