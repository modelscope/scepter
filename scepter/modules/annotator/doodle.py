# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

import torch
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml


@ANNOTATORS.register_class()
class DoodleAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.processor_type = cfg.get('PROCESSOR_TYPE', 'pidinet_sketch')
        processor_cfg = cfg.get('PROCESSOR_CFG', None)
        if self.processor_type == 'pidinet_sketch':
            self.pidinet_ins = ANNOTATORS.build(processor_cfg[0])
            self.sketch_ins = ANNOTATORS.build(processor_cfg[1])
        else:
            raise 'Unsurpport PROCESSOR for DoodleAnnotator'

    @torch.no_grad()
    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        if self.processor_type == 'pidinet_sketch':
            pidinet_res = self.pidinet_ins(image)
            sketch_res = self.sketch_ins(pidinet_res)
            doodle_res = sketch_res
        else:
            raise 'Unsurpport PROCESSOR for DoodleAnnotator'
        return doodle_res

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            DoodleAnnotator.para_dict,
                            set_name=True)
