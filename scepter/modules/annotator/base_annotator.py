# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

import torch
import torch.nn as nn

from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.model.base_model import BaseModel
from scepter.modules.utils.config import dict_to_yaml


@ANNOTATORS.register_class()
class BaseAnnotator(BaseModel, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            BaseAnnotator.para_dict,
                            set_name=True)


@ANNOTATORS.register_class()
class GeneralAnnotator(BaseAnnotator, metaclass=ABCMeta):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        anno_models = cfg.get('ANNOTATORS', [])
        self.annotators = nn.ModuleList()
        for n, anno_config in enumerate(anno_models):
            annotator = ANNOTATORS.build(anno_config, logger=logger)
            annotator.input_keys = anno_config.get('INPUT_KEYS', [])
            if isinstance(annotator.input_keys, str):
                annotator.input_keys = [annotator.input_keys]
            annotator.output_keys = anno_config.get('OUTPUT_KEYS', [])
            if isinstance(annotator.output_keys, str):
                annotator.output_keys = [annotator.output_keys]
            assert len(annotator.input_keys) == len(annotator.output_keys)
            self.annotators.append(annotator)

    def forward(self, input_dict):
        output_dict = {}
        for annotator in self.annotators:
            for idx, in_key in enumerate(annotator.input_keys):
                if in_key in input_dict:
                    image = annotator(input_dict[in_key])
                    output_dict[annotator.output_keys[idx]] = image
        return output_dict
