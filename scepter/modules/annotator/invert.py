# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml


@ANNOTATORS.register_class()
class InvertAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

    def forward(self, image):
        return 255 - image

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            InvertAnnotator.para_dict,
                            set_name=True)
