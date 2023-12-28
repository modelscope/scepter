# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from ..utils.config import dict_to_yaml
from .registry import TRANSFORMS


@TRANSFORMS.register_class()
class Identity(object):
    def __init__(self, cfg, logger=None):
        pass

    def __call__(self, item):
        return item

    @staticmethod
    def get_config_template():
        '''
        { "ENV" :
            { "description" : "",
              "A" : {
                    "value": 1.0,
                    "description": ""
               }
            }
        }
        :return:
        '''
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__, [{}],
                            set_name=True)
