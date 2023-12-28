# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from scepter.modules.transform.registry import TRANSFORMS
from scepter.modules.utils.config import dict_to_yaml


@TRANSFORMS.register_class()
class Compose(object):
    """ Compose all transform function into one.

    Args:
        transform (List[dict]): List of transform configs.

    """
    def __init__(self, cfg, logger=None):
        self.transforms = [TRANSFORMS.build(t) for t in cfg.TRANSFORMS]

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
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
