# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from torch.utils.data.sampler import Sampler

from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we


class BaseSampler(Sampler):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.seed = cfg.get('SEED', we.seed)
        self.batch_size = cfg.get('BATCH_SIZE', 1)

    def __iter__(self):
        pass

    def __len__(self):
        return 1

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
        return dict_to_yaml('SAMPLERS',
                            __class__.__name__,
                            BaseSampler.para_dict,
                            set_name=True)
