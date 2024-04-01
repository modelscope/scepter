# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torchvision

from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.data.dataset.registry import DATASETS
from scepter.modules.utils.config import dict_to_yaml


@DATASETS.register_class()
class ImageClassifyExampleDataset(BaseDataset):
    """
    Dataset for image classification wrapper

    Args:
        json_path (str): json file which contains all instances, should be a list of dict
            which contains img_path and gt_label
        image_dir (str or None): image directory, if None, img_path in json_path will be considered as absolute path
        classes (list[str] or None): image class description
    """
    para_dict = {
        'DATASET': {
            'value': 'cifar10',
            'description': 'the public dataset name'
        },
        'DATA_ROOT': {
            'value': '',
            'description': 'the download data save path'
        }
    }

    para_dict.update(BaseDataset.para_dict)

    def __init__(self, cfg, logger=None):

        super(ImageClassifyExampleDataset, self).__init__(cfg, logger=logger)

        self.dataset_name = cfg.DATASET
        self.data_root = cfg.DATA_ROOT
        self.phase = cfg.MODE
        if self.dataset_name == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(
                root=self.data_root,
                train=self.phase == 'train',
                download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get(self, index: int):
        img, target = self.dataset.__getitem__(index)
        ret = {
            'meta': {},
            'label': np.asarray(target, dtype=np.int64),
            'img': img
        }
        return ret

    def worker_init_fn(self, worker_id, num_workers=1):
        super(ImageClassifyExampleDataset,
              self).worker_init_fn(worker_id, num_workers=num_workers)

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
        return dict_to_yaml('modename_DATA',
                            __class__.__name__,
                            ImageClassifyExampleDataset.para_dict,
                            set_name=True)
