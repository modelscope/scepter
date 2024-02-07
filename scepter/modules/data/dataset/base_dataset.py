# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from scepter.modules.transform.registry import TRANSFORMS, build_pipeline
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger
from scepter.modules.utils.registry import old_python_version


class BaseDataset(Dataset, metaclass=ABCMeta):
    para_dict = {
        'MODE': {
            'value': 'train',
            'description': 'solver phase, select from [train, test, eval]'
        },
        'FILE_SYSTEM': {},
        'TRANSFORMS': [{
            'ImageToTensor': {
                'PADDING': {
                    'value': None,
                    'description': 'padding'
                },
                'PAD_IF_NEEDED': {
                    'value': False,
                    'description': 'pad if needed'
                },
                'FILL': {
                    'value': 0,
                    'description': 'fill'
                },
                'PADDING_MODE': {
                    'value': 'constant',
                    'description': 'padding mode'
                }
            }
        }]
    }

    def __init__(self, cfg, logger=None):
        mode = cfg.get('MODE', 'train')
        pipeline = cfg.get('TRANSFORMS', [])
        super(BaseDataset, self).__init__()
        self.mode = mode
        self.logger = logger
        self.worker_logger = get_logger(name='datasets')
        self.pipeline = build_pipeline(pipeline,
                                       TRANSFORMS,
                                       logger=self.worker_logger)
        self.file_systems = cfg.get('FILE_SYSTEM', None)

        if isinstance(self.file_systems, list):
            for file_sys in self.file_systems:
                self.fs_prefix = FS.init_fs_client(file_sys,
                                                   logger=self.logger,
                                                   overwrite=False)
        elif self.file_systems is not None:
            self.fs_prefix = FS.init_fs_client(self.file_systems,
                                               logger=self.logger,
                                               overwrite=False)
        self.local_we = we.get_env()
        if old_python_version:
            self.file_systems.logger = None

    def __getitem__(self, index: int):
        item = self._get(index)
        return self.pipeline(item)

    def worker_init_fn(self, worker_id, num_workers=1):
        if isinstance(self.file_systems, list):
            for file_sys in self.file_systems:
                self.fs_prefix = FS.init_fs_client(file_sys,
                                                   logger=self.logger,
                                                   overwrite=False)
        elif self.file_systems is not None:
            self.fs_prefix = FS.init_fs_client(self.file_systems,
                                               logger=self.logger,
                                               overwrite=False)
        self.worker_id = worker_id
        self.logger = self.worker_logger
        we.set_env(self.local_we)

    @abstractmethod
    def _get(self, index: int):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}: mode={self.mode}, len={len(self)}'

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
        return dict_to_yaml('DATASETS',
                            __class__.__name__,
                            BaseDataset.para_dict,
                            set_name=True)
