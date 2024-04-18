# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import math
import numbers
import os
import sys
from collections.abc import Iterable
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

from scepter.modules.data.sampler.base_sampler import BaseSampler
from scepter.modules.data.sampler.registry import SAMPLERS
from scepter.modules.data.utils.data_bucket import (BucketBatchIndex,
                                                    BucketManager)
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.directory import osp_path
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


@SAMPLERS.register_class()
class MultiLevelBatchSamplerMultiSource(BaseSampler):
    """Sampler for database with multi-level indexing.
    """
    para_dict = {
        'FIELDS': {
            'value': ['img_path', 'width', 'height', 'prompt'],
            'description': 'The fields list for input record.'
        },
        'DELIMITER': {
            'value': ',',
            'description': 'The fields delimiter for input record.'
        },
        'PATH_PREFIX': {
            'value': 'datasets',
            'description': 'The path prefix for input oss key.'
        },
        'INDEX_FILE': {
            'value': '',
            'description': 'The index file.'
        },
        'PROB': {
            'value': 1.0,
            'description': 'The prob for current sampler.'
        },
        'SELECT_SOURCES': {
            'value':
            None,
            'description':
            'Select data name from index, default is None, which means use all data; '
            'use as a [] of data key to select the used name'
        },
        'SUB_DATA_WEIGHTS': {
            'value': {},
            'description':
            'The prob for sub data weights, default is 1, '
            'which means computing the prob according to data num ratio of total num.'
        },
        'SUB_RESOLUTION_MAP': {
            'value': {},
            'description':
            'The resolution map for im_type, if the resolution is in im_type, please ignore this para.'
        },
        'KARGS': {
            'value': {},
            'description':
            'The extended parameters for transfering to downstream.'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.rng = np.random.default_rng(self.seed + we.rank)
        self.fields = cfg.get('FIELDS', [])
        self.num_fields = len(self.fields)
        self.delimiter = cfg.get('DELIMITER', ',')
        self.path_prefix = cfg.get('PATH_PREFIX', '')
        common_prob = cfg.get('PROB', 1)
        sub_data_weights = cfg.get('SUB_DATA_WEIGHTS', None)
        sub_data_weights = {} if sub_data_weights is None else sub_data_weights.get_dict(
        )
        sub_resolution_map = cfg.get('SUB_RESOLUTION_MAP', None)
        sub_resolution_map = {} if sub_resolution_map is None else sub_resolution_map.get_dict(
        )
        sub_resolution_map = {
            k.lower(): v
            for k, v in sub_resolution_map.items()
        }
        kargs = cfg.get('KARGS', None)
        kargs = {} if kargs is None else kargs.get_dict()
        select_sources = cfg.get('SELECT_SOURCES', None)
        if isinstance(select_sources, list) and len(select_sources) < 1:
            raise 'SELECT_SOURCES must be None or non-empty list.'
        self.kargs = {}

        for k, v in kargs.items():
            if isinstance(v, list) and len(v) == 0:
                continue
            if isinstance(v, dict):
                v = {k.lower(): vv for k, vv in v.items()}
            if v is None:
                continue
            self.kargs[k.lower()] = v
        # read dataset according to the source fields.
        index_file = cfg.INDEX_FILE
        assert index_file.endswith('.json')
        with FS.get_object(index_file) as local_data:
            index = json.loads(local_data.decode('utf-8'))
        self.sub_data_list = []
        self.key_args = {}
        sub_data_num = []
        for key in index:
            im_type = index[key]['image_type']
            if 'image_size' not in index[key]:
                assert im_type in sub_resolution_map
                index[key]['image_size'] = sub_resolution_map[im_type]
            index[key]['data_key'] = key
            data_name = index[key]['data_name']
            if select_sources is not None and data_name not in select_sources:
                continue
            sub_data_num.append(index[key]['total'] *
                                sub_data_weights.get(data_name, 1))
            self.sub_data_list.append(index[key])
            if data_name in self.kargs:
                self.key_args[data_name] = self.kargs.pop(data_name)

        self.probabilities = np.array(sub_data_num) / np.sum(
            np.array(sub_data_num))
        self.probabilities = self.probabilities.tolist()
        for sub_data, p in zip(self.sub_data_list, self.probabilities):
            logger.info(
                f"{sub_data['data_key']}'s sample prob: {p} * {common_prob} = "
                f"{p * common_prob} and samples'num: {sub_data['total']} in this cluster."
            )
        self.rng = np.random.default_rng(self.seed + we.rank)
        self.oss_prefix = '/'.join(index_file.split('/')[:3])
        self.index_dir = os.path.dirname(index_file)

    def __iter__(self):
        while True:
            index_id = self.rng.choice(len(self.sub_data_list),
                                       p=self.probabilities)
            index = self.sub_data_list[index_id]
            image_size = index['image_size']
            data_name = index['data_name']
            data_key = index['data_key']
            batch = []
            while len(batch) < self.batch_size:
                n = self.batch_size - len(batch)
                # read items
                items = index['list']
                for _ in range(index['index_level'] - 1):
                    list_file = self.rng.choice(items)
                    list_file = osp_path(self.oss_prefix, list_file)
                    if not list_file.startswith(self.index_dir):
                        list_file = os.path.join(
                            self.index_dir,
                            '/'.join(list_file.split('/')[-2:]))
                    with FS.get_object(list_file) as f:
                        items = f.decode('utf-8').strip().split('\n')

                # sample into batch
                m = min(n, len(items))
                batch += [
                    i
                    for i in self.rng.choice(items, m, replace=False).tolist()
                ]

                # check batch size
                if len(batch) == self.batch_size:
                    break
            assert len(batch) == self.batch_size

            ret_data = []
            for u in batch:
                one_data = {}
                res_list = u.split(self.delimiter, self.num_fields - 1)
                for idx, res in enumerate(res_list):
                    one_data[self.fields[idx]] = res
                one_data.update(self.kargs)
                if data_name in self.key_args:
                    one_data.update(self.key_args[data_name])
                one_data['image_size'] = image_size
                one_data['data_key'] = data_key
                one_data['prefix'] = osp_path(self.oss_prefix,
                                              self.path_prefix)
                ret_data.append(one_data)
            yield ret_data

    def __len__(self):
        return sys.maxsize

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
                            MultiLevelBatchSamplerMultiSource.para_dict,
                            set_name=True)


@SAMPLERS.register_class()
class MultiFoldDistributedSampler(BaseSampler):
    """Modified from DistributedSampler, which performs multi fold training for
    accelerating distributed training with large batches.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_folds (optional): Number of folds, if 1, will act same as DistributeSampler
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices

    .. warning::
        In distributed mode, calling the ``set_epoch`` method is needed to
        make shuffling work; each process will use the same random seed
        otherwise.
    """
    para_dict = {}

    def __init__(self,
                 dataset,
                 num_folds=1,
                 num_replicas=None,
                 rank=None,
                 shuffle=True):
        """
            When num_folds = 1, MultiFoldDistributedSampler degenerates to DistributedSampler.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_folds = num_folds
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(
                len(self.dataset) * self.num_folds * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = []
        for fold_idx in range(self.num_folds):
            g = torch.Generator()
            g.manual_seed(self.epoch + fold_idx)
            if self.shuffle:
                indices += torch.randperm(len(self.dataset),
                                          generator=g).tolist()
            else:
                indices += list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

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
                            MultiFoldDistributedSampler.para_dict,
                            set_name=True)


@SAMPLERS.register_class()
class EvalDistributedSampler(BaseSampler):
    """Modified from DistributedSampler.

    Notice!
    1. This sampler should only be used in test mode.
    2. This sampler will pad indices or not pad, according to `padding` flag.
     In no padding mode, the last rank device may get samples less than given batch_size.
     The last rank device may have less iteration number than other rank.
     By the way, __len__ function may return a fake number.
    """
    para_dict = {}

    def __init__(self,
                 dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 padding: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError('Invalid rank {}, rank should be in the interval'
                             ' [0, {}]'.format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.padding = padding

        self.perfect_num_samples = math.ceil(
            len(self.dataset) / self.num_replicas)
        self.perfect_total_size = self.perfect_num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.padding and len(indices) < self.perfect_total_size:
            padding_size = self.perfect_total_size - len(indices)
            indices += indices[:padding_size]

        return iter(indices)

    def __len__(self) -> int:
        return self.perfect_num_samples

    def set_epoch(self, epoch: int) -> None:
        pass

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
                            EvalDistributedSampler.para_dict,
                            set_name=True)


@SAMPLERS.register_class()
class MultiLevelBatchSampler(BaseSampler):
    """Sampler for database with multi-level indexing.
    """
    para_dict = {
        'IMAGE_SIZE': {
            'value': [1024, 1024],
            'description': 'The image size for input image.'
        },
        'FIELDS': {
            'value': ['img_path', 'width', 'height', 'prompt'],
            'description': 'The fields list for input record.'
        },
        'DELIMITER': {
            'value': ',',
            'description': 'The fields delimiter for input record.'
        },
        'PATH_PREFIX': {
            'value': 'datasets',
            'description': 'The path prefix for input oss key.'
        },
        'PROMPT_PREFIX': {
            'value': '',
            'description': 'The prompt prefix.'
        },
        'INDEX_FILE': {
            'value': '',
            'description': 'The index file.'
        },
    }

    def __init__(self,
                 batch_size,
                 index_file,
                 image_size=[1024, 1024],
                 fields=['oss_key', 'prompt'],
                 delimiter=',',
                 path_prefix='',
                 prompt_prefix='',
                 rank=0,
                 seed=8888):
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed + rank)
        if isinstance(image_size, numbers.Number):
            image_size = [image_size, image_size]
        assert isinstance(image_size, Iterable) and len(image_size) == 2
        self.image_size = image_size
        self.fields = fields
        self.num_fields = len(fields)
        self.delimiter = delimiter
        self.path_prefix = path_prefix
        self.prompt_prefix = prompt_prefix
        with FS.get_object(index_file) as local_data:
            if index_file.endswith('.json'):
                self.index = json.loads(local_data.decode('utf-8'))
            else:
                self.index = {
                    'list': local_data.decode('utf-8').strip().split('\n'),
                    'index_level': 1,
                    'num_fields': self.num_fields
                }
        self.oss_prefix = '/'.join(index_file.split('/')[:3])
        self.index_dir = os.path.dirname(index_file)

    def __iter__(self):
        while True:
            batch = []
            while len(batch) < self.batch_size:
                n = self.batch_size - len(batch)

                # read items
                items = self.index['list']
                for _ in range(self.index['index_level'] - 1):
                    list_file = self.rng.choice(items)
                    list_file = osp_path(self.oss_prefix, list_file)
                    if not list_file.startswith(self.index_dir):
                        list_file = os.path.join(
                            self.index_dir,
                            '/'.join(list_file.split('/')[-2:]))
                    with FS.get_object(list_file) as f:
                        items = f.decode('utf-8').strip().split('\n')

                # sample into batch
                m = min(n, len(items))
                batch += [
                    osp_path(self.oss_prefix,
                             os.path.join(self.path_prefix, i))
                    for i in self.rng.choice(items, m, replace=False).tolist()
                ]

                # check batch size
                if len(batch) == self.batch_size:
                    break
            assert len(batch) == self.batch_size

            fields = self.fields + ['image_size', 'prompt_prefix']
            yield [
                u.split(self.delimiter, self.num_fields - 1) +
                [self.image_size, self.prompt_prefix, fields] for u in batch
            ]

    def __len__(self):
        return sys.maxsize

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
                            MultiLevelBatchSampler.para_dict,
                            set_name=True)


@SAMPLERS.register_class()
class MixtureOfSamplers(BaseSampler):
    para_dict = {'SUB_SAMPLERS': []}

    def __init__(self, samplers, probabilities, rank=0, seed=8888):
        self.samplers = samplers
        self.iterators = [iter(u) for u in samplers]
        self.probabilities = probabilities
        self.seed = seed
        self.rng = np.random.default_rng(seed + rank)

    def __iter__(self):
        while True:
            index = self.rng.choice(len(self.iterators), p=self.probabilities)
            yield next(self.iterators[index])

    def __len__(self):
        return sys.maxsize

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
                            MixtureOfSamplers.para_dict,
                            set_name=True)


@SAMPLERS.register_class()
class LoopSampler(BaseSampler):
    para_dict = {}

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        rank = we.rank
        self.rng = np.random.default_rng(self.seed + rank)

    def __iter__(self):
        while True:
            yield self.rng.choice(sys.maxsize)

    def __len__(self):
        return sys.maxsize

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
                            LoopSampler.para_dict,
                            set_name=True)


@SAMPLERS.register_class()
class ResolutionBatchSampler(BaseSampler):
    para_dict = {}

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.data_file = cfg.DATA_FILE
        self.fields = cfg.get('FIELDS', [])
        self.num_fields = len(self.fields)
        self.delimiter = cfg.get('DELIMITER', ',')
        self.path_prefix = cfg.get('PATH_PREFIX', '')
        self.batch_size = cfg.BATCH_SIZE
        max_reso = cfg.get('MAX_RESO', (1024, 1024))
        min_bucket_reso = cfg.get('MIN_BUCKET_RESO', 256)
        max_bucket_reso = cfg.get('MAX_BUCKET_RESO', 1024)
        bucket_reso_steps = cfg.get('BUCKET_RESO_STEPS', 64)
        bucket_no_upscale = cfg.get('BUCKET_NO_UPSCALE', False)
        rank = we.rank
        self.rng = np.random.default_rng(self.seed + rank)
        assert 'img_path' in self.fields and 'width' in self.fields and 'height' in self.fields

        self.bucket_manager = BucketManager(max_reso=max_reso,
                                            min_size=min_bucket_reso,
                                            max_size=max_bucket_reso,
                                            reso_steps=bucket_reso_steps,
                                            no_upscale=bucket_no_upscale)
        if not bucket_no_upscale:
            self.bucket_manager.make_buckets()
        else:
            self.logger.info(
                'min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, '
                'because bucket reso is defined by image size automatically / bucket_no_upscale'
            )

        self.data_map = {}
        img_path_idx, width_idx, height_idx = self.fields.index(
            'img_path'), self.fields.index('width'), self.fields.index(
                'height')
        with FS.get_from(self.data_file) as local_path:
            with open(local_path) as f:
                for i, line in enumerate(f):
                    items = line.strip()
                    item_sp = items.split(self.delimiter, self.num_fields - 1)
                    img_path, width, height = item_sp[img_path_idx], int(
                        item_sp[width_idx]), int(item_sp[height_idx])
                    item_sp[img_path_idx] = os.path.join(
                        self.path_prefix, img_path)
                    bucket_reso, resized_size, ar_error = self.bucket_manager.select_bucket(
                        width, height)
                    self.bucket_manager.add_image(reso=bucket_reso, image=i)
                    self.data_map[i] = item_sp

        for i, (reso, bucket) in enumerate(
                zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
            count = len(bucket)
            if count > 0:
                # self.logger.info(f"bucket {i}: resolution {reso}, bucket {bucket}, count: {len(bucket)}")
                self.logger.info(
                    f'bucket {i}: resolution {reso}, count: {len(bucket)}')

        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, (reso, bucket) in enumerate(
                zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(
                    BucketBatchIndex(bucket_index, self.batch_size,
                                     batch_index, reso))
        self.shuffle_buckets()

    def shuffle_buckets(self):
        np.random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def __iter__(self):
        while True:
            index = self.rng.choice(len(self.buckets_indices))
            bucket_reso = self.buckets_indices[index].bucket_reso
            bucket_width, bucket_height = bucket_reso
            bucket = self.bucket_manager.buckets[
                self.buckets_indices[index].bucket_index]
            batches = self.rng.choice(bucket, self.batch_size)
            # image_index = self.buckets_indices[index].batch_index * self.batch_size
            # batch = bucket[image_index : image_index + self.batch_size]
            fields = self.fields + ['image_size', 'prompt_prefix']
            batches = [
                self.data_map[idx] + [[bucket_height, bucket_width], fields]
                for idx in batches
            ]
            yield batches

    def __len__(self):
        return sys.maxsize

    @staticmethod
    def get_config_template():
        return dict_to_yaml('SAMPLERS',
                            __class__.__name__,
                            ResolutionBatchSampler.para_dict,
                            set_name=True)
