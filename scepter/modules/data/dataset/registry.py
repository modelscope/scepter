# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import collections.abc as container_abcs
import inspect
import re
from functools import partial

import torch
from torch.utils.data import DataLoader, DistributedSampler

from scepter.modules.data.sampler import (SAMPLERS, MixtureOfSamplers,
                                          MultiFoldDistributedSampler,
                                          MultiLevelBatchSampler)
from scepter.modules.utils.registry import (Registry, deep_copy,
                                            old_python_version)

string_classes = (str, bytes)
int_classes = (int, )

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    'default_collate: batch must contain tensors, numpy arrays, numbers, '
    'dicts or lists; found {}')


def gpu_batch_collate(batch, device_id=0):
    """ Modified from pytorch default collate function.
    When using gpu operation in preprocess pipelines, tensor could be on different devices.
    While storage._new_shared will use only cuda:0, it will crash at elem.new(storage).

    Args:
        batch (list): List of contents to be collated.
        device_id (int): GPU device id where cuda type tensor will be on, default is 0.

    Returns:
        Inputs in batch.

    Raises:
        TypeError: Only support tensors, numpy arrays, numbers, dicts, lists, strings.
    """
    elem: object = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if not elem.is_cuda:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        else:
            return torch.stack(batch, 0).cuda(device_id)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return gpu_batch_collate([torch.as_tensor(b) for b in batch],
                                     device_id=device_id)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {
            key: gpu_batch_collate([d[key] for d in batch],
                                   device_id=device_id)
            for key in elem
        }
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(gpu_batch_collate(samples, device_id=device_id)
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [
            gpu_batch_collate(samples, device_id=device_id)
            for samples in transposed
        ]
    raise TypeError(default_collate_err_msg_format.format(elem_type))


class DataObject(object):
    para_dict = {
        'PIN_MEMORY': {
            'value': False,
            'description': 'pin_memory for data loader'
        },
        'SHUFFLE': {
            'value': False,
            'description': 'SHUFFLE the list or not!'
        },
        'BATCH_SIZE': {
            'value': 4,
            'description': 'batch size for data'
        },
        'NUM_WORKERS': {
            'value': 1,
            'description': 'num workers for fetching data!'
        },
        'NUM_FOLDS': {
            'value':
            0,
            'description':
            'if set, use MultiFoldDistributedSampler for distribute training!'
        },
        'SAMPLER': {
            'NAME': {
                'value':
                None,
                'description':
                'If set None, system will choose the sampler according to world size. This means '
                'when world size > 1 and num fold > 1, choose MultiFoldDistributedSampler,'
                'when world size > 1 and num fold <= 1, choose  DistributedSampler,'
                'when world size = 1, choose default torch dataloader sampler.'
                'If set TorchDefault, choose default torch dataloader, which means every process will use all dataset.'
                'If set MultiLevelBatchSampler, choose MultiLevelBatchSampler, which loads data from index file.'
                'If set MixtureOfSamplers, choose MixtureOfSamplers, which means you can use different datasets from '
                'different sources with multi MultiLevelBatchSampler by setting SUB_SAMPLERS.'
            },
            'INDEX_FILE': {
                'value':
                None,
                'description':
                'Set your index file when you choose MultiLevelBatchSampler'
            },
            'SUB_SAMPLERS': []
        }
    }

    def __init__(self, cfg, dataset, logger=None):
        self.dataset = dataset

        self.logger = logger
        self.pin_memory = cfg.get('PIN_MEMORY', False)
        self.batch_size = cfg.get('BATCH_SIZE', 4)
        self.num_workers = cfg.get('NUM_WORKERS', 1)
        self.data_sampler_config = cfg.get('SAMPLER', None)
        self.shuffle = cfg.get('MODE', 'test') == 'train'
        self.cfg = cfg
        worker_init_fn = dataset.worker_init_fn if hasattr(
            dataset, 'worker_init_fn') else None
        if worker_init_fn is not None:
            worker_init_fn = partial(worker_init_fn,
                                     num_workers=self.num_workers)

        self.registry_sampler(dataset)

        collate_fn = dataset.collate_fn if hasattr(dataset,
                                                   'collate_fn') else None
        if self.batch_sampler:
            self.dataloader = DataLoader(
                dataset,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=self.pin_memory,
                prefetch_factor=None if self.num_workers == 0 else 2,
                worker_init_fn=worker_init_fn,
                persistent_workers=self.num_workers > 0,
                timeout=2400 if self.num_workers > 0 else 0)
        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                sampler=self.sampler,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                worker_init_fn=worker_init_fn,
                persistent_workers=self.num_workers > 0,
                timeout=2400 if self.num_workers > 0 else 0)
        if self.logger is not None:
            self.logger.info(
                f'Built dataloader with len {len(self.dataloader)}')

    def reload(self, dataset):
        worker_init_fn = dataset.worker_init_fn if hasattr(
            dataset, 'worker_init_fn') else None
        if worker_init_fn is not None:
            worker_init_fn = partial(worker_init_fn,
                                     num_workers=self.num_workers)

        self.registry_sampler(dataset)

        collate_fn = dataset.collate_fn if hasattr(dataset,
                                                   'collate_fn') else None

        if self.batch_sampler:
            dataloader = DataLoader(
                dataset,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=self.pin_memory,
                prefetch_factor=None if self.num_workers == 0 else 2,
                worker_init_fn=worker_init_fn,
                persistent_workers=self.num_workers > 0,
                timeout=2400 if self.num_workers > 0 else 0)
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                sampler=self.sampler,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                worker_init_fn=worker_init_fn,
                persistent_workers=self.num_workers > 0,
                timeout=2400 if self.num_workers > 0 else 0)

        if self.logger is not None:
            self.logger.info(f'Reload dataloader with len {len(dataloader)}')
        return dataloader

    def registry_sampler(self, dataset):
        self.sampler, self.batch_sampler, self.drop_last = None, None, False
        from scepter.modules.utils.distribute import we
        seed, world_size, rank = we.seed, we.world_size, we.rank
        if not self.data_sampler_config:
            if world_size > 1:
                if self.cfg.have('NUM_FOLDS') and self.cfg.NUM_FOLDS > 1:
                    num_folds = self.cfg.NUM_FOLDS
                    print('num folds', num_folds)
                    self.sampler = MultiFoldDistributedSampler(
                        dataset,
                        num_folds,
                        world_size,
                        rank,
                        shuffle=self.shuffle)
                else:
                    self.sampler = DistributedSampler(dataset,
                                                      world_size,
                                                      rank,
                                                      shuffle=self.shuffle)
                # collate_fn = partial(gpu_batch_collate, device_id=device_id)
                self.drop_last = False
                self.shuffle = False
            else:
                self.sampler = None
                # collate_fn = None
                self.drop_last = False
        else:
            sampler_name = self.data_sampler_config.get('NAME', None)
            if sampler_name == 'MixtureOfSamplers':
                subsampler_configs = self.data_sampler_config.get(
                    'SUB_SAMPLERS', [])
                subsamplers = list()
                subsampler_probs = list()
                for ssconfig in subsampler_configs:
                    name = ssconfig.get('NAME')
                    prob = ssconfig.get('PROB', 0.0)
                    if name == 'MultiLevelBatchSampler':
                        subsampler = self._instantiate_multi_level_batch_sampler(
                            ssconfig, self.batch_size, rank, seed)
                        subsamplers.append(subsampler)
                    else:
                        # surpport register
                        ssconfig.SEED = seed
                        ssconfig.BATCH_SIZE = self.batch_size
                        subsampler = SAMPLERS.build(ssconfig,
                                                    logger=self.logger)
                        subsamplers.append(subsampler)
                    subsampler_probs.append(prob)
                self.batch_sampler = MixtureOfSamplers(subsamplers,
                                                       subsampler_probs, rank,
                                                       seed)
            elif sampler_name == 'MultiLevelBatchSampler':
                self.batch_sampler = self._instantiate_multi_level_batch_sampler(
                    self.data_sampler_config, self.batch_size, rank, seed)
            elif sampler_name == 'TorchDefault':
                self.sampler = None
            else:
                self.shuffle = False
                self.data_sampler_config.SEED = seed
                self.data_sampler_config.BATCH_SIZE = self.batch_size
                sampler = SAMPLERS.build(self.data_sampler_config,
                                         logger=self.logger)
                if sampler_name.endswith('BatchSampler'):
                    self.batch_sampler = sampler
                else:
                    self.sampler = sampler

    def _instantiate_multi_level_batch_sampler(self, sampler_config,
                                               batch_size, rank, seed):
        index_file = sampler_config.INDEX_FILE
        image_size = sampler_config.get('IMAGE_SIZE', 512)
        fields = sampler_config.get('FIELDS', ['img_path', 'prompt'])
        delimiter = sampler_config.get('DELIMITER', ',')
        path_prefix = sampler_config.get('PATH_PREFIX', '')
        prompt_prefix = sampler_config.get('PROMPT_PREFIX', '')
        return MultiLevelBatchSampler(batch_size, index_file, image_size,
                                      fields, delimiter, path_prefix,
                                      prompt_prefix, rank, seed)


def build_dataset_config(cfg, registry, logger=None, *args, **kwargs):
    """ Default builder function.

    Args:
        cfg (objective attribution): A set of objective attirbutions which contain
        parameters passes to target class or function.
            Must contains key 'type', indicates the target class or function name.
        registry (Registry): An registry to search target class or function.
        kwargs (dict, optional): Other params not in config dict.

    Returns:
        Target class object or object returned by invoking function.

    Raises:
        TypeError:
        KeyError:
        Exception:
    """
    from scepter.modules.utils.config import Config
    if not isinstance(cfg, Config):
        raise TypeError(f'config must be type dict, got {type(cfg)}')
    if not cfg.have('NAME'):
        raise KeyError(f'config must contain key NAME, got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError(
            f'registry must be type Registry, got {type(registry)}')

    cfg = deep_copy(cfg)

    req_type = cfg.get('NAME')
    if isinstance(req_type, str):
        req_type_entry = registry.get(req_type)
        if req_type_entry is None:
            raise KeyError(f'{req_type} not found in {registry.name} registry')

    if kwargs is not None:
        cfg._update_dict(kwargs)

    if old_python_version:
        logger = None

    if inspect.isclass(req_type_entry):
        try:
            dataset = req_type_entry(cfg, logger=logger, *args, **kwargs)
            do = DataObject(cfg, dataset, logger=logger)
            return do
        except Exception as e:
            raise Exception(f'Failed to init class {req_type_entry}, with {e}')
    else:
        raise TypeError(f'type must be class, got {type(req_type_entry)}')


DATASETS = Registry('DATASETS',
                    common_para=DataObject.para_dict,
                    build_func=build_dataset_config)
