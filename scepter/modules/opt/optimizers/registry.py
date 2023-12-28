# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect

from scepter.modules.utils.registry import Registry, deep_copy


def build_optimizer(cfg, registry, logger=None, *args, **kwargs):
    from scepter.modules.utils.config import Config
    if not isinstance(cfg, Config):
        raise TypeError(f'config must be type dict, got {type(cfg)}')
    if not cfg.have('NAME'):
        raise KeyError(f'config must contain key NAME, got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError(
            f'registry must be type Registry, got {type(registry)}')

    assert kwargs is not None and 'parameters' in kwargs
    parameters = kwargs['parameters']

    cfg = deep_copy(cfg)

    req_type = cfg.get('NAME')
    if isinstance(req_type, str):
        req_type_entry = registry.get(req_type)
        if req_type_entry is None:
            raise KeyError(f'{req_type} not found in {registry.name} registry')
    if inspect.isclass(req_type_entry):
        try:
            Opti = req_type_entry(cfg, logger=logger)
            return Opti(parameters)
        except Exception as e:
            raise Exception(f'Failed to init class {req_type_entry}, with {e}')
    else:
        raise TypeError(
            f'type must be str or class, got {type(req_type_entry)}')


OPTIMIZERS = Registry('OPTIMIZERS', build_func=build_optimizer)
