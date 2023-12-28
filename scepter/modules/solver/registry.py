# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect

from scepter.modules.utils.registry import Registry, deep_copy


def build_solver(cfg, registry, logger=None, *args, **kwargs):
    from scepter.modules.utils.config import Config
    if not isinstance(cfg, Config):
        raise TypeError(f'config must be type Config, got {type(cfg)}')
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

    if inspect.isclass(req_type_entry):
        try:
            return req_type_entry(cfg, logger=logger, *args, **kwargs)
        except Exception as e:
            raise Exception(f'Failed to init class {req_type_entry}, with {e}')
    else:
        raise TypeError(
            f'type must be str or class, got {type(req_type_entry)}')


SOLVERS = Registry('SOLVERS', build_func=build_solver, allow_types=('class', ))
