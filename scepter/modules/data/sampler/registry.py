# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect

from scepter.modules.utils.registry import (Registry, deep_copy,
                                            old_python_version)


def build_sampler_config(cfg, registry, logger=None, **kwargs):
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
            sampler = req_type_entry(cfg, logger=logger)
            return sampler
        except Exception as e:
            raise Exception(f'Failed to init class {req_type_entry}, with {e}')
    else:
        raise TypeError(f'type must be class, got {type(req_type_entry)}')


SAMPLERS = Registry('SAMPLERS', build_func=build_sampler_config)
