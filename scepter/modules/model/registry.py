# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from scepter.modules.utils.config import Config
from scepter.modules.utils.registry import Registry, build_from_config


def build_model(cfg, registry, logger=None, *args, **kwargs):
    """ After build model, load pretrained model if exists key `pretrain`.

    pretrain (str, dict): Describes how to load pretrained model.
        str, treat pretrain as model path;
        dict: should contains key `path`, and other parameters token by function load_pretrained();
    """
    if not isinstance(cfg, Config):
        raise TypeError(f'Config must be type dict, got {type(cfg)}')
    if cfg.have('PRETRAINED_MODEL'):
        pretrain_cfg = cfg.PRETRAINED_MODEL
        if pretrain_cfg is not None and not isinstance(pretrain_cfg, (str)):
            raise TypeError('Pretrain parameter must be a string')
    else:
        pretrain_cfg = None

    model = build_from_config(cfg, registry, logger=logger, *args, **kwargs)
    if pretrain_cfg is not None:
        if hasattr(model, 'load_pretrained_model'):
            model.load_pretrained_model(pretrain_cfg)
    return model

def build_diffusion(cfg, registry, logger=None, *args, **kwargs):
    """ After build model, load pretrained model if exists key `pretrain`.

    pretrain (str, dict): Describes how to load pretrained model.
        str, treat pretrain as model path;
        dict: should contains key `path`, and other parameters token by function load_pretrained();
    """
    if not isinstance(cfg, Config):
        raise TypeError(f'Config must be type dict, got {type(cfg)}')
    return build_from_config(cfg, registry, logger=logger, *args, **kwargs)

def build_scheduler(cfg, registry, logger=None, *args, **kwargs):
    if not isinstance(cfg, Config):
        raise TypeError(f'Config must be type dict, got {type(cfg)}')
    return build_from_config(cfg, registry, logger=logger, *args, **kwargs)

def build_diffusion_sampler(cfg, registry, logger=None, *args, **kwargs):
    if not isinstance(cfg, Config):
        raise TypeError(f'Config must be type dict, got {type(cfg)}')
    return build_from_config(cfg, registry, logger=logger, *args, **kwargs)


MODELS = Registry('MODELS', build_func=build_model)
TOKENIZERS = Registry('TOKENIZER', build_func=build_model)
EMBEDDERS = Registry('EMBEDDERS', build_func=build_model)
BACKBONES = Registry('BACKBONES', build_func=build_model)
NECKS = Registry('NECKS', build_func=build_model)
HEADS = Registry('HEADS', build_func=build_model)
BRICKS = Registry('BRICKS', build_func=build_model)
STEMS = BRICKS
LOSSES = Registry('LOSSES', build_func=build_model)
TUNERS = Registry('TUNERS', build_func=build_model)

# reigister cls for diffusion.

DIFFUSIONS = Registry('DIFFUSIONS', build_func=build_diffusion)
NOISE_SCHEDULERS = Registry('NOISE_SCHEDULERS', build_func=build_diffusion)
DIFFUSION_SAMPLERS = Registry('DIFFUSION_SAMPLERS', build_func=build_diffusion_sampler)
