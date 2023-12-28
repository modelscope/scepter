# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from scepter.modules.utils.config import Config
from scepter.modules.utils.registry import Registry, build_from_config


def build_pipeline(pipeline, registry, logger=None, *args, **kwargs):

    if isinstance(pipeline, list):
        if len(pipeline) == 0:
            return build_from_config(Config(cfg_dict={'NAME': 'Identity'},
                                            load=False),
                                     registry,
                                     logger=logger,
                                     *args,
                                     **kwargs)
        elif len(pipeline) == 1:
            return build_pipeline(pipeline[0], registry, logger, *args,
                                  **kwargs)
        else:
            return build_from_config(Config(cfg_dict={
                'NAME': 'Compose',
                'TRANSFORMS': pipeline
            },
                                            load=False),
                                     registry,
                                     logger=logger,
                                     *args,
                                     **kwargs)
    elif isinstance(pipeline, Config):
        return build_from_config(pipeline,
                                 registry,
                                 logger=logger,
                                 *args,
                                 **kwargs)
    elif pipeline is None:
        return build_from_config(Config(cfg_dict={'NAME': 'Identity'},
                                        load=False),
                                 registry,
                                 logger=logger,
                                 *args,
                                 **kwargs)
    else:
        raise TypeError(
            f'Expect pipeline_cfg to be dict or list or None, got {type(pipeline)}'
        )


TRANSFORMS = Registry('TRANSFORMS', build_func=build_pipeline)
