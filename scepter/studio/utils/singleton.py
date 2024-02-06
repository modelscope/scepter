# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
class Singleton(object):
    @classmethod
    def get_instance(cls, cfg, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(cfg, **kwargs)
        return cls._instance
