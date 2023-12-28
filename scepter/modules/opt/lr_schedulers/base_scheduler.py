# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.


class BaseScheduler():
    def __init__(self, cfg, logger=None):
        self.logger = logger

    def __call__(self, parameters):
        return self
