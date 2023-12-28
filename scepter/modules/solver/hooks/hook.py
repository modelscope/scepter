# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABCMeta


class Hook(object, metaclass=ABCMeta):
    def __init__(self, cfg, logger=None):
        self.logger = logger

    def before_solve(self, solver):
        pass

    def after_solve(self, solver):
        pass

    def before_epoch(self, solver):
        pass

    def after_epoch(self, solver):
        pass

    def before_all_iter(self, solver):
        pass

    def before_iter(self, solver):
        pass

    def after_iter(self, solver):
        pass

    def after_all_iter(self, solver):
        pass
