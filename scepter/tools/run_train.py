# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.logger import get_logger


def run_task(cfg):
    std_logger = get_logger(name='scepter')
    solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
    solver.set_up_pre()
    solver.set_up()
    solver.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')

    cfg = Config(load=True, parser_ins=parser)
    we.init_env(cfg, logger=None, fn=run_task)
