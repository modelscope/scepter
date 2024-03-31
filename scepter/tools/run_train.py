# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import importlib
import os
import sys

from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.logger import get_logger

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)


def run_task(cfg):
    std_logger = get_logger(name='scepter')
    solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
    solver.set_up_pre()
    solver.set_up()
    solver.solve()


def update_config(cfg):
    if hasattr(cfg.args, 'learning_rate') and cfg.args.learning_rate:
        if cfg.SOLVER.OPTIMIZER.get('LEARNING_RATE', None) is not None:
            print(
                f'learning_rate change from {cfg.SOLVER.OPTIMIZER.LEARNING_RATE} to {cfg.args.learning_rate}'
            )
        cfg.SOLVER.OPTIMIZER.LEARNING_RATE = float(cfg.args.learning_rate)
    if hasattr(cfg.args, 'max_steps') and cfg.args.max_steps:
        if cfg.SOLVER.get('MAX_STEPS', None) is not None:
            print(
                f'max_steps change from {cfg.SOLVER.MAX_STEPS} to {cfg.args.max_steps}'
            )
        cfg.SOLVER.MAX_STEPS = int(cfg.args.max_steps)
    return cfg


def run():
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='The learning rate for our network!',
                        default=None)
    parser.add_argument('--max_steps',
                        dest='max_steps',
                        help='The max steps for training!',
                        default=None)

    cfg = Config(load=True, parser_ins=parser)
    cfg = update_config(cfg)
    we.init_env(cfg, logger=None, fn=run_task)


if __name__ == '__main__':
    run()
