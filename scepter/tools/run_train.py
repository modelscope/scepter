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


def update_config(cfg):
    if hasattr(cfg.args, 'learning_rate') and cfg.args.learning_rate:
        print(
            f'learning_rate change from {cfg.SOLVER.OPTIMIZER.LEARNING_RATE} to {cfg.args.learning_rate}'
        )
        cfg.SOLVER.OPTIMIZER.LEARNING_RATE = float(cfg.args.learning_rate)
    if hasattr(cfg.args, 'max_steps') and cfg.args.max_steps:
        print(
            f'max_steps change from {cfg.SOLVER.MAX_STEPS} to {cfg.args.max_steps}'
        )
        cfg.SOLVER.MAX_STEPS = int(cfg.args.max_steps)
    return cfg


if __name__ == '__main__':
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
