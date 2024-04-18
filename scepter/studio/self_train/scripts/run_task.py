# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os

import cv2
import numpy as np
import torch

from scepter.modules.solver.hooks.checkpoint import CheckpointHook
from scepter.modules.solver.hooks.data_probe import ProbeDataHook
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger


def run_task(cfg):
    # torch.cuda.set_per_process_memory_fraction(0.4, we.device_id)
    # torch.cuda.empty_cache()
    std_logger = get_logger(name='scepter')
    std_logger.info(f'Pytorch version: {torch.__version__}')
    # std_logger.info(f"Os environment: {os.environ}")
    if cfg.args.stage == 'train':
        solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
        save_config(cfg)
        solver.set_up_pre()
        solver.set_up()
        ori_steps = solver.max_steps

        if 'train' in solver.datas:
            dataset = solver.datas['train'].dataset
            if hasattr(dataset, 'real_number'):
                solver.max_steps = int(
                    cfg.SOLVER.MAX_EPOCHS * dataset.real_number /
                    (solver.datas['train'].batch_size * we.world_size))
                std_logger.info(
                    f'max step is changed from {ori_steps} to {solver.max_steps} '
                    f'according to the setting epoches {cfg.SOLVER.MAX_EPOCHS} '
                    f'and dataset size {dataset.real_number}')
                if 'train' in solver.hooks_dict:
                    for hook in solver.hooks_dict['train']:
                        if isinstance(hook, CheckpointHook):
                            ori_interval = hook.interval
                            hook.interval = int(hook.interval *
                                                solver.max_steps /
                                                cfg.SOLVER.MAX_EPOCHS)
                            std_logger.info(
                                f'checkpoint save interval is changed from {ori_interval} '
                                f'to {hook.interval} according to the setting epoches '
                                f'interval {ori_interval}')

                if 'eval' in solver.hooks_dict:
                    for hook in solver.hooks_dict['eval']:
                        if isinstance(hook, ProbeDataHook):
                            ori_interval = hook.prob_interval
                            hook.prob_interval = int(hook.prob_interval *
                                                     solver.max_steps /
                                                     cfg.SOLVER.MAX_EPOCHS)
                            std_logger.info(
                                f'prob interval is changed from {ori_interval} '
                                f'to {hook.prob_interval} according to the setting epoches '
                                f'interval {ori_interval}')
                        solver.eval_interval = hook.prob_interval

        solver.solve()


def save_image(image, save_path, backend='cv2'):
    if backend == 'cv2':
        image = image.copy()
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
        cv2.imwrite(save_path, image)


def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0],
                           x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def save_config(cfg):
    from scepter.modules.utils.distribute import get_dist_info
    rank, _ = get_dist_info()
    if rank == 0:
        config_path = os.path.join(cfg.SOLVER.WORK_DIR,
                                   cfg.args.cfg_file.split('/')[-1])
        with FS.put_to(config_path) as local_config_path:
            with open(local_config_path, 'w') as f_out:
                f_out.write(cfg.dump(is_secure=True))


def update_config(cfg):
    if cfg.args.work_dir and cfg.args.work_dir != '':
        cfg.SOLVER.WORK_DIR = cfg.args.work_dir
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Argparser for Cate process:\n')
    parser.add_argument(
        '--stage',
        dest='stage',
        help='Running stage!',
        default='train',
        choices=['train', 'inference', 'upsampler_inference', 'control'])
    parser.add_argument('--base_model',
                        dest='base_model',
                        help='Base model name!',
                        default='sd')
    parser.add_argument(
        '--prompt',
        dest='prompt',
        help='Prompt sentence!',
        default='a woman is walking on the street in a rainy day.')
    parser.add_argument('--n_prompt',
                        dest='n_prompt',
                        help='Add Prompt sentence!',
                        default='')
    parser.add_argument('--image',
                        dest='image',
                        help='Image to be upsampled!',
                        default='')
    parser.add_argument('--num_samples',
                        dest='num_samples',
                        help="Output image's number!",
                        default=4,
                        type=int)
    parser.add_argument('--sampler',
                        dest='sampler',
                        help='sampler',
                        default='ddim',
                        type=str)
    parser.add_argument('--sample_steps',
                        dest='sample_steps',
                        help='sample_steps',
                        default=50,
                        type=int)
    parser.add_argument('--inference_resolution',
                        dest='inference_resolution',
                        help='inference resolution',
                        default=1024,
                        type=int)
    parser.add_argument('--seed',
                        dest='seed',
                        help='seed',
                        default=2023,
                        type=int)
    parser.add_argument('--save_folder',
                        dest='save_folder',
                        help="Output image's save folder!",
                        default='test_images')
    parser.add_argument('--pretrained_model',
                        dest='pretrained_model',
                        help='The pretrained model for our network!',
                        default='')
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='The learning rate for our network!',
                        default=None)
    parser.add_argument('--max_steps',
                        dest='max_steps',
                        help='The max steps for our network!',
                        default=None)
    parser.add_argument('--control_mode',
                        dest='control_mode',
                        help='',
                        default=None)
    parser.add_argument('--work_dir', dest='work_dir', help='', default=None)
    cfg = Config(load=True, parser_ins=parser)
    cfg = update_config(cfg)
    we.init_env(cfg, logger=None, fn=run_task)
