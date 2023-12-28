# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp

from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger


def run_task(cfg):
    std_logger = get_logger(name='scepter')
    solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
    solver.set_up()
    if not cfg.args.pretrained_model == '':
        with FS.get_from(cfg.args.pretrained_model,
                         wait_finish=True) as local_path:
            solver.model.load_state_dict(
                torch.load(local_path, map_location='cuda')['model'])
    solver.test_mode()
    num_samples = cfg.args.num_samples
    prompt = [cfg.args.prompt] * num_samples
    n_prompt = [cfg.args.n_prompt] * num_samples
    sampler = cfg.args.sampler
    sample_steps = cfg.args.sample_steps
    seed = cfg.args.seed
    guide_scale = cfg.args.guide_scale
    guide_rescale = cfg.args.guide_rescale
    image_size = cfg.args.image_size
    if image_size is not None:
        if ',' in image_size:
            h, w = image_size.split(',')
            image_size = [int(h), int(w)]
        else:
            image_size = [int(image_size), int(image_size)]
    
    batch_data = {}
    if solver.sample_args:
        batch_data.update(solver.sample_args.get_lowercase_dict())
    if image_size is not None:
        batch_data.update({'image_size': image_size})
    batch_data.update({
        'prompt': prompt,
        'n_prompt': n_prompt,
        'sampler': sampler,
        'sample_steps': sample_steps,
        'seed': seed,
        'guide_scale': guide_scale,
        'guide_rescale': guide_rescale,
    })
    
    dtype = getattr(torch, cfg.SOLVER.DTYPE)
    with amp.autocast(enabled=True, dtype=dtype):
        batch_data = transfer_data_to_cuda(batch_data)
        ret = solver.run_step_test(batch_data)
    save_folder = os.path.join(solver.work_dir, cfg.args.save_folder)
    for idx, out in enumerate(ret):
        img = out['image']
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        filename = '{}_{}.png'.format('inference', idx)
        save_file = os.path.join(save_folder, filename)
        with FS.put_to(save_file) as local_path:
            image = img.copy()
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
            cv2.imwrite(local_path, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument(
        '--prompt',
        dest='prompt',
        help='Prompt sentence!',
        default='a woman is walking on the street in a rainy day.')
    parser.add_argument('--n_prompt',
                        dest='n_prompt',
                        help='Add Prompt sentence!',
                        default='')
    parser.add_argument('--num_samples',
                        dest='num_samples',
                        help="Output image's number!",
                        default=4,
                        type=int)
    parser.add_argument('--sampler',
                        dest='sampler',
                        help='Sampler method!',
                        default='ddim',
                        type=str)
    parser.add_argument('--sample_steps',
                        dest='sample_steps',
                        help='Sample steps!',
                        default=50,
                        type=int)
    parser.add_argument('--seed',
                        dest='seed',
                        help='Random seed!',
                        default=2023,
                        type=int)
    parser.add_argument('--guide_scale',
                        dest='guide_scale',
                        help='Guidance scale!',
                        default=7.5,
                        type=float)
    parser.add_argument('--guide_rescale',
                        dest='guide_rescale',
                        help='Guidance rescale!',
                        default=0.5,
                        type=float)
    parser.add_argument('--image_size',
                        dest='image_size',
                        help='Output image size! (h, w)',
                        default=None,
                        type=str)
    parser.add_argument('--save_folder',
                        dest='save_folder',
                        help="Output image's save folder!",
                        default='test_images')
    parser.add_argument('--pretrained_model',
                        dest='pretrained_model',
                        help='The pretrained model for our network!',
                        default='')
    cfg = Config(load=True, parser_ins=parser)
    we.init_env(cfg, logger=None, fn=run_task)
