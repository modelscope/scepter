# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import importlib
import os
import sys

import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision.transforms as TT
from PIL import Image

from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
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
        img = Image.fromarray((img * 255).astype(np.uint8))
        filename = '{}_{}.png'.format('inference', idx)
        save_file = os.path.join(save_folder, filename)
        with FS.put_to(save_file) as local_path:
            img.save(local_path)
            std_logger.info(f'Processed {filename} save to {local_path}')


def run_task_control(cfg):
    from scepter.modules.annotator.utils import AnnotatorProcessor

    std_logger = get_logger(name='scepter')
    solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
    solver.set_up()
    if not cfg.args.pretrained_model == '':
        with FS.get_from(cfg.args.pretrained_model,
                         wait_finish=True) as local_path:
            state = torch.load(local_path, map_location='cuda')
            state = state['model'] if 'model' in state else state
            missing, unexpected = solver.model.model.control_blocks[
                0].load_state_dict(state, strict=False)
            if we.rank == 0:
                std_logger.info(
                    f'Restored from {cfg.args.pretrained_model} with '
                    f'{len(missing)} missing and {len(unexpected)} unexpected keys'
                )

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
            image_size = int(image_size)

    with FS.get_from(cfg.args.image, wait_finish=True) as local_path:
        image = Image.open(local_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = TT.CenterCrop(image_size)(TT.Resize(image_size)(image))

    if cfg.args.control_mode != 'source':
        anno_processor = AnnotatorProcessor(anno_type=cfg.args.control_mode)
        hint = anno_processor.run(image, cfg.args.control_mode)
    else:
        hint = image
    hint = TT.ToTensor()(hint)[None, ...].repeat(num_samples, 1, 1,
                                                 1).to(we.device_id)

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
        'hint': hint
    })

    dtype = getattr(torch, cfg.SOLVER.DTYPE)
    with amp.autocast(enabled=True, dtype=dtype):
        batch_data = transfer_data_to_cuda(batch_data)
        ret = solver.run_step_test(batch_data)
    save_folder = os.path.join(solver.work_dir, cfg.args.save_folder)
    for idx, out in enumerate(ret):
        for name in ['image', 'hint']:
            img = out[name]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
            filename = '{}_{}_{}.png'.format('inference', name, idx)
            save_file = os.path.join(save_folder, filename)
            with FS.put_to(save_file) as local_path:
                img.save(local_path)
                std_logger.info(f'Processed {filename} save to {local_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--task',
                        dest='task',
                        help='Running task!',
                        default='t2i',
                        choices=['t2i', 'control'])
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
    parser.add_argument('--image',
                        dest='image',
                        help='For image-guided task (control, upsample)',
                        default='')
    parser.add_argument('--control_mode',
                        dest='control_mode',
                        help='For controllable image synthesis task',
                        choices=['source', 'canny', 'pose'],
                        default=None)
    cfg = Config(load=True, parser_ins=parser)
    if cfg.args.task == 'control':
        task_fn = run_task_control
    else:
        task_fn = run_task
    we.init_env(cfg, logger=None, fn=task_fn)
