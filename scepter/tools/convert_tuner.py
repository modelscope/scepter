# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import importlib
import json
import os
import sys

import safetensors
import torch
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.module_transform import (
    convert_ldm_clip_checkpoint_v1, convert_ldm_unet_tuner_checkpoint,
    convert_lora_checkpoint, convert_tuner_civitai_to_scepter,
    create_unet_diffusers_config)

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def scepter_to_civitai(cfg):
    scepter_checkpoint = {}
    local_source, _ = FS.map_to_local(cfg.args.source)
    if not FS.exists(local_source):
        FS.get_dir_to_local_dir(cfg.args.source, local_source)
    for name in FS.walk_dir(local_source):
        path = os.path.join(local_source, name)
        if not FS.isdir(path):
            continue
        for sub_name in FS.walk_dir(path):
            if '.bin' in sub_name:
                checkpoint_path = os.path.join(local_source, name, sub_name)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                scepter_checkpoint.update(checkpoint)
    unet_config = create_unet_diffusers_config(v2=False)
    ckpt_unet = convert_ldm_unet_tuner_checkpoint(
        v2=False,
        checkpoint=scepter_checkpoint,
        config=unet_config,
        unet_key='model.')
    lora_state_dict = convert_lora_checkpoint(ckpt_unet=ckpt_unet)
    ckpt_te = convert_ldm_clip_checkpoint_v1(scepter_checkpoint)
    lora_te_state_dict = convert_lora_checkpoint(ckpt_text=ckpt_te)
    lora_state_dict.update(lora_te_state_dict)
    with FS.put_to(cfg.args.target) as local_file:
        safetensors.torch.save_file(lora_state_dict, local_file)
    if not FS.exists(cfg.args.target):
        raise Exception(
            f'Transform Error From {cfg.args.source} To {cfg.args.target}.')
    else:
        print(
            f'Transform Success From {cfg.args.source} To {cfg.args.target}.')


def civitai_to_scepter(cfg):
    civitai_lora = {}
    local_source, _ = FS.map_to_local(cfg.args.source)
    if not FS.exists(local_source):
        FS.get_from(cfg.args.source, local_source)
    with safetensors.safe_open(local_source, framework='pt',
                               device='cpu') as f:
        for k in f.keys():
            civitai_lora[k] = f.get_tensor(k)
    tuner_config, scepter_tuner, unload_params = convert_tuner_civitai_to_scepter(
        civitai_lora)
    local_target_dir, _ = FS.map_to_local(cfg.args.target)
    FS.make_dir(local_target_dir)
    config_path = os.path.join(local_target_dir, '0_SwiftLoRA',
                               'adapter_config.json')
    module_path = os.path.join(local_target_dir, '0_SwiftLoRA',
                               'adapter_model.bin')
    with FS.put_to(config_path) as local_config:
        with open(local_config, 'w') as fw:
            fw.write(json.dumps(tuner_config))
    with FS.put_to(module_path) as local_module:
        torch.save(scepter_tuner, local_module)
    if not FS.exists(cfg.args.target):
        raise Exception(
            f'Transform Error From {cfg.args.source} To {cfg.args.target}.')
    else:
        print(
            f'Transform Success From {cfg.args.source} To {cfg.args.target}.')


def transform_module_format(cfg):
    if 'FILE_SYSTEM' in cfg:
        if isinstance(cfg.FILE_SYSTEM, list):
            for file_cfg in cfg.FILE_SYSTEM:
                FS.init_fs_client(file_cfg)
        else:
            FS.init_fs_client(cfg.FILE_SYSTEM)
    if cfg.args.export:
        scepter_to_civitai(cfg)
    else:
        civitai_to_scepter(cfg)


def run():
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--source',
                        dest='source',
                        help='The source model path!',
                        default=None)
    parser.add_argument('--target',
                        dest='target',
                        help='The target model path!',
                        default=None)
    parser.add_argument('--export',
                        dest='export',
                        type=str2bool,
                        help='Use export mode',
                        default=True)

    cfg = Config(load=True, parser_ins=parser)
    transform_module_format(cfg)


if __name__ == '__main__':
    run()
