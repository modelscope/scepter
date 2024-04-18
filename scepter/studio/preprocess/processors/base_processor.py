# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch

from scepter.studio.utils.env import get_available_memory


class BaseCaptionProcessor(object):
    def __init__(self, cfg, language='en'):
        self.use_device = cfg.get('DEVICE', 'cpu')
        self.use_memory = cfg.get('MEMORY', 10)
        self.language = language
        self.system_paras = cfg.get('PARAS', [])
        self.language_level_paras = {}
        for sys_para in self.system_paras:
            if language == 'en':
                cur_lang = sys_para.get('LANGUAGE_NAME', None)
            else:
                cur_lang = sys_para.get('LANGUAGE_ZH_NAME', None)
            if cur_lang is not None:
                self.language_level_paras[cur_lang] = sys_para

    def unload_model(self):
        mem = get_available_memory()
        free_mem = int(mem['available'] / (1024**2))
        total_mem = int(mem['total'] / (1024**2))
        self.delete_instance = False
        if free_mem < 0.5 * total_mem:
            self.delete_instance = True
        return True, ''

    def load_model(self):
        is_flg, msg = self.check_memory()
        return is_flg, msg

    @property
    def get_language_choice(self):
        language_choices = list(self.language_level_paras.keys())
        return language_choices

    @property
    def get_language_default(self):
        language_choices = list(self.language_level_paras.keys())
        return language_choices[0] if len(language_choices) > 0 else None

    def get_para_by_language(self, language):
        return self.language_level_paras.get(language, {})

    def check_memory(self):
        mem_msg = ''
        if self.use_device == 'gpu':
            # Check Cuda Memory
            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                    free_mem = int(free_mem / (1024**2))
                    total_mem = int(total_mem / (1024**2))
                    if free_mem < self.use_memory:
                        mem_msg += (
                            f'Needed {self.use_memory}M, but free mem '
                            f'is {free_mem:.3f}M(total is {total_mem})M \n')
            else:
                mem_msg += 'Needed GPU device, but this device is not available!'
        elif self.use_device == 'cpu':
            mem = get_available_memory()
            free_mem = int(mem['available'] / (1024**2))
            total_mem = int(mem['total'] / (1024**2))
            if free_mem < self.use_memory:
                mem_msg += (f'Needed {self.use_memory}M, but free mem '
                            f'is {free_mem:.3f}M(total is {total_mem})M \n')
        if mem_msg == '':
            return True, mem_msg
        return False, mem_msg

    def __call__(self, image, **kwargs):
        raise NotImplementedError


class BaseImageProcessor(object):
    def __init__(self, cfg, language='en'):
        self.use_device = cfg.get('DEVICE', 'cpu')
        self.use_memory = cfg.get('MEMORY', 10)
        self.language = language
        self.system_paras = cfg.get('PARAS', {})
        self.language_level_paras = {}

    def unload_model(self):
        mem = get_available_memory()
        free_mem = int(mem['available'] / (1024**2))
        total_mem = int(mem['total'] / (1024**2))
        self.delete_instance = False
        if free_mem < 0.5 * total_mem:
            self.delete_instance = True
        return True, ''

    @property
    def system_para(self):
        return self.system_paras

    def load_model(self):
        is_flg, msg = self.check_memory()
        return is_flg, msg

    def check_memory(self):
        mem_msg = ''
        if self.use_device == 'gpu':
            # Check Cuda Memory
            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
                    free_mem = int(free_mem / (1024**2))
                    total_mem = int(total_mem / (1024**2))
                    if free_mem < self.use_memory:
                        mem_msg += (
                            f'Needed {self.use_memory}M, but free mem '
                            f'is {free_mem:.3f}M(total is {total_mem})M \n')
            else:
                mem_msg += 'Needed GPU device, but this device is not available!'
        elif self.use_device == 'cpu':
            mem = get_available_memory()
            free_mem = int(mem['available'] / (1024**2))
            total_mem = int(mem['total'] / (1024**2))
            if free_mem < self.use_memory:
                mem_msg += (f'Needed {self.use_memory}M, but free mem '
                            f'is {free_mem:.3f}M(total is {total_mem})M \n')
        if mem_msg == '':
            return True, mem_msg
        return False, mem_msg

    def __call__(self, image, **kwargs):
        raise NotImplementedError
