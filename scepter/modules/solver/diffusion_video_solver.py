# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import numpy as np
from tqdm import tqdm

from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.solver import LatentDiffusionSolver
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.distribute import we
from scepter.modules.utils.probe import ProbeData

@SOLVERS.register_class()
class LatentDiffusionVideoSolver(LatentDiffusionSolver):
    para_dict = LatentDiffusionSolver.para_dict
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.fps = cfg.get("FPS", 8)

    def save_results(self, results):
        log_data, log_label = [], []
        for result in results:
            ret_videos, ret_labels = [], []
            if 'edit_video' in result:
                ret_videos.append((result['edit_video'].permute(1, 2, 3, 0).cpu().numpy() *
                             255).astype(np.uint8))
                ret_labels.append("left: edit video")
            if 'edit_image' in result:
                ret_videos.append((result['edit_image'].permute(1, 2, 3, 0).cpu().numpy() *
                                   255).astype(np.uint8))
                ret_labels.append("left: edit image")
            if 'target_video' in result:
                if len(ret_videos) > 0:
                    ret_labels.append("middle: target video")
                else:
                    ret_labels.append("left: target video")
                ret_videos.append((result['target_video'].permute(1, 2, 3, 0).cpu().numpy() *
                                   255).astype(np.uint8))

            ret_videos.append((result['reconstruct_video'].permute(1, 2, 3, 0).cpu().numpy() *
                             255).astype(np.uint8))
            ret_labels.append("right: generation video" + " Prompt: " + result['instruction'])

            log_data.append(ret_videos)
            log_label.append(ret_labels)
        return log_data, log_label

    def run_train(self):
        self.train_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        data_iter = iter(self.datas[self._mode].dataloader)
        self.print_memory_status()
        for step in range(self.max_steps):
            if 'eval' in self._mode_set and (self.eval_interval > 0 and
                                             step % self.eval_interval == 0):
                self.run_eval()
                self.train_mode()
            batch_data = next(data_iter)
            self.before_iter(self.hooks_dict[self._mode])
            if 'meta' in batch_data and isinstance(batch_data['meta'], dict):
                self.register_probe({
                    'data_key':
                    ProbeData(batch_data['meta'].get('data_key', []),
                              view_distribute=True)
                })
            self.register_probe({
                'prompt': batch_data['prompt'],
                'batch_size': len(batch_data['prompt'])
            })
            self.current_batch_data[self.mode] = batch_data
            if self.sample_args:
                self.current_batch_data[self.mode].update(
                    self.sample_args.get_lowercase_dict())
            batch_data = transfer_data_to_cuda(batch_data)
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_train(
                    batch_data,
                    step,
                    step=self.total_iter,
                    rank=we.rank)
                self._iter_outputs[self._mode] = self._reduce_scalar(results)
            self.after_iter(self.hooks_dict[self._mode])
            if we.debug:
                self.print_trainable_params_status(prefix='model.')
            if 'eval' in self._mode_set and (self.eval_interval > 0
                                             and step == self.max_steps - 1):
                self.run_eval()
                self.train_mode()
        self.after_all_iter(self.hooks_dict[self._mode])

    @torch.no_grad()
    def run_eval(self):
        self.eval_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        all_results = []
        for batch_idx, batch_data in tqdm(
                enumerate(self.datas[self._mode].dataloader)):
            self.before_iter(self.hooks_dict[self._mode])
            if self.sample_args:
                batch_data.update(self.sample_args.get_lowercase_dict())
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_eval(transfer_data_to_cuda(batch_data),
                                             batch_idx,
                                             step=self.total_iter,
                                             rank=we.rank)
                all_results.extend(results)
            self.after_iter(self.hooks_dict[self._mode])
        log_data, log_label = self.save_results(all_results)
        self.register_probe({'eval_label': log_label})
        self.register_probe({
            'eval_video':
                ProbeData(log_data,
                          is_image=False,
                          is_video=True,
                          fps=self.fps,
                          build_html=True,
                          build_label=log_label)
        })
        self.after_all_iter(self.hooks_dict[self._mode])

    @torch.no_grad()
    def run_test(self):
        self.test_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        all_results = []
        for batch_idx, batch_data in tqdm(
                enumerate(self.datas[self._mode].dataloader)):
            self.before_iter(self.hooks_dict[self._mode])
            if self.sample_args:
                batch_data.update(self.sample_args.get_lowercase_dict())
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_eval(transfer_data_to_cuda(batch_data),
                                             batch_idx,
                                             step=self.total_iter,
                                             rank=we.rank)
                all_results.extend(results)
            self.after_iter(self.hooks_dict[self._mode])
        log_data, log_label = self.save_results(all_results)
        self.register_probe({'test_label': log_label})
        self.register_probe({
            'test_video':
                ProbeData(log_data,
                          is_image=False,
                          is_video=True,
                          fps=self.fps,
                          build_html=True,
                          build_label=log_label)
        })
        self.after_all_iter(self.hooks_dict[self._mode])

    @property
    def probe_data(self):
        if not we.debug and self.mode == 'train':
            batch_data = self.current_batch_data[self.mode]
            if self.sample_args is not None:
                batch_data.update(self.sample_args.get_lowercase_dict())
            self.eval_mode()
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                batch_data['log_train_num'] = self.log_train_num
                all_results = self.run_step_eval(transfer_data_to_cuda(batch_data))
            self.train_mode()
            log_data, log_label = self.save_results(all_results)
            self.register_probe({
                'train_video':
                    ProbeData(log_data,
                              is_image=False,
                              is_video=True,
                              fps=self.fps,
                              build_html=True,
                              build_label=log_label)
            })
            self.register_probe({'train_label': log_label})
        return super(LatentDiffusionSolver, self).probe_data

    @staticmethod
    def get_config_template():
        return dict_to_yaml('SOLVER',
                            __class__.__name__,
                            LatentDiffusionVideoSolver.para_dict,
                            set_name=True)