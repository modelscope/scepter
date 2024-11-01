# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from tqdm import tqdm

from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.distribute import we
from scepter.modules.utils.probe import ProbeData

from .diffusion_solver import LatentDiffusionSolver
from .registry import SOLVERS


@SOLVERS.register_class()
class ACESolver(LatentDiffusionSolver):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.log_train_num = cfg.get('LOG_TRAIN_NUM', -1)

    def save_results(self, results):
        log_data, log_label = [], []
        for result in results:
            ret_images, ret_labels = [], []
            edit_image = result.get('edit_image', None)
            edit_mask = result.get('edit_mask', None)
            if edit_image is not None:
                for i, edit_img in enumerate(result['edit_image']):
                    if edit_img is None:
                        continue
                    ret_images.append(
                        (edit_img.permute(1, 2, 0).cpu().numpy() * 255).astype(
                            np.uint8))
                    ret_labels.append(f'edit_image{i}; ')
                    if edit_mask is not None:
                        ret_images.append(
                            (edit_mask[i].permute(1, 2, 0).cpu().numpy() *
                             255).astype(np.uint8))
                        ret_labels.append(f'edit_mask{i}; ')

            target_image = result.get('target_image', None)
            target_mask = result.get('target_mask', None)
            if target_image is not None:
                ret_images.append(
                    (target_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        np.uint8))
                ret_labels.append('target_image; ')
                if target_mask is not None:
                    ret_images.append(
                        (target_mask.permute(1, 2, 0).cpu().numpy() *
                         255).astype(np.uint8))
                    ret_labels.append('target_mask; ')

            reconstruct_image = result.get('reconstruct_image', None)
            if reconstruct_image is not None:
                ret_images.append(
                    (reconstruct_image.permute(1, 2, 0).cpu().numpy() *
                     255).astype(np.uint8))
                ret_labels.append(f"{result['instruction']}")
            log_data.append(ret_images)
            log_label.append(ret_labels)
        return log_data, log_label

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
            'eval_image':
            ProbeData(log_data,
                      is_image=True,
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
            'test_image':
            ProbeData(log_data,
                      is_image=True,
                      build_html=True,
                      build_label=log_label)
        })

        self.after_all_iter(self.hooks_dict[self._mode])

    @property
    def probe_data(self):
        if not we.debug and self.mode == 'train':
            batch_data = transfer_data_to_cuda(
                self.current_batch_data[self.mode])
            self.eval_mode()
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                batch_data['log_num'] = self.log_train_num
                results = self.run_step_eval(batch_data)
            self.train_mode()
            log_data, log_label = self.save_results(results)
            self.register_probe({
                'train_image':
                ProbeData(log_data,
                          is_image=True,
                          build_html=True,
                          build_label=log_label)
            })
            self.register_probe({'train_label': log_label})
        return super(LatentDiffusionSolver, self).probe_data
