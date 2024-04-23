# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import os.path as osp
import shutil
import sys
import warnings

import torch
import torch.distributed as du
from swift import push_to_hub

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

_DEFAULT_CHECKPOINT_PRIORITY = 300


@HOOKS.register_class()
class CheckpointHook(Hook):
    """ Checkpoint resume or save hook.
    Args:
        interval (int): Save interval, by epoch.
        save_best (bool): Save the best checkpoint by a metric key, default is False.
        save_best_by (str): How to get the best the checkpoint by the metric key, default is ''.
            + means the higher the best (default).
            - means the lower the best.
            E.g. +acc@1, -err@1, acc@5(same as +acc@5)
    """
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_CHECKPOINT_PRIORITY,
            'description': 'the priority for processing!'
        },
        'INTERVAL': {
            'value': 1,
            'description': 'the interval of saving checkpoint!'
        },
        'SAVE_BEST': {
            'value': False,
            'description': 'If save the best model or not!'
        },
        'SAVE_BEST_BY': {
            'value':
            '',
            'description':
            'If save the best model, which order should be sorted, +/-!'
        },
        'DISABLE_SNAPSHOT': {
            'value':
            False,
            'description':
            'Skip to save snapshot checkpoint.'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(CheckpointHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_CHECKPOINT_PRIORITY)
        self.interval = cfg.get('INTERVAL', 1)
        self.save_name_prefix = cfg.get('SAVE_NAME_PREFIX', 'ldm_step')
        self.save_last = cfg.get('SAVE_LAST', False)
        self.save_best = cfg.get('SAVE_BEST', False)
        self.save_best_by = cfg.get('SAVE_BEST_BY', '')
        self.push_to_hub = cfg.get('PUSH_TO_HUB', False)
        self.hub_model_id = cfg.get('HUB_MODEL_ID', None)
        self.hub_private = cfg.get('HUB_PRIVATE', False)
        self.disable_save_snapshot = cfg.get('DISABLE_SNAPSHOT', False)
        self.last_ckpt = None
        if self.save_best and not self.save_best_by:
            warnings.warn(
                "CheckpointHook: Parameter 'save_best_by' is not set, turn off save_best function."
            )
            self.save_best = False
        self.higher_the_best = True
        if self.save_best:
            if self.save_best_by.startswith('+'):
                self.save_best_by = self.save_best_by[1:]
            elif self.save_best_by.startswith('-'):
                self.save_best_by = self.save_best_by[1:]
                self.higher_the_best = False
        if self.save_best and not self.save_best_by:
            warnings.warn(
                "CheckpointHook: Parameter 'save_best_by' is not valid, turn off save_best function."
            )
            self.save_best = False
        self._last_best = None if not self.save_best else (
            sys.float_info.min if self.higher_the_best else sys.float_info.max)

    def before_solve(self, solver):
        if solver.resume_from is None:
            return
        if not FS.exists(solver.resume_from):
            solver.logger.error(f'File not exists {solver.resume_from}')
            return

        with FS.get_from(solver.resume_from, wait_finish=True) as local_file:
            solver.logger.info(f'Loading checkpoint from {solver.resume_from}')
            checkpoint = torch.load(local_file,
                                    map_location=torch.device('cpu'))

        solver.load_checkpoint(checkpoint)
        if self.save_best and '_CheckpointHook_best' in checkpoint:
            self._last_best = checkpoint['_CheckpointHook_best']

    def after_iter(self, solver):
        if solver.total_iter != 0 and (
            (solver.total_iter + 1) % self.interval == 0
                or solver.total_iter == solver.max_steps - 1):
            solver.logger.info(
                f'Saving checkpoint after {solver.total_iter + 1} steps')
            if we.rank == 0:
                save_path = osp.join(
                    solver.work_dir,
                    'checkpoints/{}-{}.pth'.format(self.save_name_prefix,
                                                   solver.total_iter + 1))
                if not self.disable_save_snapshot:
                    with FS.put_to(save_path) as local_path:
                        with open(local_path, 'wb') as f:
                            checkpoint = solver.save_checkpoint()
                            torch.save(checkpoint, f)

                from swift import SwiftModel
                if isinstance(solver.model, SwiftModel):
                    save_path = osp.join(
                        solver.work_dir,
                        'checkpoints/{}-{}'.format(self.save_name_prefix,
                                                   solver.total_iter + 1))
                    local_folder, _ = FS.map_to_local(save_path)
                    solver.model.save_pretrained(local_folder)
                    FS.put_dir_from_local_dir(local_folder, save_path)
                else:
                    if hasattr(solver, 'save_pretrained'):
                        save_path = osp.join(
                            solver.work_dir, 'checkpoints/{}-{}-bin'.format(
                                self.save_name_prefix, solver.total_iter + 1))
                        local_folder, _ = FS.map_to_local(save_path)
                        FS.make_dir(local_folder)
                        ckpt, cfg = solver.save_pretrained()
                        with FS.put_to(
                                os.path.join(
                                    local_folder,
                                    'pytorch_model.bin')) as local_path:
                            with open(local_path, 'wb') as f:
                                torch.save(ckpt, f)
                        with FS.put_to(
                                os.path.join(
                                    local_folder,
                                    'configuration.json')) as local_path:
                            json.dump(cfg, open(local_path, 'w'))
                        FS.put_dir_from_local_dir(local_folder, save_path)

                if self.save_last and solver.total_iter == solver.max_steps - 1:
                    with FS.get_fs_client(save_path) as client:
                        last_path = osp.join(
                            solver.work_dir,
                            f'checkpoints/{self.save_name_prefix}-last')
                        client.make_link(last_path, save_path)
                self.last_ckpt = save_path

            torch.cuda.synchronize()
            if we.is_distributed:
                torch.distributed.barrier()

    def after_epoch(self, solver):
        if du.is_available() and du.is_initialized() and du.get_rank() != 0:
            return
        if (solver.epoch + 1) % self.interval == 0:
            solver.logger.info(
                f'Saving checkpoint after {solver.epoch} epochs')
            checkpoint = solver.save_checkpoint()
            if checkpoint is None or len(checkpoint) == 0:
                return
            cur_is_best = False
            if self.save_best:
                # Try to get current state from epoch_outputs["eval"]
                cur_state = None \
                    if self.save_best_by not in solver.epoch_outputs['eval'] \
                    else solver.epoch_outputs['eval'][self.save_best_by]
                # Try to get current state from agg_iter_outputs["eval"] if do_final_eval is False
                if cur_state is None:
                    cur_state = None \
                        if self.save_best_by not in solver.agg_iter_outputs['eval'] \
                        else solver.agg_iter_outputs['eval'][self.save_best_by]
                # Try to get current state from agg_iter_outputs["train"] if no evaluation
                if cur_state is None:
                    cur_state = None \
                        if self.save_best_by not in solver.agg_iter_outputs['train'] \
                        else solver.agg_iter_outputs['train'][self.save_best_by]
                if cur_state is not None:
                    if self.higher_the_best and cur_state > self._last_best:
                        self._last_best = cur_state
                        cur_is_best = True
                    elif not self.higher_the_best and cur_state < self._last_best:
                        self._last_best = cur_state
                        cur_is_best = True
                    checkpoint['_CheckpointHook_best'] = self._last_best
            # minus 1, means index
            save_path = osp.join(solver.work_dir,
                                 'epoch-{:05d}.pth'.format(solver.epoch))

            with FS.get_fs_client(save_path) as client:
                local_file = client.convert_to_local_path(save_path)
                with open(local_file, 'wb') as f:
                    torch.save(checkpoint, f)
                client.put_object_from_local_file(local_file, save_path)

                if cur_is_best:
                    best_path = osp.join(solver.work_dir, 'best.pth')
                    client.make_link(best_path, save_path)
            # save pretrain checkout
            if 'pre_state_dict' in checkpoint:
                save_path = osp.join(
                    solver.work_dir,
                    'epoch-{:05d}_pretrain.pth'.format(solver.epoch))
                with FS.get_fs_client(save_path) as client:
                    local_file = client.convert_to_local_path(save_path)
                    with open(local_file, 'wb') as f:
                        torch.save(checkpoint['pre_state_dict'], f)
                    client.put_object_from_local_file(local_file, save_path)

    def create_or_update_model_card(self, model, output_dir: str):
        """
        Updates or create the model card.
        """
        if not os.path.exists(os.path.join(output_dir, 'README.md')):
            lines = []
        else:
            with open(os.path.join(output_dir, 'README.md'), 'r') as f:
                lines = f.readlines()

        # write the lines back to README.md
        with open(os.path.join(output_dir, 'README.md'), 'w') as f:
            f.writelines(lines)

    def after_all_iter(self, solver):
        if we.rank == 0:
            if self.push_to_hub and self.last_ckpt:
                if os.path.isfile(self.last_ckpt):
                    base_dir = os.path.dirname(self.last_ckpt)
                    base_file = os.path.basename(self.last_ckpt)
                    save_path = os.path.join(base_dir, 'after_all_iter')
                    os.makedirs(save_path)
                    self.create_or_update_model_card(solver.model, save_path)
                    try:
                        os.link(self.last_ckpt,
                                os.path.join(save_path, base_file))
                    except OSError:
                        shutil.copyfile(self.last_ckpt,
                                        os.path.join(save_path, base_file))

                push_to_hub(repo_name=self.hub_model_id,
                            output_dir=self.last_ckpt,
                            private=self.hub_private)
                current_dir = os.path.dirname(__file__)
                base_path = os.sep.join(current_dir.split(os.sep)[:-4])
                base_path = os.path.join(base_path, 'config')
                file_name = self.hub_model_id.replace(os.sep, '_')
                content = {'name': self.hub_model_id}
                import json
                with open(os.path.join(base_path, file_name), 'w') as f:
                    json.dump(content, f)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            CheckpointHook.para_dict,
                            set_name=True)
