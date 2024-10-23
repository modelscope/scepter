# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import torch
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import barrier, we
from scepter.modules.utils.file_system import FS

_DEFAULT_PROBE_PRIORITY = 1000


@HOOKS.register_class()
class ProbeDataHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_PROBE_PRIORITY,
            'description': 'The priority for processing!'
        },
        'PROB_INTERVAL': {
            'value': 1000,
            'description': 'the interval for log print!'
        },
        'SAVE_NAME_PREFIX': {
            'value': 'step',
            'description': 'the prefix for save name!'
        },
        'SAVE_PROBE_PREFIX': {
            'value': None,
            'description': 'the prefix for save probe!'
        },
        'SAVE_LAST': {
            'value': False,
            'description': 'whether to save last!'
        },
        'SAVE_IMAGE_POSTFIX': {
            'value': 'jpg',
            'description': 'the postfix for save image!'
        },
        'SAVE_VIDEO_POSTFIX': {
            'value': 'mp4',
            'description': 'the postfix for save video!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(ProbeDataHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_PROBE_PRIORITY)
        self.prob_interval = cfg.get('PROB_INTERVAL', 1000)
        self.save_name_prefix = cfg.get('SAVE_NAME_PREFIX', 'step')
        self.save_probe_prefix = cfg.get('SAVE_PROBE_PREFIX', None)
        self.save_last = cfg.get('SAVE_LAST', False)
        self.save_image_postfix = cfg.get('SAVE_IMAGE_POSTFIX', 'jpg')
        self.save_video_postfix = cfg.get('SAVE_VIDEO_POSTFIX', 'mp4')

    def before_all_iter(self, solver):
        if not solver.mode == 'train' and hasattr(solver, 'eval_interval'):
            solver.eval_interval = self.prob_interval
    def before_iter(self, solver):
        pass

    def get_key_level_prefix(self, key, save_folder, total_iter):
        if self.save_probe_prefix is not None:
            ret_prefix = os.path.join(save_folder,
                                      self.save_probe_prefix)
        else:
            ret_prefix = os.path.join(
                save_folder,
                key.replace('/', '_') + f'_step_{total_iter}')
        return ret_prefix
    def after_iter(self, solver):
        if solver.mode == 'train' and solver.total_iter % self.prob_interval == 0:
            save_folder = os.path.join(
                solver.work_dir,
                f'{solver.mode}_probe/{self.save_name_prefix}-{solver.total_iter}'
            )
            setattr(solver,
                    f'{solver.mode}_pre_save_paras',
                    {
                         "save_folder":save_folder,
                         "save_probe_prefix": self.save_probe_prefix,
                         "image_postfix": self.save_image_postfix,
                         "video_postfix": self.save_video_postfix,
                         "step": solver.total_iter,
                    }
                )
            gather_probe_dict = solver.probe_data
            if we.rank == 0:
                ret_data = {}
                for k, v in gather_probe_dict.items():
                    ret_prefix = self.get_key_level_prefix(k, save_folder, solver.total_iter)
                    ret_one = v.to_log(ret_prefix, image_postfix=self.save_image_postfix,
                                       video_postfix=self.save_video_postfix)
                    if (isinstance(ret_one, list)
                            or isinstance(ret_one, dict)) and len(ret_one) < 1:
                        continue
                    ret_data[k] = ret_one
                with FS.put_to(os.path.join(save_folder,
                                            'meta.json')) as local_path:
                    json.dump(ret_data,
                              open(local_path, 'w'),
                              ensure_ascii=False)
            solver.clear_probe()
            torch.cuda.synchronize()
            barrier()

    def after_all_iter(self, solver):
        if not solver.mode == 'train':
            step = solver._total_iter[
                'train'] if 'train' in solver._total_iter else 0
            save_folder = os.path.join(
                solver.work_dir,
                f'{solver.mode}_probe/{self.save_name_prefix}-{step}'
            )
            setattr(solver,
                    f'{solver.mode}_pre_save_paras',
                    {
                        "save_folder": save_folder,
                        "save_probe_prefix": self.save_probe_prefix,
                        "image_postfix": self.save_image_postfix,
                        "step": step,
                        "video_postfix": self.save_video_postfix
                    }
                    )
            probe_dict = solver.probe_data
            if we.rank == 0:
                ret_data = {}
                for k, v in probe_dict.items():
                    ret_prefix = self.get_key_level_prefix(k, save_folder, step)
                    ret_one = v.to_log(ret_prefix, image_postfix=self.save_image_postfix,
                                       video_postfix=self.save_video_postfix)
                    if (isinstance(ret_one, list)
                            or isinstance(ret_one, dict)) and len(ret_one) < 1:
                        continue
                    ret_data[k] = ret_one
                with FS.put_to(os.path.join(save_folder,
                                            'meta.json')) as local_path:
                    json.dump(ret_data,
                              open(local_path, 'w'),
                              ensure_ascii=False)

                if self.save_last and step == solver.max_steps:
                    with FS.get_fs_client(save_folder) as client:
                        last_save_folder = os.path.join(
                            solver.work_dir,
                            f'{solver.mode}_probe/{self.save_name_prefix}-last'
                        )
                        print(last_save_folder, save_folder)
                        client.make_link(last_save_folder, save_folder)

            solver.clear_probe()
            torch.cuda.synchronize()
            barrier()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            ProbeDataHook.para_dict,
                            set_name=True)
