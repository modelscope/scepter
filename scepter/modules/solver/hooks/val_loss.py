# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from scepter.modules.data.dataset import DATASETS
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import barrier, gather_data, we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.math_plot import plot_multi_curves

_DEFAULT_VAL_PRIORITY = 200


def float_format(o):
    if isinstance(o, float):
        return f"{o: .6f}"
    raise TypeError(f"Type {type(o)} not serializable")


@HOOKS.register_class()
class ValLossHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_VAL_PRIORITY,
            'description': 'The priority for processing!'
        },
        'VAL_INTERVAL': {
            'value': 1000,
            'description': 'the interval for log print!'
        },
        'VAL_LIMITATION_SIZE': {
            'value': 1000000,
            'description': 'the limitation size for validation!'
        },
        'VAL_SEED': {
            'value': 2025,
            'description': 'the validation seed for t or generator sample!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(ValLossHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_VAL_PRIORITY)
        self.val_interval = cfg.get('VAL_INTERVAL', 1000)
        self.val_dim = cfg.get('VAL_DIM', 'all')
        self.meta_field = cfg.get('META_FIELD', ['edit_type', 'data_type'])
        self.save_folder = cfg.get('SAVE_FOLDER', 'val_loss')
        self.val_limitation_size = cfg.get('VAL_LIMITATION_SIZE', 1000000)
        self.val_seed = cfg.get('VAL_SEED', 2025)
        self.data = DATASETS.build(cfg.DATA, logger=logger)

    def before_all_iter(self, solver):
        solver.eval_mode()
        self.eval_set_size = len(self.data.dataset)
        if self.eval_set_size > self.val_limitation_size:
            self.logger.info(
                f"The samples number {self.eval_set_size} of validation set "
                f"should not great than {self.val_limitation_size}")
        assert self.eval_set_size < self.val_limitation_size
        if not hasattr(solver, 'run_step_val'):
            self.logger.info(
                f"The val-loss hook should have the function run_step_val"  # noqa
            )  # noqa
        assert hasattr(solver, 'run_step_val')
        if not self.data.batch_size == 1:
            self.logger.info(
                f"The batch_size of validation set should be 1 "  # noqa
                f"when you use the validation hook to make the results deterministic."  # noqa
            )
        assert self.data.batch_size == 1
        timestamp_generator = torch.Generator(device=we.device_id)
        timestamp_generator.manual_seed(self.val_seed)
        u = torch.rand((self.eval_set_size, ),
                       device=we.device_id,
                       generator=timestamp_generator)
        self.t = (u * (solver.timesteps - 1)).round().long()
        solver.val_interval = self.val_interval
        solver.train_mode()

    def get_val_loss(self, solver, step):
        all_loss = []
        # batch-size must be 1
        for batch_data in tqdm(self.data.dataloader):
            # generate t list
            sample_id = int(batch_data['sample_id'][0])
            meta_info = {m_f: batch_data[m_f][0] for m_f in self.meta_field}
            meta_info['sample_id'] = sample_id

            batch_data['t'] = torch.stack(
                [self.t[sample_id % self.eval_set_size]])
            noise_generator = torch.Generator(device=we.device_id)
            noise_generator.manual_seed(sample_id + 10000 * self.val_seed)
            # get generator according to the sample_id
            with torch.no_grad():
                loss = solver.run_step_val(batch_data, noise_generator)
            meta_info['loss'] = float(loss[sample_id])
            all_loss.append(meta_info)
        all_loss = json.dumps(all_loss, default=float_format)
        all_loss = gather_data([all_loss])
        if we.rank == 0:
            reduce_loss = []
            for loss in all_loss:
                reduce_loss.extend(json.loads(loss))
            compute_results = self.compute_avg_loss(reduce_loss)
            self.save_record(solver, compute_results, reduce_loss, step)
        return

    def compute_avg_loss(self, loss_list):
        all_avg_ls = []
        avg_ls = {}
        for ls in loss_list:
            for m_f in self.meta_field:
                m_f_v = ls[m_f]
                ls_key = m_f + '_' + m_f_v
                if ls_key not in avg_ls:
                    avg_ls[ls_key] = []
                avg_ls[ls_key].append(ls['loss'])
            all_avg_ls.append(ls['loss'])
        compute_results = {
            'all': sum(all_avg_ls) / len(all_avg_ls),
        }
        compute_results.update(
            {m_f: sum(avg_ls[m_f]) / len(avg_ls[m_f])
             for m_f in avg_ls})
        return compute_results

    def save_record(self, solver, compute_results, all_loss, step):
        save_folder = os.path.join(solver.work_dir, self.save_folder)
        # save history
        save_history = os.path.join(save_folder, 'history.json')

        draw_curve = False

        if FS.exists(save_history):
            results = json.loads(FS.get_object(save_history).decode())
            all_loss = {loss['sample_id']: loss for loss in all_loss}
            for loss in results['detail']:
                loss['loss'] = {int(k): v for k, v in loss['loss'].items()}
                loss['loss'][step] = all_loss[loss['sample_id']]['loss']
            for k, v in compute_results.items():
                results['summary'][k] = {
                    int(kk): vv
                    for kk, vv in results['summary'][k].items()
                }
                results['summary'][k][step] = v
            draw_curve = True
        else:
            results = {'detail': [], 'summary': {}}
            for loss in all_loss:
                loss_v = loss.pop('loss')
                loss['loss'] = {step: loss_v}
                results['detail'].append(loss)
            for k, v in compute_results.items():
                if k not in results['summary']:
                    results['summary'][k] = {}
                results['summary'][k][step] = v
        #
        FS.put_object(
            json.dumps(results, default=float_format).encode(), save_history)
        # plot current curve
        if draw_curve:
            self.plot_results(results['summary'],
                              os.path.join(save_folder, 'curve'))
        # print current log
        print_msg = ''
        for k, v in compute_results.items():
            print_msg += f"{k}: {v: .4f} "
        self.logger.info(f"Step {step} validation loss: {print_msg}")

    def plot_results(self, plot_data, save_folder):
        y = []
        steps = []
        # one image
        for label, curve_data in plot_data.items():
            curve_data = [[step, value] for step, value in curve_data.items()]
            curve_data.sort(key=lambda x: x[0])
            steps = [step for step, value in curve_data]
            value = [value for step, value in curve_data]
            k_y = [{'data': np.array(value), 'label': label}]
            save_path = os.path.join(save_folder, 'detail', f"{label}.png")
            with FS.put_to(save_path) as local_file:
                plot_multi_curves(x=np.array(steps),
                                  y=k_y,
                                  x_label='steps',
                                  y_label=None,
                                  title=f"{label}'s validation loss",
                                  save_path=local_file)
            y = y + k_y
        if len(steps) > 0:
            save_path = os.path.join(save_folder, f"summary.png")  # noqa
            with FS.put_to(save_path) as local_file:
                plot_multi_curves(
                    x=np.array(steps),
                    y=y,
                    x_label='steps',
                    y_label=None,
                    title=f"validation loss",  # noqa
                    save_path=local_file)

    def after_iter(self, solver):
        if solver.mode == 'train' and solver.total_iter % self.val_interval == 0:
            step = solver.total_iter
            solver.eval_mode()
            self.get_val_loss(solver, step)
            solver.train_mode()
            torch.cuda.synchronize()
            barrier()

    def after_all_iter(self, solver):
        if solver.mode == 'train':
            step = solver.total_iter
            solver.eval_mode()
            self.get_val_loss(solver, step)
            solver.train_mode()
            torch.cuda.synchronize()
            barrier()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            ValLossHook.para_dict,
                            set_name=True)
