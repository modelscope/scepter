# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import json
import os
import queue
import time

import gradio as gr
import yaml

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.self_train.self_train_ui.component_names import ModelUIName
from scepter.studio.self_train.utils.config_parser import get_base_model_list
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄
delete_symbol = '\U0001f5d1'  # 🗑️
add_symbol = '\U00002795'  # ➕
confirm_symbol = '\U00002714'  # ✔️


class ModelUI(UIBase):
    def __init__(self, cfg, all_cfg_value, is_debug=False, language='en'):
        self.BASE_CFG_VALUE = all_cfg_value
        self.language = language
        self.base_model_info = get_base_model_list(self.BASE_CFG_VALUE)
        self.work_dir, _ = FS.map_to_local(cfg.WORK_DIR)
        os.makedirs(self.work_dir, exist_ok=True)
        self.default_model_name = 'step-last'
        self.old_default_model_name = 'checkpoint.pth'
        self.delete_folder_queue = queue.Queue()
        self.model_list = []
        # self.model_list.extend(self.base_model_info.get('model_choices', []))
        have_model_list = []
        if not self.work_dir.endswith('/'):
            self.work_dir += '/'
        for one_dir in FS.walk_dir(self.work_dir):
            if one_dir.startswith(self.work_dir):
                one_dir = one_dir[len(self.work_dir):]
            if not os.path.isdir(os.path.join(self.work_dir, one_dir)):
                continue
            if len(one_dir.split('/')) > 1:
                continue
            if ('@' in one_dir
                    and (os.path.exists(
                        os.path.join(self.work_dir, one_dir, 'checkpoints',
                                     self.default_model_name))
                         or os.path.exists(
                             os.path.join(self.work_dir, one_dir,
                                          self.old_default_model_name)))):
                if len(one_dir.split('@')) > 4:
                    have_model_list.append([one_dir, one_dir.split('@')[-1]])
        have_model_list.sort(key=lambda x: -int(x[-1][:-3]))
        self.model_list.extend([v[0].split('/')[-1]
                                for v in have_model_list][:50])
        self.is_debug = is_debug
        self.component_names = ModelUIName(language)

    def get_ckpt_list(self, output_model):
        all_ckpt_list = []
        if output_model is None or output_model == '':
            return all_ckpt_list
        output_model_path = os.path.join(self.work_dir, output_model)
        all_ckpt_path = os.path.join(output_model_path, 'checkpoints')
        if os.path.exists(all_ckpt_path):
            for name in os.listdir(all_ckpt_path):
                if name == self.default_model_name:
                    continue
                path = os.path.join(all_ckpt_path, name)
                if os.path.isdir(path):
                    all_ckpt_list.append(name)
        all_ckpt_list = sorted(all_ckpt_list,
                               key=lambda x: int(x.split('-')[-1]))
        if os.path.exists(os.path.join(all_ckpt_path,
                                       self.default_model_name)):
            all_ckpt_list.append(self.default_model_name)
        return all_ckpt_list

    def get_gallery_list(self, model_name, ckpt_name):
        ckpt_probe_dir = os.path.join(self.work_dir, model_name, 'eval_probe',
                                      ckpt_name, 'image')
        all_gallery_list = []
        if os.path.exists(ckpt_probe_dir):
            for name in os.listdir(ckpt_probe_dir):
                path = os.path.join(ckpt_probe_dir, name)
                all_gallery_list.append(path)
        return all_gallery_list

    def create_ui(self, *args, **kwargs):
        with gr.Box():
            gr.Markdown(self.component_names.output_model_block)
            with gr.Row(variant='panel', equal_height=True):
                with gr.Column(scale=7, min_width=0):
                    self.output_model_name = gr.Dropdown(
                        label=self.component_names.output_model_name,
                        choices=self.model_list,
                        value=self.base_model_info.get('model_default', ''),
                        show_label=False,
                        container=False,
                        interactive=True)
                with gr.Column(scale=2, min_width=0):
                    self.output_ckpt_name = gr.Dropdown(
                        label=self.component_names.output_ckpt_name,
                        value='',
                        choices=[],
                        show_label=False,
                        container=False,
                        interactive=True)
                with gr.Column(scale=1, min_width=0):
                    self.refresh_model_gbtn = gr.Button(value=refresh_symbol)
                with gr.Column(scale=1, min_width=0):
                    self.add_model_gbtn = gr.Button(value=add_symbol)
                with gr.Column(scale=1, min_width=0):
                    self.delete_model_gbtn = gr.Button(value=delete_symbol)

            with gr.Row(variant='panel', equal_height=True,
                        visible=False) as self.extra_model_panel:
                with gr.Column(scale=4, min_width=0):
                    self.extra_model_txt = gr.Text(
                        label=self.component_names.extra_model_gtxt,
                        show_label=False,
                        container=False,
                        placeholder='Add Extra Model')
                with gr.Column(scale=1, min_width=0):
                    self.confirm_add = gr.Button(value=confirm_symbol)
            with gr.Row(variant='panel', equal_height=True):
                with gr.Column(scale=3, min_width=0):
                    with gr.Box():
                        self.log_message = gr.Text(
                            placeholder='Please select model'
                            'or press export button.',
                            autoscroll=True,
                            lines=10,
                            label=self.component_names.log_block)
                with gr.Column(scale=1, min_width=0,
                               visible=False) as self.export_log_panel:
                    # with gr.Column(scale=1, min_width=0):
                    self.export_log = gr.Button(
                        value=self.component_names.btn_export_log)
                    # with gr.Column(scale=1, min_width=0):
                    self.export_url = gr.File(
                        label=self.component_names.export_file,
                        visible=False,
                        value=None,
                        interactive=False,
                        show_label=True)

            with gr.Row(variant='panel', equal_height=True):
                with gr.Accordion(label=self.component_names.gallery_block,
                                  open=True):
                    self.eval_gallery = gr.Gallery(
                        label=self.component_names.eval_gallery,
                        value=[],
                        preview=True,
                        selected_index=None)

            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    self.go_to_inferece_btn = gr.Button(
                        self.component_names.go_to_inference)

    def set_callbacks(self, trainer_ui, manager):
        self.manager = manager

        def model_name_change(model_name):
            if model_name is None:
                return '', gr.Column(), '', []
            message = trainer_ui.trainer_ins.get_log(model_name)
            status = trainer_ui.trainer_ins.get_status(model_name)
            ckpt_list = self.get_ckpt_list(model_name)
            ckpt_value = ckpt_list[-1] if len(ckpt_list) > 0 else ''
            if ckpt_value is not None and len(ckpt_value) > 0:
                gallery_value = self.get_gallery_list(model_name, ckpt_value)
            else:
                gallery_value = []
            select_index = 0 if len(gallery_value) > 0 else None
            return (message, gr.Column(visible=status in ('running',
                                                          'success')),
                    gr.Dropdown(choices=ckpt_list, value=ckpt_value),
                    gr.Gallery(value=gallery_value,
                               preview=True,
                               selected_index=select_index))

        self.output_model_name.change(fn=model_name_change,
                                      inputs=[self.output_model_name],
                                      outputs=[
                                          self.log_message,
                                          self.export_log_panel,
                                          self.output_ckpt_name,
                                          self.eval_gallery
                                      ],
                                      queue=False)

        def ckpt_name_change(model_name, ckpt_value):
            if ckpt_value is not None and len(ckpt_value) > 0:
                gallery_value = self.get_gallery_list(model_name, ckpt_value)
            else:
                gallery_value = []
            if len(gallery_value) > 0:
                return gr.Gallery(value=gallery_value,
                                  preview=True,
                                  selected_index=0)
            else:
                return gr.Gallery(value=gallery_value,
                                  preview=True,
                                  selected_index=None)

        self.output_ckpt_name.change(
            fn=ckpt_name_change,
            inputs=[self.output_model_name, self.output_ckpt_name],
            outputs=[self.eval_gallery],
            queue=False)

        def add_model():
            return gr.Row(visible=True)

        self.add_model_gbtn.click(fn=add_model,
                                  inputs=[],
                                  outputs=[self.extra_model_panel],
                                  queue=False)

        def confirm_add(model_name):
            model_folder = os.path.join(self.work_dir, model_name)
            have_model = os.path.exists(model_folder)
            if not have_model:
                gr.Error(self.component_names.model_err5.format(model_name))
            if model_name not in self.model_list and have_model:
                self.model_list.append(model_name)
            return gr.Row(visible=False), gr.Dropdown(choices=self.model_list,
                                                      value=model_name)

        self.confirm_add.click(
            fn=confirm_add,
            inputs=[self.extra_model_txt],
            outputs=[self.extra_model_panel, self.output_model_name],
            queue=False)

        def refresh_model(model_name):
            if not self.delete_folder_queue.empty():
                try:
                    del_folder = self.delete_folder_queue.get_nowait()
                    if os.path.exists(del_folder):
                        os.system(f'rm -rf {del_folder}')
                        self.delete_folder_queue.put_nowait(del_folder)
                except Exception:
                    pass

            message = trainer_ui.trainer_ins.get_log(model_name)
            status = trainer_ui.trainer_ins.get_status(model_name)
            ckpt_list = self.get_ckpt_list(model_name)
            ckpt_value = ckpt_list[-1] if len(ckpt_list) > 0 else ''
            ret_gallery = ckpt_name_change(model_name, ckpt_value)
            return (message, gr.Column(visible=status in ('running',
                                                          'success')),
                    gr.Dropdown(choices=self.model_list, value=model_name),
                    gr.Dropdown(choices=ckpt_list,
                                value=ckpt_value), ret_gallery)

        self.refresh_model_gbtn.click(fn=refresh_model,
                                      inputs=[self.output_model_name],
                                      outputs=[
                                          self.log_message,
                                          self.export_log_panel,
                                          self.output_model_name,
                                          self.output_ckpt_name,
                                          self.eval_gallery
                                      ],
                                      queue=False)

        def delete_model(model_name):
            index = 0
            trainer_ui.trainer_ins.stop_task(model_name)
            if model_name in self.model_list:
                index = self.model_list.index(model_name)
                self.model_list.remove(model_name)
                folder = os.path.join(self.work_dir, model_name)
                for _ in range(1):
                    if os.path.exists(folder):
                        try:
                            os.system(f'rm -rf {folder}')
                        except Exception:
                            time.sleep(2)
                    else:
                        break
                self.delete_folder_queue.put_nowait(folder)
            if index <= len(self.model_list) - 1:
                model_name = self.model_list[index]
            elif len(self.model_list) > 0:
                model_name = self.model_list[0]
            else:
                model_name = None

            return gr.Dropdown(choices=self.model_list, value=model_name)

        self.delete_model_gbtn.click(fn=delete_model,
                                     inputs=[self.output_model_name],
                                     outputs=[self.output_model_name])

        def go_to_inferece(output_model, output_ckpt_name):
            params_path = os.path.join(self.work_dir, output_model,
                                       'params.json')
            if os.path.exists(params_path):
                params_info = json.loads(open(params_path).read())
                assert params_info['work_name'] == output_model
                base_model = params_info['base_model']
                base_model_revision = params_info['base_model_revision']
                tuner_name = params_info['tuner_name']
                model_path = os.path.join(self.work_dir, output_model,
                                          'checkpoints', output_ckpt_name)
                eval_prompts = params_info[
                    'eval_prompts'] if 'eval_prompts' in params_info else []
                image_dir = os.path.join(self.work_dir, output_model,
                                         'eval_probe', output_ckpt_name,
                                         'image')
                if os.path.exists(image_dir):
                    image_path = [
                        os.path.join(image_dir, name)
                        for name in os.listdir(image_dir)
                    ]
                else:
                    image_path = []
            else:
                _, base_model, base_model_revision, tuner_name, _ = output_model.split(
                    '@', 4)
                model_path = os.path.join(self.work_dir, output_model,
                                          self.old_default_model_name)
                output_ckpt_name = self.old_default_model_name
                image_path = []
                eval_prompts = []
                params_info = {}

            if isinstance(image_path, list):
                image_path = image_path[0] if len(image_path) > 0 else None
            if isinstance(eval_prompts, list):
                eval_prompts = eval_prompts[0] if len(eval_prompts) > 0 else ''
            cfg_file = os.path.join(self.work_dir, output_model,
                                    f'meta_{output_ckpt_name}.yaml')
            output_model = output_model + '@' + output_ckpt_name
            tuner_dict = {
                'NAME':
                output_model,
                'NAME_ZH':
                output_model,
                # 'BASE_MODEL': base_model,
                'BASE_MODEL':
                base_model_revision,
                'TUNER_TYPE':
                tuner_name,
                'DESCRIPTION':
                '',
                'MODEL_PATH':
                model_path,
                'IMAGE_PATH':
                image_path,
                'PROMPT_EXAMPLE':
                eval_prompts,
                'SOURCE':
                'self_train',
                'CKPT_NAME':
                output_ckpt_name,
                'PARAMS':
                params_info,
                'IS_SHARE':
                self.BASE_CFG_VALUE[base_model][base_model_revision]
                ['is_share']
            }
            tuner_cfg = Config(cfg_dict=tuner_dict, load=False)

            if not os.path.exists(model_path):
                gr.Error(self.component_names.model_err4)
            self.manager.inference.model_manage_ui.pipe_manager.register_tuner(
                tuner_cfg,
                name=tuner_cfg.NAME_ZH
                if self.language == 'zh' else tuner_cfg.NAME,
                is_customized=True)

            pipeline_level_modules = self.manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if tuner_cfg.BASE_MODEL not in pipeline_level_modules:
                gr.Error(self.component_names.model_err3 +
                         tuner_cfg.BASE_MODEL)
            pipeline_ins = pipeline_level_modules[tuner_cfg.BASE_MODEL]
            diffusion_model = f"{tuner_cfg.BASE_MODEL}_{pipeline_ins.diffusion_model['name']}"

            default_choices = self.manager.inference.model_manage_ui.pipe_manager.module_level_choices
            if 'customized_tuners' in default_choices:
                if tuner_cfg.BASE_MODEL not in default_choices[
                        'customized_tuners']:
                    default_choices['customized_tuners'] = {}
                tunner_choices = default_choices['customized_tuners'][
                    tuner_cfg.BASE_MODEL]['choices']
                tunner_default = default_choices['customized_tuners'][
                    tuner_cfg.BASE_MODEL]['default']
                if not isinstance(tunner_default, list):
                    tunner_default = [tunner_default]
            else:
                tunner_choices = []
                tunner_default = []

            with open(cfg_file, 'w') as f_out:
                yaml.dump(copy.deepcopy(tuner_cfg.cfg_dict),
                          f_out,
                          encoding='utf-8',
                          allow_unicode=True,
                          default_flow_style=False)

            # example_image = tuner_cfg.get('IMAGE_PATH', None)
            # if isinstance(example_image, list) and len(example_image)>0:
            #     example_image = example_image[0]
            #
            # prompt_example = tuner_cfg.get('PROMPT_EXAMPLE', None)
            # if isinstance(prompt_example, list) and len(prompt_example)>0:
            #     prompt_example = prompt_example[0]

            base_model = tuner_cfg.get('BASE_MODEL', '')

            if not base_model == '':
                if base_model not in manager.inference.tuner_ui.name_level_tuners:
                    manager.inference.tuner_ui.name_level_tuners[
                        base_model] = {}
                manager.inference.tuner_ui.name_level_tuners[base_model][
                    output_model] = tuner_cfg

            return (
                gr.Tabs(selected='inference'), cfg_file,
                gr.Tabs(selected='tuner_ui'),
                gr.CheckboxGroup(
                    value='使用微调' if self.language == 'zh' else 'Use Tuners'),
                gr.Dropdown(value=diffusion_model),
                gr.Dropdown(choices=tunner_choices, value=tunner_default)
                # gr.Text(value=tuner_cfg.get('TUNER_TYPE', '')),
                # gr.Text(value=tuner_cfg.get('BASE_MODEL', '')),
                # gr.Image(value=example_image),
                # gr.Text(value=tuner_cfg.get('DESCRIPTION', '')),
                # gr.Text(value=prompt_example)
            )

        self.go_to_inferece_btn.click(
            go_to_inferece,
            inputs=[self.output_model_name, self.output_ckpt_name],
            outputs=[
                manager.tabs, manager.inference.infer_info,
                manager.inference.setting_tab,
                manager.inference.check_box_for_setting,
                manager.inference.model_manage_ui.diffusion_model,
                manager.inference.tuner_ui.custom_tuner_model
                # manager.inference.tuner_ui.tuner_type,
                # manager.inference.tuner_ui.base_model,
                # manager.inference.tuner_ui.tuner_example,
                # manager.inference.tuner_ui.tuner_desc,
                # manager.inference.tuner_ui.tuner_prompt_example
            ],
            queue=True)

        def export_train_log(model_name):
            current_log_folder = os.path.join(self.work_dir, model_name)
            _ = trainer_ui.trainer_ins.get_log(model_name)
            out_log = os.path.join(current_log_folder, 'output_std_log.txt')
            if os.path.exists(out_log):
                zip_path = f'{trainer_ui.current_train_model}_log.zip'
                cache_folder = f'{trainer_ui.current_train_model}_log'
                if not os.path.exists(
                        os.path.join(current_log_folder, 'tensorboard')):
                    os.makedirs(os.path.join(current_log_folder,
                                             'tensorboard'),
                                exist_ok=True)
                res = os.popen(
                    f"cd {current_log_folder} && mkdir -p '{cache_folder}' "
                    f"&& cp -rf tensorboard '{cache_folder}/tensorboard' "
                    f"&& cp -rf 'output_std_log.txt' '{cache_folder}/output_std_log.txt' "
                    f"&& zip -r '{zip_path}' '{cache_folder}'/* "
                    f"&& rm -rf '{cache_folder}'")
                print(res.readlines())
                return gr.File(value=os.path.join(current_log_folder,
                                                  zip_path),
                               visible=True)
            else:
                gr.Error(self.component_names.training_warn1)
                return gr.File()

        self.export_log.click(export_train_log,
                              inputs=[self.output_model_name],
                              outputs=[self.export_url],
                              queue=False)
