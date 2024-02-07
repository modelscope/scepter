# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import gradio as gr

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.self_train.self_train_ui.component_names import \
    InferenceUIName
from scepter.studio.self_train.utils.config_parser import (
    get_base_model_list, get_inference_para_by_model_version)
from scepter.studio.utils.uibase import UIBase


class InferenceUI(UIBase):
    def __init__(self, cfg, all_cfg_value, is_debug=False, language='en'):
        self.BASE_CFG_VALUE = all_cfg_value
        self.language = language
        self.base_model_info = get_base_model_list(self.BASE_CFG_VALUE)
        self.work_dir, _ = FS.map_to_local(cfg.WORK_DIR)
        os.makedirs(self.work_dir, exist_ok=True)
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
            if '@' in one_dir and os.path.exists(
                    os.path.join(self.work_dir, one_dir, 'checkpoint.pth')):
                if len(one_dir.split('@')) > 4:
                    have_model_list.append([one_dir, one_dir.split('@')[-1]])
        have_model_list.sort(key=lambda x: -int(x[-1][:-3]))
        self.model_list.extend([v[0].split('/')[-1]
                                for v in have_model_list][:50])
        self.infer_para_data = get_inference_para_by_model_version(
            self.BASE_CFG_VALUE, self.base_model_info.get('model_name', []),
            self.base_model_info.get('version_name', []))
        self.is_debug = is_debug
        self.component_names = InferenceUIName(language)

    def create_ui(self, *args, **kwargs):
        with gr.Box():
            gr.Markdown(self.component_names.output_model_block)
            with gr.Row():
                with gr.Column(scale=2, min_width=0):
                    self.output_model_name = gr.Dropdown(
                        label=self.component_names.output_model_name,
                        choices=self.model_list,
                        value=self.base_model_info.get('model_default', ''),
                        interactive=True)
                with gr.Column(scale=1, min_width=0):
                    self.refresh_model_gbtn = gr.Button(
                        self.component_names.refresh_model_gbtn)
            with gr.Row():
                with gr.Column(scale=2, min_width=0):
                    self.extra_model_gtxt = gr.Text(
                        label=self.component_names.extra_model_gtxt,
                        show_label=False,
                        placeholder='Add Extra Model')
                with gr.Column(scale=1, min_width=0):
                    self.extra_model_gbtn = gr.Button(
                        self.component_names.extra_model_gbtn)
            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    self.go_to_inferece_btn = gr.Button(
                        self.component_names.go_to_inference)

    def set_callbacks(self, trainer_ui, manager):
        self.manager = manager

        def add_model(model_name):
            if model_name not in self.model_list:
                self.model_list.append(model_name)
            return '', gr.Dropdown(choices=self.model_list)

        def refresh_model():
            return gr.Dropdown(choices=self.model_list)

        self.extra_model_gbtn.click(
            fn=add_model,
            inputs=[self.extra_model_gtxt],
            outputs=[self.extra_model_gtxt, self.output_model_name],
            queue=False)
        self.refresh_model_gbtn.click(fn=refresh_model,
                                      inputs=[],
                                      outputs=[self.output_model_name],
                                      queue=False)

        def go_to_inferece(output_model):
            output_model_path = os.path.join(self.work_dir, output_model)
            _, _, base_model, _, resolution, _ = output_model_path.split('@')
            tuner_cfg = Config(cfg_dict={}, load=False)
            tuner_cfg.NAME = output_model
            tuner_cfg.NAME_ZH = output_model
            tuner_cfg.BASE_MODEL = base_model
            model_path = os.path.join(output_model_path, 'checkpoint.pth')
            if not os.path.exists(model_path):
                gr.Error(self.component_names.inference_err4)
            tuner_cfg.MODEL_PATH = model_path
            self.manager.inference.model_manage_ui.pipe_manager.register_tuner(
                tuner_cfg,
                name=tuner_cfg.NAME_ZH
                if self.language == 'zh' else tuner_cfg.NAME,
                is_customized=True)

            pipeline_level_modules = self.manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if tuner_cfg.BASE_MODEL not in pipeline_level_modules:
                gr.Error(self.component_names.inference_err3 +
                         tuner_cfg.BASE_MODEL)
            pipeline_ins = pipeline_level_modules[tuner_cfg.BASE_MODEL]
            diffusion_model = f"{tuner_cfg.BASE_MODEL}_{pipeline_ins.diffusion_model['name']}"

            default_choices = self.manager.inference.model_manage_ui.pipe_manager.module_level_choices
            if 'customized_tuners' in default_choices and tuner_cfg.BASE_MODEL in default_choices[
                    'customized_tuners']:
                tunner_choices = default_choices['customized_tuners'][
                    tuner_cfg.BASE_MODEL]['choices']
                tunner_default = default_choices['customized_tuners'][
                    tuner_cfg.BASE_MODEL]['default']
                if not isinstance(tunner_default, list):
                    tunner_default = [tunner_default]
            else:
                tunner_choices = []
                tunner_default = ''
            return (gr.Tabs(selected='inference'),
                    gr.Dropdown(choices=tunner_choices, value=tunner_default),
                    gr.Dropdown(value=diffusion_model),
                    gr.Tabs(selected='tuner_ui'))

        self.go_to_inferece_btn.click(
            go_to_inferece,
            inputs=[self.output_model_name],
            outputs=[
                manager.tabs, manager.inference.tuner_ui.custom_tuner_model,
                manager.inference.model_manage_ui.diffusion_model,
                manager.inference.setting_tab
            ],
            queue=False)
