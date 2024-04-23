# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os

import gradio as gr
import yaml

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.tuner_manager.manager_ui.component_names import \
    TunerManagerNames
from scepter.studio.utils.uibase import UIBase


class InfoUI(UIBase):
    def __init__(self, cfg, language='en'):
        self.component_names = TunerManagerNames(language)
        self.work_dir = cfg.WORK_DIR
        self.export_folder = os.path.join(self.work_dir, cfg.EXPORT_DIR)
        self.language = language

    def create_ui(self, *args, **kwargs):
        with gr.Column():
            with gr.Box():
                gr.Markdown(self.component_names.info_block_name)
                with gr.Row(variant='panel', equal_height=True):
                    with gr.Column(variant='panel', scale=1, min_width=0):
                        with gr.Group(visible=True):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    self.tuner_name = gr.Text(
                                        value='',
                                        label=self.component_names.tuner_name,
                                        interactive=False)
                                with gr.Column(scale=1):
                                    self.new_name = gr.Text(
                                        value='',
                                        label=self.component_names.rename)
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    self.tuner_type = gr.Text(
                                        value='',
                                        label=self.component_names.tuner_type,
                                        interactive=False)
                                with gr.Column(scale=1):
                                    self.base_model = gr.Text(
                                        value='',
                                        label=self.component_names.
                                        base_model_name,
                                        interactive=False)
                                with gr.Column(scale=1):
                                    self.tuner_desc = gr.Text(
                                        value='',
                                        label=self.component_names.tuner_desc,
                                        lines=4)
                    with gr.Column(variant='panel', scale=1, min_width=0):
                        with gr.Group(visible=True):
                            with gr.Row(equal_height=True):
                                self.tuner_example = gr.Image(
                                    label=self.component_names.tuner_example,
                                    source='upload',
                                    value=None,
                                    interactive=True)
                            with gr.Row(equal_height=True):
                                self.tuner_prompt_example = gr.Text(
                                    value='',
                                    label=self.component_names.
                                    tuner_prompt_example,
                                    lines=2)
                            with gr.Row(equal_height=True):
                                self.ms_url = gr.Text(
                                    value='',
                                    label=self.component_names.ms_url,
                                    lines=2)
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, min_width=0):
                                    self.go_to_inferece_btn = gr.Button(
                                        self.component_names.go_to_inference)
                                with gr.Column(scale=1, min_width=0):
                                    self.local_download_bt = gr.Button(
                                        label='Download to Local Dir',
                                        value=self.component_names.
                                        download_to_local,
                                        # elem_classes='type_row',
                                        elem_id='save_button')
                                    self.export_url = gr.File(
                                        label=self.component_names.export_file,
                                        visible=False,
                                        value=None,
                                        interactive=False,
                                        show_label=True)

    def set_callbacks(self, manager):
        def go_to_inferece(new_name, tuner_desc, tuner_prompt_example,
                           tuner_type, base_model):
            sub_dir = f'{base_model}-{tuner_type}'
            tar_path = os.path.join(self.work_dir, sub_dir)
            model_dir = os.path.join(tar_path, new_name)
            if not os.path.exists(model_dir):
                tuner_list = Config(
                    cfg_file=os.path.join(self.work_dir, 'tuner_list.yaml'))
                for tuner_item in tuner_list.get('TUNERS', []):
                    if tuner_item.NAME == new_name:
                        model_dir = tuner_item.MODEL_PATH
            tuner_example = os.path.join(model_dir, 'image.jpg')
            if not os.path.exists(tuner_example):
                tuner_example = None
            tuner_dict = {
                'NAME': new_name,
                'NAME_ZH': new_name,
                'SOURCE': 'self_train',
                'DESCRIPTION': tuner_desc,
                'BASE_MODEL': base_model,
                'MODEL_PATH': model_dir,
                'IMAGE_PATH': tuner_example,
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            tuner_cfg = Config(cfg_dict=tuner_dict, load=False)
            cfg_file = os.path.join(model_dir, 'meta.yaml')
            if not os.path.exists(model_dir):
                gr.Error(self.component_names.model_err4)

            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if tuner_cfg.BASE_MODEL not in pipeline_level_modules:
                gr.Error(self.component_names.model_err3 +
                         tuner_cfg.BASE_MODEL)
            pipeline_ins = pipeline_level_modules[tuner_cfg.BASE_MODEL]
            diffusion_model = f"{tuner_cfg.BASE_MODEL}_{pipeline_ins.diffusion_model['name']}"

            default_choices = manager.inference.model_manage_ui.pipe_manager.module_level_choices
            if 'customized_tuners' in default_choices:
                if tuner_cfg.BASE_MODEL not in default_choices[
                        'customized_tuners']:
                    default_choices['customized_tuners'] = {}
                tunner_choices = default_choices['customized_tuners'][
                    tuner_cfg.BASE_MODEL]['choices']
                tunner_default = tuner_cfg.NAME if self.language == 'en' else tuner_cfg.NAME_ZH
                if tunner_default not in tunner_choices:
                    if self.language == 'zh':
                        gr.Error(self.component_names.model_err5 +
                                 tuner_cfg.NAME_ZH)
                    else:
                        gr.Error(self.component_names.model_err5 +
                                 tuner_cfg.NAME)
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

            base_model = tuner_cfg.get('BASE_MODEL', '')

            if not base_model == '':
                if base_model not in manager.inference.tuner_ui.name_level_tuners:
                    manager.inference.tuner_ui.name_level_tuners[
                        base_model] = {}
                manager.inference.tuner_ui.name_level_tuners[base_model][
                    tuner_cfg.NAME] = tuner_cfg

            return (
                gr.Tabs(selected='inference'), cfg_file,
                gr.Tabs(selected='tuner_ui'),
                gr.CheckboxGroup(
                    value='使用微调' if self.language == 'zh' else 'Use Tuners'),
                gr.Dropdown(value=diffusion_model),
                gr.Dropdown(choices=tunner_choices, value=tunner_default))

        self.go_to_inferece_btn.click(
            go_to_inferece,
            inputs=[
                self.new_name, self.tuner_desc, self.tuner_prompt_example,
                self.tuner_type, self.base_model
            ],
            outputs=[
                manager.tabs, manager.inference.infer_info,
                manager.inference.setting_tab,
                manager.inference.check_box_for_setting,
                manager.inference.model_manage_ui.diffusion_model,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def export_zip(tuner_name, base_model, tuner_type):
            sub_dir = f'{base_model}-{tuner_type}'
            if os.path.exists(os.path.join(self.work_dir, sub_dir,
                                           tuner_name)):
                model_dir = os.path.join(self.work_dir, sub_dir, tuner_name)
            else:
                model_dir = ''
                tuner_list = Config(
                    cfg_file=os.path.join(self.work_dir, 'tuner_list.yaml'))
                for tuner_item in tuner_list.get('TUNERS', []):
                    if tuner_item.NAME == tuner_name:
                        model_dir = tuner_item.MODEL_PATH
                        break
                if not os.path.exists(model_dir) or model_dir == '':
                    raise gr.Error(self.component_names.model_err4)
            zip_path = os.path.join(self.export_folder, f'{tuner_name}.zip')
            with FS.put_to(zip_path) as local_zip:
                res = os.popen(
                    f"cd '{model_dir}' "
                    f"&& zip -r '{os.path.abspath(local_zip)}' ./* ")
                print(res.readlines())
            if not FS.exists(zip_path):
                raise gr.Error(self.component_names.export_zip_err1)
            local_zip = FS.get_from(zip_path)
            return gr.File(value=local_zip, visible=True)

        self.local_download_bt.click(
            export_zip,
            inputs=[self.tuner_name, self.base_model, self.tuner_type],
            outputs=[self.export_url],
            queue=False)
