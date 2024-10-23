# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
from collections import OrderedDict

import gradio as gr
import torch
import yaml
from safetensors.torch import save_file
from scepter.modules.utils.config import Config
from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS, IoString
from scepter.modules.utils.module_transform import (
    convert_ldm_clip_checkpoint_v1, convert_ldm_unet_tuner_checkpoint,
    convert_lora_checkpoint, create_unet_diffusers_config)
from scepter.studio.tuner_manager.manager_ui.component_names import \
    TunerManagerNames
from scepter.studio.utils.uibase import UIBase


def check_data(data):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = check_data(v)
        return data
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            data[idx] = check_data(v)
        return data
    else:
        if isinstance(data, IoString):
            return str(data)
        return data


class InfoUI(UIBase):
    def __init__(self, cfg, language='en'):
        self.component_names = TunerManagerNames(language)
        self.work_dir = cfg.WORK_DIR
        self.export_folder = os.path.join(self.work_dir, cfg.EXPORT_DIR)
        self.language = language

    def create_ui(self, *args, **kwargs):
        with gr.Column():
            with gr.Group():
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
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    self.tuner_desc = gr.Text(
                                        value='',
                                        label=self.component_names.tuner_desc,
                                        lines=4)
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    self.save_bt2 = gr.Button(
                                        value=self.component_names.save_symbol,
                                        elem_classes='type_row',
                                        elem_id='save_button',
                                        visible=True,
                                        # lines=4
                                    )
                    with gr.Column(variant='panel', scale=1, min_width=0):
                        with gr.Group(visible=True):
                            with gr.Row(equal_height=True):
                                self.tuner_example = gr.Image(
                                    label=self.component_names.tuner_example,
                                    sources=['upload'],
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
                                    lines=1)
                            with gr.Row(equal_height=True):
                                self.hf_url = gr.Text(
                                    value='',
                                    label=self.component_names.hf_url,
                                    lines=1)
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, min_width=0):
                                    self.go_to_inferece_btn = gr.Button(
                                        self.component_names.go_to_inference)
                                with gr.Column(scale=1, min_width=0):
                                    self.local_download_bt = gr.Button(
                                        value=self.component_names.
                                        download_to_local,
                                        # elem_classes='type_row',
                                        elem_id='save_button')
                            with gr.Row(
                                    equal_height=True,
                                    visible=False,
                                    variant='panel') as self.download_choices:
                                with gr.Column(scale=4, min_width=0):
                                    self.download_select = gr.Dropdown(
                                        choices=['.zip', '.safetensors'],
                                        value='.zip',
                                        interactive=True,
                                        label='Select Download Format',
                                        show_label=False)
                                with gr.Column(scale=1, min_width=0):
                                    self.download_confirm = gr.Button(
                                        value=self.component_names.submit,
                                        elem_classes='type_row',
                                        elem_id='save_button')
                            with gr.Row(equal_height=True):
                                self.export_url = gr.File(
                                    label=self.component_names.export_file,
                                    visible=False,
                                    value=None,
                                    interactive=False,
                                    show_label=True)

    def set_callbacks(self, manager, browser_ui):
        # def set_callbacks(self, manager):
        def go_to_inferece(new_name, tuner_desc, tuner_prompt_example,
                           tuner_type, base_model, login_user_name):
            all_tuners = manager.tuner_manager.browser_ui.saved_tuners_category.get(
                login_user_name, OrderedDict())
            sub_dir = f'{base_model}-{tuner_type}'
            current_model = all_tuners.get(sub_dir, {}).get(new_name, None)
            if current_model is None:
                raise gr.Error(self.component_names.model_err4)
            model_dir, _ = FS.map_to_local(current_model['MODEL_PATH'])
            if not os.path.exists(model_dir):
                FS.get_dir_to_local_dir(current_model['MODEL_PATH'])

            tuner_dict = {
                'NAME': new_name,
                'NAME_ZH': new_name,
                'SOURCE': 'self_train',
                'DESCRIPTION': tuner_desc,
                'BASE_MODEL': base_model,
                'MODEL_PATH': model_dir,
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            if 'IMAGE_PATH' in current_model:
                tuner_example_abspath = os.path.join(
                    model_dir, current_model['IMAGE_PATH'])
                if FS.exists(tuner_example_abspath):
                    tuner_dict.update({'IMAGE_PATH': tuner_example_abspath})

            tuner_cfg = Config(cfg_dict=tuner_dict, load=False)
            cfg_file = os.path.join(model_dir, 'meta.yaml')
            if not os.path.exists(model_dir):
                raise gr.Error(self.component_names.model_err4)

            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if tuner_cfg.BASE_MODEL not in pipeline_level_modules:
                raise gr.Error(self.component_names.model_err3 +
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
                        raise gr.Error(self.component_names.model_err5 +
                                       tuner_cfg.NAME_ZH)
                    else:
                        raise gr.Error(self.component_names.model_err5 +
                                       tuner_cfg.NAME)
                if not isinstance(tunner_default, list):
                    tunner_default = [tunner_default]
            else:
                tunner_choices = []
                tunner_default = []

            with open(cfg_file, 'w') as f_out:
                yaml.dump(copy.deepcopy(check_data(tuner_cfg.cfg_dict)),
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

            if tuner_cfg.BASE_MODEL == 'EDIT':
                selected_tab = 'stylebooth_ui' if tuner_cfg.BASE_MODEL == 'EDIT' else 'tuner_ui'
                checkboxes = [
                    '使用微调', 'StyleBooth'
                ] if self.language == 'zh' else ['Use Tuners', 'StyleBooth']
            else:
                selected_tab = 'tuner_ui'
                checkboxes = ['使用微调'
                              ] if self.language == 'zh' else ['Use Tuners']

            return (gr.Tabs(selected='inference'), cfg_file,
                    gr.Tabs(selected=selected_tab),
                    gr.CheckboxGroup(value=checkboxes),
                    gr.Dropdown(value=diffusion_model),
                    gr.Dropdown(choices=tunner_choices, value=tunner_default))

        self.go_to_inferece_btn.click(
            go_to_inferece,
            inputs=[
                self.new_name, self.tuner_desc, self.tuner_prompt_example,
                self.tuner_type, self.base_model, manager.user_name
            ],
            outputs=[
                manager.tabs, manager.inference.infer_info,
                manager.inference.setting_tab,
                manager.inference.check_box_for_setting,
                manager.inference.model_manage_ui.diffusion_model,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def export_zip(tuner_name, base_model, tuner_type, login_user_name):
            all_tuners = manager.tuner_manager.browser_ui.saved_tuners_category.get(
                login_user_name, OrderedDict())
            sub_dir = f'{base_model}-{tuner_type}'
            current_model = all_tuners.get(sub_dir, {}).get(tuner_name, None)
            if current_model is None:
                raise gr.Error(self.component_names.model_err4)
            enable_share = current_model.get('ENABLE_SHARE', True)
            if not enable_share:
                gr.Info('Error: The model is not allowed to Export!')
                return gr.File()
            model_dir = FS.get_dir_to_local_dir(current_model['MODEL_PATH'])
            zip_path = get_md5(os.path.join(self.export_folder,
                                            tuner_name)) + '.zip'
            with FS.put_to(zip_path) as local_zip:
                res = os.popen(
                    f"cd '{os.path.abspath(model_dir)}' "
                    f"&& zip -r '{os.path.abspath(local_zip)}' ./* ")
                print(res.readlines())
            if not FS.exists(zip_path):
                raise gr.Error(self.component_names.export_zip_err1)
            local_zip = FS.get_from(zip_path)
            gr.Info(self.component_names.save_end)
            return gr.File(value=local_zip, visible=True)

        def export_safetensors(tuner_name, base_model, tuner_type,
                               login_user_name):
            # only support sd1.5
            all_tuners = manager.tuner_manager.browser_ui.saved_tuners_category.get(
                login_user_name, OrderedDict())
            sub_dir = f'{base_model}-{tuner_type}'
            current_model = all_tuners.get(sub_dir, {}).get(tuner_name, None)
            if current_model is None:
                raise gr.Error(self.component_names.model_err4)
            enable_share = current_model.get('ENABLE_SHARE', True)
            if not enable_share:
                gr.Info('Error: The model is not allowed to Export!')
                return gr.File()

            model_dir = FS.get_dir_to_local_dir(current_model['MODEL_PATH'])

            swift_checkpoint = {}
            swift_dirs = os.listdir(model_dir)
            for swift_dir in swift_dirs:
                if not os.path.isdir(os.path.join(model_dir, swift_dir)):
                    continue
                swift_files = os.listdir(os.path.join(model_dir, swift_dir))
                for swift_file in swift_files:
                    if '.bin' in swift_file:
                        checkpoint_path = os.path.join(model_dir, swift_dir,
                                                       swift_file)
                        checkpoint = torch.load(checkpoint_path,
                                                map_location='cpu')
                        swift_checkpoint.update(checkpoint)

            unet_config = create_unet_diffusers_config(v2=False)
            ckpt_unet = convert_ldm_unet_tuner_checkpoint(
                v2=False,
                checkpoint=swift_checkpoint,
                config=unet_config,
                unet_key='model.')
            lora_state_dict = convert_lora_checkpoint(ckpt_unet=ckpt_unet)
            ckpt_te = convert_ldm_clip_checkpoint_v1(swift_checkpoint)
            lora_te_state_dict = convert_lora_checkpoint(ckpt_text=ckpt_te)
            lora_state_dict.update(lora_te_state_dict)

            save_path = get_md5(os.path.join(self.export_folder,
                                             tuner_name)) + '.safetensors'
            with FS.put_to(save_path) as local_file:
                save_file(lora_state_dict, local_file)
            if not FS.exists(save_path):
                raise gr.Error(self.component_names.export_zip_err1)
            local_zip = FS.get_from(save_path)
            gr.Info(self.component_names.save_end)
            return gr.File(value=local_zip, visible=True)

        def export_file(tuner_name, base_model, tuner_type, download_type,
                        login_user_name):
            gr.Info(self.component_names.save_start)
            if download_type == '.zip':
                return export_zip(tuner_name, base_model, tuner_type,
                                  login_user_name)
            elif download_type == '.safetensors':
                return export_safetensors(tuner_name, base_model, tuner_type,
                                          login_user_name)

        def change_visible():
            return gr.update(visible=True)

        self.local_download_bt.click(change_visible,
                                     inputs=[],
                                     outputs=[self.download_choices],
                                     queue=False)

        self.download_confirm.click(export_file,
                                    inputs=[
                                        self.tuner_name, self.base_model,
                                        self.tuner_type, self.download_select,
                                        manager.user_name
                                    ],
                                    outputs=[self.export_url],
                                    queue=True)

        def save_tuner_func(tuner_name, new_name, tuner_desc, tuner_example,
                            tuner_prompt_example, base_model, tuner_type,
                            login_user_name):
            return browser_ui.save_tuner(manager, tuner_name, new_name,
                                         tuner_desc, tuner_example,
                                         tuner_prompt_example, base_model,
                                         tuner_type, login_user_name)

        self.save_bt2.click(save_tuner_func,
                            inputs=[
                                self.tuner_name, self.new_name,
                                self.tuner_desc, self.tuner_example,
                                self.tuner_prompt_example, self.base_model,
                                self.tuner_type, manager.user_name
                            ],
                            outputs=[
                                browser_ui.diffusion_models,
                                browser_ui.tuner_models, self.tuner_name,
                                manager.inference.tuner_ui.custom_tuner_model
                            ],
                            queue=True)
