# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict

import gradio as gr

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.tuner_manager.manager_ui.component_names import \
    TunerManagerNames
from scepter.studio.tuner_manager.utils.dict import (delete_2level_dict,
                                                     update_2level_dict)
from scepter.studio.tuner_manager.utils.path import is_valid_filename
from scepter.studio.tuner_manager.utils.yaml import save_yaml
from scepter.studio.utils.uibase import UIBase


class BrowserUI(UIBase):
    def __init__(self, cfg, language='en'):
        self.work_dir = cfg.WORK_DIR
        self.yaml = os.path.join(self.work_dir, cfg.TUNER_LIST_YAML)
        if not FS.exists(self.yaml):
            self.saved_tuners = []
            with FS.put_to(self.yaml) as local_path:
                save_yaml({'TUNERS': self.saved_tuners}, local_path)
        else:
            with FS.get_from(self.yaml) as local_path:
                self.saved_tuners = Config.get_plain_cfg(
                    Config(cfg_file=local_path).TUNERS)
        self.saved_tuners_to_category()
        self.component_names = TunerManagerNames(language)
        self.language = language

    def saved_tuners_to_category(self):
        self.saved_tuners_category = OrderedDict()
        for tuner in self.saved_tuners:
            first_level = f"{tuner['BASE_MODEL']}-{tuner['TUNER_TYPE']}"
            second_level = f"{tuner['NAME']}"
            update_2level_dict(self.saved_tuners_category,
                               {first_level: {
                                   second_level: tuner
                               }})

    def category_to_saved_tuners(self):
        self.saved_tuners = []
        for k, v in self.saved_tuners_category.items():
            for kk, vv in v.items():
                self.saved_tuners.append(vv)

    def get_choices_and_values(self):
        diffusion_models_choice = list(self.saved_tuners_category.keys())
        diffusion_model = diffusion_models_choice[0] if len(
            diffusion_models_choice) > 0 else None
        tuner_models_choice = []
        tuner_model = None
        if diffusion_model:
            tuner_models_choice = list(
                self.saved_tuners_category.get(diffusion_model, {}).keys())
            tuner_model = tuner_models_choice[0] if len(
                tuner_models_choice) > 0 else None
        return diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model

    def create_ui(self, *args, **kwargs):
        diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model = self.get_choices_and_values(
        )
        with gr.Column():
            with gr.Box():
                gr.Markdown(self.component_names.browser_block_name)
                with gr.Row(variant='panel', equal_height=True):
                    with gr.Column(scale=4, min_width=0):
                        self.diffusion_models = gr.Dropdown(
                            label=self.component_names.base_models,
                            choices=diffusion_models_choice,
                            value=diffusion_model,
                            multiselect=False,
                            interactive=True)
                    with gr.Column(scale=4, min_width=0):
                        self.tuner_models = gr.Dropdown(
                            label=self.component_names.tuner_name,
                            choices=tuner_models_choice,
                            value=tuner_model,
                            multiselect=False,
                            interactive=True)
                    with gr.Column(scale=1, min_width=0):
                        self.save_button = gr.Button(
                            label='Save',
                            value=self.component_names.save_symbol,
                            elem_classes='type_row',
                            elem_id='save_button',
                            visible=True)
                        self.delete_button = gr.Button(
                            label='Delete',
                            value=self.component_names.delete_symbol,
                            elem_classes='type_row',
                            elem_id='delete_button',
                            visible=False)
                    with gr.Column(scale=1, min_width=0):
                        self.refresh_button = gr.Button(
                            label='Delete',
                            value=self.component_names.refresh_symbol,
                            elem_classes='type_row',
                            elem_id='refresh_button',
                            visible=True)

    def check_new_name(self, tuner_name):
        if tuner_name.strip() == '':
            return False, f"'{tuner_name}' is whitespace! Only support 'a-z', 'A-Z', '0-9' and '_'."
        if not is_valid_filename(tuner_name):
            return False, f"'{tuner_name}' is not a valid tuner name! Only support 'a-z', 'A-Z', '0-9' and '_'."
        for tuner in self.saved_tuners:
            if tuner_name == tuner['NAME']:
                return False, f"Tuner name '{tuner_name}' has been taken!"
        return True, 'legal'

    def save_tuner(self, src_path, sub_dir, tuner_name, tuner_example):
        tar_path = os.path.join(self.work_dir, sub_dir)
        if not FS.exists(tar_path):
            FS.make_dir(tar_path)
        tar_path = os.path.join(tar_path, tuner_name)

        FS.put_dir_from_local_dir(src_path, tar_path)

        tuner_example_path = None
        if tuner_example is not None:
            from PIL import Image
            tuner_example_path = os.path.join(tar_path, f'{tuner_name}.jpg')
            tuner_example = Image.fromarray(tuner_example)
            with FS.put_to(tuner_example_path) as local_path:
                tuner_example.save(local_path)
        return tar_path, tuner_example_path

    def add_tuner(self, new_tuner, manager, now_diffusion_model):
        self.saved_tuners.append(new_tuner)
        self.saved_tuners_to_category()
        with FS.put_to(self.yaml) as local_path:
            save_yaml({'TUNERS': self.saved_tuners}, local_path)

        # register to pipeline
        new_tuner = Config(cfg_dict=new_tuner, load=False)
        manager.inference.model_manage_ui.pipe_manager.register_tuner(
            new_tuner,
            name=new_tuner.NAME_ZH
            if self.language == 'zh' else new_tuner.NAME,
            is_customized=True)

        # update choices
        pipe_manager = manager.inference.model_manage_ui.pipe_manager
        now_pipeline = pipe_manager.model_level_info[now_diffusion_model][
            'pipeline'][0]
        default_choices = pipe_manager.module_level_choices
        custom_tunner_choices = []
        if 'customized_tuners' in default_choices and now_pipeline in default_choices[
                'customized_tuners']:
            custom_tunner_choices = default_choices['customized_tuners'][
                now_pipeline]['choices']

        # update tuner ui name_level_tuners
        name_level_tuners = manager.inference.tuner_ui.name_level_tuners
        if new_tuner.BASE_MODEL not in name_level_tuners:
            name_level_tuners[new_tuner.BASE_MODEL] = {}
        if self.language == 'zh':
            name_level_tuners[new_tuner.BASE_MODEL][
                new_tuner.NAME_ZH] = new_tuner
        else:
            name_level_tuners[new_tuner.BASE_MODEL][new_tuner.NAME] = new_tuner

        return custom_tunner_choices

    def delete_tuner(self, first_level, second_level, manager,
                     now_diffusion_model):
        self.saved_tuners_category, del_tuner = delete_2level_dict(
            self.saved_tuners_category, first_level, second_level)
        self.category_to_saved_tuners()
        save_yaml({'TUNERS': self.saved_tuners}, self.yaml)

        # update choices
        pipe_manager = manager.inference.model_manage_ui.pipe_manager
        now_pipeline = pipe_manager.model_level_info[now_diffusion_model][
            'pipeline'][0]
        default_choices = pipe_manager.module_level_choices
        custom_tuner_choices = []
        if 'customized_tuners' in default_choices and now_pipeline in default_choices[
                'customized_tuners']:
            custom_tuner_choices = default_choices['customized_tuners'][
                now_pipeline]['choices']

        # update tuner ui name_level_tuners
        del_tuner = Config(cfg_dict=del_tuner, load=False)
        name_level_tuners = manager.inference.tuner_ui.name_level_tuners
        if del_tuner.BASE_MODEL in name_level_tuners:
            if self.language == 'zh':
                del name_level_tuners[del_tuner.BASE_MODEL][del_tuner.NAME_ZH]
            else:
                del name_level_tuners[del_tuner.BASE_MODEL][del_tuner.NAME]

        return custom_tuner_choices

    def set_callbacks(self, manager, info_ui):
        def refresh_browser():
            diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model = self.get_choices_and_values(
            )
            return (gr.Dropdown(choices=diffusion_models_choice,
                                value=diffusion_model),
                    gr.Dropdown(choices=tuner_models_choice,
                                value=tuner_model))

        self.refresh_button.click(
            refresh_browser,
            inputs=[],
            outputs=[self.diffusion_models, self.tuner_models])

        def diffusion_model_change(diffusion_model):
            choices = list(
                self.saved_tuners_category.get(diffusion_model, {}).keys())
            return gr.Dropdown(choices=choices,
                               value=choices[-1] if len(choices) > 0 else None)

        self.diffusion_models.change(diffusion_model_change,
                                     inputs=[self.diffusion_models],
                                     outputs=[self.tuner_models],
                                     queue=True)

        def tuner_model_change(tuner_model, diffusion_model):
            tuner_info = {}
            if tuner_model is not None:
                tuner_info = self.saved_tuners_category[diffusion_model][
                    tuner_model]
            image_path = tuner_info.get('IMAGE_PATH', None)
            if image_path is not None:
                image_path = FS.get_from(image_path)
            return (gr.Text(value=tuner_info.get('NAME', '')),
                    gr.Text(value=tuner_info.get('NAME', ''),
                            interactive=True),
                    gr.Text(value=tuner_info.get('TUNER_TYPE', '')),
                    gr.Text(value=tuner_info.get('BASE_MODEL', '')),
                    gr.Text(value=tuner_info.get('DESCRIPTION', ''),
                            interactive=True), gr.Image(value=image_path),
                    gr.Text(value=tuner_info.get('PROMPT_EXAMPLE', ''),
                            interactive=True))

        self.tuner_models.change(
            tuner_model_change,
            inputs=[self.tuner_models, self.diffusion_models],
            outputs=[
                info_ui.tuner_name, info_ui.new_name, info_ui.tuner_type,
                info_ui.base_model, info_ui.tuner_desc, info_ui.tuner_example,
                info_ui.tuner_prompt_example
            ],
            queue=False)

        def save_tuner(tuner_name, new_name, tuner_desc, tuner_example,
                       tuner_prompt_example, now_diffusion_model, info_path):
            is_legal, msg = self.check_new_name(new_name)
            if not is_legal:
                gr.Info('Save failed because ' + msg)
                return (gr.Dropdown(), gr.Text(), gr.Dropdown())

            info = Config(cfg_file=info_path)
            model_dir = info.MODEL_PATH
            sub_dir = f'{info.BASE_MODEL}-{info.TUNER_TYPE}'
            model_dir, tuner_example = self.save_tuner(model_dir, sub_dir,
                                                       new_name, tuner_example)

            # config info update
            new_tuner = {
                'NAME': new_name,
                'NAME_ZH': new_name,
                'SOURCE': 'self_train',
                'DESCRIPTION': tuner_desc,
                'BASE_MODEL': info.BASE_MODEL,
                'MODEL_PATH': model_dir,
                'IMAGE_PATH': tuner_example,
                'TUNER_TYPE': info.TUNER_TYPE,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            custom_tuner_choices = self.add_tuner(new_tuner, manager,
                                                  now_diffusion_model)

            return (gr.Dropdown(choices=list(
                self.saved_tuners_category.keys()),
                                value=sub_dir),
                    gr.Dropdown(choices=list(
                        self.saved_tuners_category.get(sub_dir, {}).keys()),
                                value=new_name), gr.Text(value=new_name),
                    gr.Dropdown(choices=custom_tuner_choices))

        self.save_button.click(
            save_tuner,
            inputs=[
                info_ui.tuner_name, info_ui.new_name, info_ui.tuner_desc,
                info_ui.tuner_example, info_ui.tuner_prompt_example,
                manager.inference.model_manage_ui.diffusion_model,
                manager.inference.infer_info
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def delete_tuner(tuner_name, tuner_type, base_model,
                         now_diffusion_model):
            first_level = f'{base_model}-{tuner_type}'
            second_level = f'{tuner_name}'
            custom_tuner_choices = self.delete_tuner(first_level, second_level,
                                                     manager,
                                                     now_diffusion_model)
            return (gr.Dropdown(
                choices=list(self.saved_tuners_category.keys()),
                value=None), gr.Dropdown(choices=custom_tuner_choices))

        self.delete_button.click(
            delete_tuner,
            inputs=[
                info_ui.tuner_name, info_ui.tuner_type, info_ui.base_model,
                manager.inference.model_manage_ui.diffusion_model
            ],
            outputs=[
                self.diffusion_models,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)
