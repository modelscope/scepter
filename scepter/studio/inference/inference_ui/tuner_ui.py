# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import gradio as gr
from tqdm import tqdm

from scepter.modules.utils.file_system import FS
from scepter.studio.inference.inference_ui.component_names import TunerUIName
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


class TunerUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.cfg = cfg
        self.pipe_manager = pipe_manager
        self.default_choices = pipe_manager.module_level_choices
        default_diffusion_model = self.default_choices['diffusion_model'][
            'default']
        self.default_pipeline = pipe_manager.model_level_info[
            default_diffusion_model]['pipeline'][0]
        self.tunner_choices = []
        if self.default_pipeline in self.default_choices['tuners']:
            self.tunner_choices = self.default_choices['tuners'][
                self.default_pipeline]['choices']
            self.tunner_default = self.default_choices['tuners'][
                self.default_pipeline]['default']
        self.custom_tuner_choices = []
        if self.default_pipeline in self.default_choices.get(
                'customized_tuners', []):
            self.custom_tuner_choices = self.default_choices[
                'customized_tuners'][self.default_pipeline]['choices']

        self.tunner_default = None
        self.component_names = TunerUIName(language)
        self.cfg_tuners = cfg.TUNERS + cfg.CUSTOM_TUNERS
        self.name_level_tuners = {}
        for one_tuner in tqdm(self.cfg_tuners):
            if one_tuner.BASE_MODEL not in self.name_level_tuners:
                self.name_level_tuners[one_tuner.BASE_MODEL] = {}
            # if one_tuner.get('IMAGE_PATH', None):
            #     one_tuner.IMAGE_PATH = FS.get_from(one_tuner.IMAGE_PATH)
            if language == 'zh':
                self.name_level_tuners[one_tuner.BASE_MODEL][
                    one_tuner.NAME_ZH] = one_tuner
            else:
                self.name_level_tuners[one_tuner.BASE_MODEL][
                    one_tuner.NAME] = one_tuner

    def create_ui(self, *args, **kwargs):
        self.state = gr.State(value=False)
        with gr.Column(equal_height=True, visible=False) as self.tab:
            with gr.Row(scale=1):
                with gr.Column(variant='panel', scale=1, min_width=0):
                    with gr.Group(visible=True):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                self.tuner_model = gr.Dropdown(
                                    label=self.component_names.tuner_model,
                                    choices=self.tunner_choices,
                                    value=None,
                                    multiselect=True,
                                    interactive=True)
                            with gr.Column(scale=1):
                                self.custom_tuner_model = gr.Dropdown(
                                    label=self.component_names.
                                    custom_tuner_model,
                                    choices=self.custom_tuner_choices,
                                    value=None,
                                    multiselect=True,
                                    interactive=True)
                                self.save_button = gr.Button(
                                    label=self.component_names.save_button,
                                    value=self.component_names.save_button,
                                    elem_classes='type_row',
                                    elem_id='save_button',
                                    visible=True)
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                self.tuner_type = gr.Text(
                                    value='',
                                    label=self.component_names.tuner_type)
                            with gr.Column(scale=1):
                                self.base_model = gr.Text(
                                    value='',
                                    label=self.component_names.base_model)
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
                                interactive=False)
                        with gr.Row(equal_height=True):
                            self.tuner_prompt_example = gr.Text(
                                value='',
                                label=self.component_names.
                                tuner_prompt_example,
                                lines=2)

            with gr.Accordion(label=self.component_names.advance_block_name,
                              open=False):
                self.tuner_scale = gr.Slider(
                    label=self.component_names.tuner_scale,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=1.0,
                    interactive=True)

            self.example_block = gr.Accordion(
                label=self.component_names.example_block_name, open=True)
        gallery_ui = kwargs.pop('gallery_ui', None)
        gallery_ui.register_components({
            'tuner_state':
            self.state,
            'tuner_model':
            self.tuner_model,
            'tuner_scale':
            self.tuner_scale,
            'custom_tuner_model':
            self.custom_tuner_model,
        })

    def set_callbacks(self, model_manage_ui, **kwargs):
        manager = kwargs.pop('manager')
        gallery_ui = kwargs.pop('gallery_ui')
        with self.example_block:
            gr.Examples(examples=self.component_names.examples,
                        inputs=[self.tuner_model, gallery_ui.prompt])

        def tuner_model_change(tuner_model, diffusion_model):
            diffusion_model_info = self.pipe_manager.model_level_info[
                diffusion_model]
            now_pipeline = diffusion_model_info['pipeline'][0]
            tuner_info = {}
            if tuner_model is not None and len(tuner_model) > 0:
                tuner_info = self.name_level_tuners.get(now_pipeline, {}).get(
                    tuner_model[-1], {})
            if tuner_info.get(
                    'IMAGE_PATH',
                    None) and not os.path.exists(tuner_info.IMAGE_PATH):
                tuner_info.IMAGE_PATH = FS.get_from(tuner_info.IMAGE_PATH)
            return (gr.Text(value=tuner_info.get('TUNER_TYPE', '')),
                    gr.Text(value=tuner_info.get('BASE_MODEL', '')),
                    gr.Text(value=tuner_info.get('DESCRIPTION', '')),
                    gr.Image(value=tuner_info.get('IMAGE_PATH', None)),
                    gr.Text(value=tuner_info.get('PROMPT_EXAMPLE', '')))

        self.tuner_model.change(
            tuner_model_change,
            inputs=[self.tuner_model, model_manage_ui.diffusion_model],
            outputs=[
                self.tuner_type, self.base_model, self.tuner_desc,
                self.tuner_example, self.tuner_prompt_example
            ],
            queue=False)

        self.custom_tuner_model.change(
            tuner_model_change,
            inputs=[self.custom_tuner_model, model_manage_ui.diffusion_model],
            outputs=[
                self.tuner_type, self.base_model, self.tuner_desc,
                self.tuner_example, self.tuner_prompt_example
            ],
            queue=False)

        def save_customized_tuner(tuner_model, diffusion_model):
            diffusion_model_info = self.pipe_manager.model_level_info[
                diffusion_model]
            now_pipeline = diffusion_model_info['pipeline'][0]
            tuner_info = {}
            if tuner_model is not None and len(tuner_model) > 0:
                tuner_info = self.name_level_tuners.get(now_pipeline, {}).get(
                    tuner_model[-1], {})
            if tuner_info.get(
                    'IMAGE_PATH',
                    None) and not os.path.exists(tuner_info.IMAGE_PATH):
                tuner_info.IMAGE_PATH = FS.get_from(tuner_info.IMAGE_PATH)
            return (gr.Tabs(selected='tuner_manager'),
                    gr.Text(value=tuner_info.NAME), gr.Text(value=''),
                    gr.Text(value=tuner_info.get('TUNER_TYPE', '')),
                    gr.Text(value=tuner_info.get('BASE_MODEL', '')),
                    gr.Text(value=tuner_info.get('DESCRIPTION', '')),
                    gr.Image(value=tuner_info.get('IMAGE_PATH', None)),
                    gr.Text(value=tuner_info.get('PROMPT_EXAMPLE', '')))

        self.save_button.click(
            save_customized_tuner,
            inputs=[self.custom_tuner_model, model_manage_ui.diffusion_model],
            outputs=[
                manager.tabs, manager.tuner_manager.info_ui.tuner_name,
                manager.tuner_manager.info_ui.new_name,
                manager.tuner_manager.info_ui.tuner_type,
                manager.tuner_manager.info_ui.base_model,
                manager.tuner_manager.info_ui.tuner_desc,
                manager.tuner_manager.info_ui.tuner_example,
                manager.tuner_manager.info_ui.tuner_prompt_example
            ])
