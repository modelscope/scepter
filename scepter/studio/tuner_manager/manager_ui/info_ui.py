# -*- coding: utf-8 -*-

import gradio as gr

from scepter.studio.tuner_manager.manager_ui.component_names import \
    TunerManagerNames
from scepter.studio.utils.uibase import UIBase


class InfoUI(UIBase):
    def __init__(self, cfg, language='en'):
        self.component_names = TunerManagerNames(language)

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

    def set_callbacks(self, manager):
        pass
