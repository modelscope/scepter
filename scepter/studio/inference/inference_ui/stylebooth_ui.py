# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import gradio as gr

from scepter.studio.inference.inference_ui.component_names import \
    StyleboothUIName
from scepter.studio.utils.uibase import UIBase


class StyleboothUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.cfg = cfg
        self.pipe_manager = pipe_manager
        self.component_names = StyleboothUIName(language)

    def create_ui(self, *args, **kwargs):
        self.state = gr.State(value=False)
        with gr.Column(equal_height=False, visible=False) as self.tab:
            with gr.Row():
                self.selected_app = gr.Dropdown(
                    label=self.component_names.dropdown_name,
                    choices=self.component_names.apps,
                    value=self.component_names.apps[0],
                    type='index',
                    interactive=False)
            with gr.Row(equal_height=True):
                with gr.Column(variant='panel',
                               scale=1,
                               min_width=0,
                               visible=True) as self.col1:
                    self.edit_image = gr.Image(
                        label=self.component_names.source_image,
                        type='pil',
                        tool='editor',
                        interactive=True)
                with gr.Column(variant='panel',
                               scale=1,
                               min_width=0,
                               visible=True) as self.col2:
                    with gr.Group(visible=True):
                        self.exemplar_image = gr.Image(
                            label=self.component_names.exemplar_image,
                            type='pil',
                            interactive=True,
                            visible=False)
                        with gr.Row(equal_height=True):
                            with gr.Group(visible=True):
                                self.instruction_format = gr.Dropdown(
                                    label=self.component_names.ins_format,
                                    choices=self.component_names.
                                    tb_ins_format_choice,
                                    value=self.component_names.
                                    tb_ins_format_choice[0],
                                    multiselect=False,
                                    interactive=True,
                                    allow_custom_value=True)
                                self.target_style = gr.Dropdown(
                                    label=self.component_names.style_format.
                                    format(self.component_names.tb_identifier),
                                    choices=self.component_names.
                                    tb_target_style,
                                    value=None,
                                    multiselect=False,
                                    interactive=True,
                                    allow_custom_value=True)
                                self.compose_instruction = gr.Button(
                                    label=self.component_names.compose_button,
                                    value=self.component_names.compose_button,
                                    elem_classes='type_row',
                                    elem_id='push',
                                    visible=True)
                with gr.Column(variant='panel', scale=1, min_width=0):
                    with gr.Group(visible=True):
                        self.guide_scale_text = gr.Slider(
                            label=self.component_names.guide_scale_text,
                            minimum=1,
                            maximum=10,
                            step=0.5,
                            value=7.5,
                            interactive=True)
                        self.guide_scale_image = gr.Slider(
                            label=self.component_names.guide_scale_image,
                            minimum=1,
                            maximum=10,
                            step=0.5,
                            value=1.5,
                            interactive=True)
                        self.guide_rescale = gr.Slider(
                            label=self.component_names.guide_rescale,
                            minimum=0,
                            maximum=1.0,
                            step=0.1,
                            value=0.5,
                            interactive=True)
                        # self.resolution = gr.Slider(
                        #     label=self.component_names.resolution,
                        #     minimum=384, maximum=768, step=32, value=512,
                        #     interactive=True)
            # gallery_ui = kwargs.pop("gallery_ui")
            # with gr.Column(visible=True):
            #     with gr.Column(visible=True) as self.general_examples:
            #         gr.Examples(
            #             examples=self.component_names.general_examples,
            #             inputs=[gallery_ui.prompt, self.source_image],
            #             outputs=[gallery_ui.prompt, self.source_image],
            #             fn=lambda x, y: (x, y),
            #             cache_examples=False)
            #     with gr.Column(visible=False) as self.tuner_examples:
            #         gr.Examples(
            #             examples=self.component_names.tuner_examples,
            #             inputs=[self.editor_model, gallery_ui.prompt, self.source_image],
            #             outputs=[self.editor_model, gallery_ui.prompt, self.source_image],
            #             fn=lambda x, y, z: (x, y, z),
            #             cache_examples=True)
        gallery_ui = kwargs.pop('gallery_ui', None)
        gallery_ui.register_components({
            'stylebooth_state':
            self.state,
            'style_edit_image':
            self.edit_image,
            'style_exemplar_image':
            self.exemplar_image,
            'style_guide_scale_text':
            self.guide_scale_text,
            'style_guide_scale_image':
            self.guide_scale_image,
        })

    def set_callbacks(self, model_manage_ui, diffusion_ui, gallery_ui,
                      **kwargs):
        def app_change(app):
            selected = app
            # selected = None
            # for i, name in enumerate(self.component_names.apps):
            #     if name == app:
            #         selected = i
            #         continue
            # assert selected in (0, 1)
            if not selected:
                format_choices = self.component_names.tb_ins_format_choice
                style_choices = self.component_names.tb_target_style
            else:
                format_choices = self.component_names.eb_ins_format_choice
                style_choices = self.component_names.eb_target_style

            identifier = self.component_names.tb_identifier if selected == 0 else self.component_names.eb_identifier
            return (gr.update(value=None, visible=(selected != 0)),
                    gr.update(choices=format_choices, value=format_choices[0]),
                    gr.update(choices=style_choices,
                              value=style_choices[0],
                              interactive=(selected == 0),
                              label=self.component_names.style_format.format(
                                  identifier)))

        self.selected_app.change(app_change,
                                 inputs=[self.selected_app],
                                 outputs=[
                                     self.exemplar_image,
                                     self.instruction_format, self.target_style
                                 ],
                                 queue=False)

        def compose_instruction(format, style, app):
            if app == 1:
                return format
            identifier = self.component_names.tb_identifier if app == 0 else self.component_names.eb_identifier
            return format.replace(identifier, style)

        self.compose_instruction.click(compose_instruction,
                                       inputs=[
                                           self.instruction_format,
                                           self.target_style, self.selected_app
                                       ],
                                       outputs=[gallery_ui.prompt],
                                       queue=False)
