# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import gradio as gr
import numpy as np
import torchvision.transforms as TT
from PIL import Image

from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.distribute import we
from scepter.studio.inference.inference_ui.component_names import ControlUIName
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


class ControlUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.cfg = cfg
        self.pipe_manager = pipe_manager

        controlable_anno = cfg.CONTROLABLE_ANNOTATORS
        self.controlable_annotators = {}
        self.control_choices = []
        self.control_defult = None
        for control_anno in controlable_anno:
            self.controlable_annotators[control_anno.TYPE] = {
                'cfg': control_anno,
                'device': 'offline',
                'model': None
            }
            if control_anno.IS_DEFAULT:
                self.control_defult = control_anno.TYPE
            self.control_choices.append(control_anno.TYPE)
        if self.control_defult is None:
            self.control_defult = self.control_choices[0] if len(
                self.control_choices) > 0 else None

        default_choices = pipe_manager.module_level_choices
        default_diffusion_model = default_choices['diffusion_model']['default']
        default_pipeline = pipe_manager.model_level_info[
            default_diffusion_model]['pipeline'][0]
        if default_pipeline in default_choices[
                'controllers'] and self.control_defult is not None:
            self.controller_choices = default_choices['controllers'][
                default_pipeline][self.control_defult]['choices']
            self.controller_default = default_choices['controllers'][
                default_pipeline][self.control_defult]['default']
        else:
            self.controller_choices = []
            self.controller_default = ''
        self.component_names = ControlUIName(language)

    def load_annotator(self, annotator):
        if annotator['device'] == 'offline':
            annotator['model'] = ANNOTATORS.build(annotator['cfg'])
            annotator['device'] = 'cpu'
        if annotator['device'] == 'cpu':
            annotator['model'] = annotator['model'].to(we.device_id)
            annotator['device'] = we.device_id
        return annotator

    def unload_annotator(self, annotator):
        if not annotator['device'] == 'offline' and not annotator[
                'device'] == 'cpu':
            annotator['model'] = annotator['model'].to('cpu')
            annotator['device'] = 'cpu'
        return annotator

    def create_ui(self, *args, **kwargs):
        self.state = gr.State(value=False)
        with gr.Column(visible=False) as self.tab:
            # gr.Markdown(self.component_names.preprocess)
            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    self.source_image = gr.Image(
                        label=self.component_names.source_image,
                        type='pil',
                        tool='editor',
                        interactive=True)
                with gr.Column(scale=1, min_width=0):
                    self.cond_image = gr.Image(
                        label=self.component_names.cond_image,
                        type='pil',
                        tool='editor',
                        interactive=True)
            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    with gr.Row():
                        self.control_mode = gr.Dropdown(
                            label=self.component_names.control_preprocessor,
                            choices=self.control_choices,
                            value=self.control_defult,
                            interactive=True)
                        self.crop_type = gr.Dropdown(
                            label=self.component_names.crop_type,
                            choices=['CenterCrop', 'NoCrop'],
                            value='CenterCrop',
                            interactive=True)
                        self.control_model = gr.Dropdown(
                            label=self.component_names.control_model,
                            choices=self.controller_choices,
                            value=self.controller_default,
                            interactive=True)
                with gr.Column(scale=1, min_width=0):
                    self.cond_button = gr.Button('Extract')
                    gr.Markdown(self.component_names.direction)

            with gr.Accordion(label=self.component_names.advance_block_name,
                              open=False):
                self.control_scale = gr.Slider(
                    label=self.component_names.control_scale,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=1.0,
                    interactive=True)

            self.example_block = gr.Accordion(
                label=self.component_names.example_block_name, open=True)
        gallery_ui = kwargs.pop('gallery_ui', None)
        gallery_ui.register_components({
            'control_state': self.state,
            'control_model': self.control_model,
            'control_scale': self.control_scale,
            'crop_type': self.crop_type,
            'control_cond_image': self.cond_image,
        })

    def set_callbacks(self, model_manage_ui, diffusion_ui, **kwargs):
        gallery_ui = kwargs.pop('gallery_ui')
        with self.example_block:
            gr.Examples(
                examples=self.component_names.examples,
                inputs=[self.control_mode, self.cond_image, gallery_ui.prompt],
                examples_per_page=20)

        def extract_condition(source_image, control_mode, crop_type,
                              output_height, output_width):
            if control_mode not in self.controlable_annotators:
                gr.Error(self.component_names.control_err1 + ' ' +
                         control_mode)
            annotator = self.controlable_annotators[control_mode]
            annotator = self.load_annotator(annotator)
            if crop_type == 'CenterCrop':
                source_image = TT.Resize(max(output_height,
                                             output_width))(source_image)
                source_image = TT.CenterCrop(
                    (output_height, output_width))(source_image)
            cond_image = annotator['model'](np.array(source_image))
            self.controlable_annotators[control_mode] = self.unload_annotator(
                annotator)
            if cond_image is None:
                gr.Error(self.component_names.control_err2)
            cond_image = Image.fromarray(cond_image)
            return gr.Image(value=cond_image)

        self.cond_button.click(extract_condition,
                               inputs=[
                                   self.source_image, self.control_mode,
                                   self.crop_type, diffusion_ui.output_height,
                                   diffusion_ui.output_width
                               ],
                               outputs=[self.cond_image],
                               queue=False)

        def change_control_mode(control_mode, diffusion_model):
            default_choices = self.pipe_manager.module_level_choices
            now_pipeline = self.pipe_manager.model_level_info[diffusion_model][
                'pipeline'][0]
            if now_pipeline in default_choices[
                    'controllers'] and control_mode in default_choices[
                        'controllers'][now_pipeline]:
                controller_choices = default_choices['controllers'][
                    now_pipeline][control_mode]['choices']
                controller_default = default_choices['controllers'][
                    now_pipeline][control_mode]['default']
            else:
                controller_choices = []
                controller_default = ''
            return gr.Dropdown(choices=controller_choices,
                               value=controller_default)

        self.control_mode.change(
            change_control_mode,
            inputs=[self.control_mode, model_manage_ui.diffusion_model],
            outputs=[self.control_model],
            queue=False)
