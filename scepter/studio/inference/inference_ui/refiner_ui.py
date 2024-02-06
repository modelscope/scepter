# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import gradio as gr

from scepter.studio.inference.inference_ui.component_names import RefinerUIName
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


class RefinerUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.cfg = cfg
        self.pipe_manager = pipe_manager
        self.diffusion_paras = self.load_all_paras()
        self.component_names = RefinerUIName(language)

    def load_all_paras(self):
        diffusion_paras = self.cfg.DIFFUSION_PARAS
        return diffusion_paras

    def create_ui(self, *args, **kwargs):
        self.state = gr.State(value=False)
        with gr.Group(visible=False) as self.tab:
            with gr.Row(equal_height=True):
                with gr.Column(variant='panel', scale=1, min_width=0):
                    self.refiner_diffusion_model = gr.Dropdown(
                        label=self.component_names.refine_diffusion_model,
                        choices=[],
                        value=None,
                        interactive=True)
                with gr.Column(variant='panel', scale=1, min_width=0):
                    self.refiner_cond_model = gr.Dropdown(
                        label=self.component_names.refine_cond_model,
                        choices=[],
                        value=None,
                        interactive=True)
            with gr.Row(equal_height=True):
                self.refine_strength = gr.Slider(
                    label=self.component_names.refine_strength,
                    minimum=self.diffusion_paras.REFINE_STRENGTH.get(
                        'MIN', 0.0),
                    maximum=self.diffusion_paras.REFINE_STRENGTH.get(
                        'MAX', 1.0),
                    step=0.05,
                    value=self.diffusion_paras.REFINE_STRENGTH.get(
                        'DEFAULT', 7.5),
                    interactive=True)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    self.refine_sampler = gr.Dropdown(
                        label=self.component_names.refine_sample,
                        choices=self.diffusion_paras.REFINE_SAMPLERS.get(
                            'VALUES', []),
                        value=self.diffusion_paras.REFINE_SAMPLERS.get(
                            'DEFAULT', ''),
                        interactive=True)
                with gr.Column(scale=1):
                    self.refine_discretization = gr.Dropdown(
                        label=self.component_names.refine_discretization,
                        choices=self.diffusion_paras.REFINE_DISCRETIZATION.get(
                            'VALUES', []),
                        value=self.diffusion_paras.REFINE_DISCRETIZATION.get(
                            'DEFAULT', ''),
                        interactive=True)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    self.refine_guide_scale = gr.Slider(
                        label=self.component_names.refine_guide_scale,
                        minimum=self.diffusion_paras.REFINE_GUIDE_SCALE.get(
                            'MIN', 1),
                        maximum=self.diffusion_paras.REFINE_GUIDE_SCALE.get(
                            'MAX', 10),
                        step=0.5,
                        value=self.diffusion_paras.REFINE_GUIDE_SCALE.get(
                            'DEFAULT', 7.5),
                        interactive=True)
                with gr.Column(scale=1):
                    self.refine_guide_rescale = gr.Slider(
                        label=self.component_names.refine_guide_rescale,
                        minimum=self.diffusion_paras.REFINE_GUIDE_RESCALE.get(
                            'MIN', 1),
                        maximum=self.diffusion_paras.REFINE_GUIDE_RESCALE.get(
                            'MAX', 1.0),
                        step=0.1,
                        value=self.diffusion_paras.REFINE_GUIDE_RESCALE.get(
                            'DEFAULT', 0.5),
                        interactive=True)

    def set_callbacks(self, model_manage_ui, **kwargs):
        pass
