# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import random

import gradio as gr

from scepter.studio.inference.inference_ui.component_names import \
    DiffusionUIName
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


class DiffusionUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.cfg = cfg
        self.pipe_manager = pipe_manager

        default_choices = pipe_manager.module_level_choices
        default_diffusion = default_choices['diffusion_model']['default']
        now_pipeline = pipe_manager.model_level_info[default_diffusion][
            'pipeline'][0]

        self.default_resolutions = pipe_manager.pipeline_level_modules[
            now_pipeline].paras.RESOLUTIONS
        self.default_input = pipe_manager.pipeline_level_modules[
            now_pipeline].input

        self.diffusion_paras = self.load_all_paras()
        # deal with resolution
        self.h_level_dict = {}
        for hw_tuple in self.diffusion_paras.RESOLUTIONS.get('VALUES', []):
            h, w = hw_tuple
            if h not in self.h_level_dict:
                self.h_level_dict[h] = []
            self.h_level_dict[h].append(w)
        self.component_names = DiffusionUIName(language)

    def merge_resolutions(self, ori_h_level_dict, default_resolutions):
        h_level_dict = copy.deepcopy(ori_h_level_dict)
        for res in default_resolutions:
            h, w = res
            if h not in h_level_dict:
                h_level_dict[h] = []
            h_level_dict[h].append(w)
        if len(self.default_resolutions) > 0:
            default_res = default_resolutions[0]
        else:
            default_res = self.diffusion_paras.RESOLUTIONS.DEFAULT
        return h_level_dict, default_res

    def get_default(self, ori_diffusion_paras, cur_default):
        diffusion_paras = copy.deepcopy(ori_diffusion_paras)
        for key in diffusion_paras:
            if key.lower() in cur_default:
                diffusion_paras.get(key).DEFAULT = cur_default.get(key.lower())
                value = diffusion_paras.get(key).get('VALUES')
                if value is not None and cur_default.get(
                        key.lower()) not in value:
                    value.VALUES.append(cur_default.get(key.lower()))
        return diffusion_paras

    def load_all_paras(self):
        diffusion_paras = self.cfg.DIFFUSION_PARAS
        return diffusion_paras

    def create_ui(self, *args, **kwargs):
        self.cur_paras = self.get_default(self.diffusion_paras,
                                          self.default_input)
        self.example_block = gr.Row(equal_height=True, visible=True)
        with gr.Row(equal_height=True):
            self.negative_prompt = gr.Textbox(
                label=self.component_names.negative_prompt,
                show_label=True,
                placeholder=self.component_names.negative_prompt_placeholder,
                info=self.component_names.negative_prompt_description,
                value=self.cur_paras.NEGATIVE_PROMPT.get('DEFAULT', ''),
                lines=2)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                self.prompt_prefix = gr.Textbox(
                    label=self.component_names.prompt_prefix,
                    value=self.cur_paras.PROMPT_PREFIX.get('DEFAULT', ''),
                    interactive=True)
            with gr.Column(scale=2):
                self.sampler = gr.Dropdown(
                    label=self.component_names.sample,
                    choices=self.cur_paras.SAMPLE.get('VALUES', []),
                    value=self.cur_paras.SAMPLE.get('DEFAULT', ''),
                    interactive=True)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                self.discretization = gr.Dropdown(
                    label=self.component_names.discretization,
                    choices=self.cur_paras.DISCRETIZATION.get('VALUES', []),
                    value=self.cur_paras.DISCRETIZATION.get('DEFAULT', ''),
                    interactive=True)
            self.cur_h_level_dict, default_res = self.merge_resolutions(
                self.h_level_dict, self.default_resolutions)
            with gr.Column(scale=1):
                self.output_height = gr.Dropdown(
                    label=self.component_names.resolutions_height,
                    choices=[key for key in self.cur_h_level_dict.keys()],
                    value=default_res[0],
                    interactive=True)
            with gr.Column(scale=1):
                self.output_width = gr.Dropdown(
                    label=self.component_names.resolutions_width,
                    choices=self.cur_h_level_dict[default_res[0]],
                    value=default_res[1],
                    interactive=True)
        with gr.Row(equal_height=True):
            self.image_number = gr.Slider(
                label=self.component_names.image_number,
                minimum=self.cur_paras.SAMPLES.get('MIN', 1),
                maximum=self.cur_paras.SAMPLES.get('MAX', 4),
                step=1,
                value=self.cur_paras.SAMPLES.get('DEFAULT', 1),
                interactive=True)
        with gr.Row(equal_height=True):
            self.sample_steps = gr.Slider(
                label=self.component_names.sample_steps,
                minimum=self.cur_paras.SAMPLE_STEPS.get('MIN', 1),
                maximum=self.cur_paras.SAMPLE_STEPS.get('MAX', 100),
                step=1,
                value=self.cur_paras.SAMPLE_STEPS.get('DEFAULT', 30),
                interactive=True)

            self.guide_scale = gr.Slider(
                label=self.component_names.guide_scale,
                minimum=self.cur_paras.GUIDE_SCALE.get('MIN', 1),
                maximum=self.cur_paras.GUIDE_SCALE.get('MAX', 10),
                step=0.5,
                value=self.cur_paras.GUIDE_SCALE.get('DEFAULT', 7.5),
                interactive=True)
            self.guide_rescale = gr.Slider(
                label=self.component_names.guide_rescale,
                minimum=self.cur_paras.GUIDE_RESCALE.get('MIN', 1),
                maximum=self.cur_paras.GUIDE_RESCALE.get('MAX', 1.0),
                step=0.1,
                value=self.cur_paras.GUIDE_RESCALE.get('DEFAULT', 0.5),
                interactive=True)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                self.seed_random = gr.Checkbox(
                    label=self.component_names.random_seed, value=True)
        with gr.Row(equal_height=True, visible=False) as self.seed_panel:
            with gr.Column(scale=2):
                self.image_seed = gr.Textbox(label=self.component_names.seed,
                                             value=-1,
                                             max_lines=1,
                                             interactive=True)
            with gr.Column(scale=1):
                self.refresh_seed = gr.Button(value=refresh_symbol)
        gallery_ui = kwargs.pop('gallery_ui', None)
        gallery_ui.register_components({
            'negative_prompt': self.negative_prompt,
            'prompt_prefix': self.prompt_prefix,
            'sample': self.sampler,
            'discretization': self.discretization,
            'output_height': self.output_height,
            'output_width': self.output_width,
            'image_number': self.image_number,
            'sample_steps': self.sample_steps,
            'guide_scale': self.guide_scale,
            'guide_rescale': self.guide_rescale,
            'image_seed': self.image_seed,
        })

    def set_callbacks(self, model_manage_ui, **kwargs):
        gallery_ui = kwargs.pop('gallery_ui')
        with self.example_block:
            gr.Examples(label=self.component_names.example_block_name,
                        examples=self.component_names.examples,
                        inputs=gallery_ui.prompt)

        def random_checked(r):
            value = -1
            return (gr.Row(visible=not r), gr.Textbox(value=value))

        def refresh_seed():
            return random.randint(0, 10**12)

        self.seed_random.change(random_checked,
                                inputs=[self.seed_random],
                                outputs=[self.seed_panel, self.image_seed],
                                queue=False,
                                show_progress=False)
        self.refresh_seed.click(refresh_seed,
                                outputs=[self.image_seed],
                                queue=False)

        def change_height(h):
            if h not in self.cur_h_level_dict:
                return gr.Dropdown()
            all_choices = self.cur_h_level_dict[h]
            if len(all_choices) > 0:
                default = all_choices[-1]
            else:
                default = -1
            return gr.Dropdown(choices=all_choices, value=default)

        self.output_height.change(change_height,
                                  inputs=[self.output_height],
                                  outputs=[self.output_width],
                                  queue=False)
