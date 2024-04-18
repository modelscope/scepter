# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import gradio as gr

from scepter.studio.inference.inference_ui.component_names import \
    ModelManageUIName
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


class ModelManageUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.pipe_manager = pipe_manager
        self.default_choices = pipe_manager.module_level_choices
        self.component_names = ModelManageUIName(language)

    def create_ui(self, *args, **kwargs):
        self.diffusion_state = gr.State(
            value=self.default_choices['diffusion_model']['default'])
        with gr.Group():
            gr.Markdown(value=self.component_names.model_block_name)
            with gr.Row(variant='panel', equal_height=True):
                with gr.Column(scale=1, min_width=0) as self.diffusion_panel:
                    self.diffusion_model = gr.Dropdown(
                        label=self.component_names.diffusion_model,
                        choices=self.default_choices['diffusion_model']
                        ['choices'],
                        value=self.default_choices['diffusion_model']
                        ['default'],
                        interactive=True)
                with gr.Column(scale=1, min_width=0) as self.first_stage_panel:
                    self.first_stage_model = gr.Dropdown(
                        label=self.component_names.first_stage_model,
                        choices=self.default_choices['first_stage_model']
                        ['choices'],
                        value=self.default_choices['first_stage_model']
                        ['default'],
                        interactive=False)
                with gr.Column(scale=1, min_width=0) as self.cond_stage_panel:
                    self.cond_stage_model = gr.Dropdown(
                        label=self.component_names.cond_stage_model,
                        choices=self.default_choices['cond_stage_model']
                        ['choices'],
                        value=self.default_choices['cond_stage_model']
                        ['default'],
                        interactive=False)
            # with gr.Accordion(
            #         label=self.component_names.postprocess_model_name,
            #         open=False):
            # with gr.Row(equal_height=True):
            #     self.advance_postprocess_checkbox = gr.CheckboxGroup(
            #         # choices=['Refiners', 'Tuners'], show_label=False)
            #         choices=['Tuners'], show_label=False)
            # with gr.Row(equal_height=True,
            #             visible=False) as self.refine_diffusion_panel:
            #     with gr.Column(variant='panel', scale=1, min_width=0):
            #         self.refiner_diffusion_model = gr.Dropdown(
            #             label=self.component_names.refine_diffusion_model,
            #             choices=self.default_choices[
            #                 'refiner_diffusion_model']['choices'],
            #             value=self.default_choices[
            #                 'refiner_diffusion_model']['default'],
            #             interactive=True)
            #     with gr.Column(variant='panel', scale=1, min_width=0):
            #         self.refiner_cond_model = gr.Dropdown(
            #             label=self.component_names.refine_cond_model,
            #             choices=self.default_choices['refiner_cond_model']
            #             ['choices'],
            #             value=self.default_choices['refiner_cond_model']
            #             ['default'],
            #             interactive=True)
            # with gr.Column(variant='panel', scale=1, min_width=0):
            #     self.tuner_button = gr.Button(value=refresh_symbol)
            #
            #     def refresh_choices():
            #         return gr.update(choices=get_tuner_choices())
            #
            #     self.tuner_button.click(refresh_choices, [],
            #                             [self.tuner_model])
            # with gr.Column(variant='panel', scale=4, min_width=0):
            #     with gr.Group() as self.tuners_group:
            #         with gr.Row(variant='panel') as self.tuners_panel:
            #             with gr.Column(
            #                     scale=1,
            #                     min_width=0) as self.tuners_management:
            #                 self.load_Lora_tuner_btn = gr.Button(
            #                     value=self.component_names.
            #                     load_lora_tuner)
            #                 self.load_swift_tuner_btn = gr.Button(
            #                     value=self.component_names.
            #                     load_swift_tuner)
            #             with gr.Column(scale=1,
            #                            min_width=0) as self.load_panel:
            #                 self.tuner_name = gr.Text(
            #                     label='tuner_name')
            #         with gr.Row(variant='panel') as self.tuner_info:
            #             with gr.Accordion(label=self.component_names.
            #                               postprocess_model_name,
            #                               open=False):
            #                 self.tuner_name = gr.Text(
            #                     label='tuner_name')
        gallery_ui = kwargs.pop('gallery_ui', None)
        gallery_ui.register_components({
            'diffusion_model':
            self.diffusion_model,
            'first_stage_model':
            self.first_stage_model,
            'cond_stage_model':
            self.cond_stage_model,
        })

    def set_callbacks(self, diffusion_ui, tuner_ui, control_ui, mantra_ui,
                      **kwargs):
        # def select_refine_tuner(all_select, evt: gr.SelectData):
        #     if 'Refiners' in all_select:
        #         refine_panel = gr.Row(visible=True)
        #         refine_tab = gr.Group(visible=True)
        #         refine_state = True
        #     else:
        #         refine_panel = gr.Row(visible=False)
        #         refine_tab = gr.Group(visible=False)
        #         refine_state = False
        #     # if 'Tuners' in all_select:
        #     #     tuner_panel = gr.Row(visible=True)
        #     # else:
        #     #     tuner_panel = gr.Row(visible=False)
        #     return refine_panel, refine_tab, refine_state
        #
        # self.advance_postprocess_checkbox.select(
        #     select_refine_tuner,
        #     inputs=[self.advance_postprocess_checkbox],
        #     outputs=[
        #         self.refine_diffusion_panel, self.tuner_choice_panel,
        #         advance_ui.refine_tab, advance_ui.refine_state
        #     ])
        def diffusion_model_change(diffusion_state, diffusion_model,
                                   control_mode):
            if diffusion_state != diffusion_model:
                last_pipline = self.pipe_manager.model_level_info[
                    diffusion_state]['pipeline'][0]
                last_pipeline_ins = self.pipe_manager.pipeline_level_modules[
                    last_pipline]
                last_pipeline_ins.dynamic_unload(name='all')
            now_pipeline = self.pipe_manager.model_level_info[diffusion_model][
                'pipeline'][0]
            pipeline_ins = self.pipe_manager.pipeline_level_modules[
                now_pipeline]
            pipeline_ins.dynamic_load(name='all')
            all_module_name = {}
            for module_name in self.pipe_manager.module_list:
                module = getattr(pipeline_ins, module_name)
                if module is None:
                    continue
                model_name = f"{now_pipeline}_{module['name']}"
                all_module_name[module_name] = model_name
            tunner_choices = []
            if now_pipeline in self.default_choices['tuners']:
                tunner_choices = self.default_choices['tuners'][now_pipeline][
                    'choices']
            custom_tunner_choices = []
            custom_tunner_default = []
            if now_pipeline in self.default_choices.get(
                    'customized_tuners', []):
                custom_tunner_choices = self.default_choices[
                    'customized_tuners'][now_pipeline]['choices']
                custom_tunner_default = self.default_choices[
                    'customized_tuners'][now_pipeline]['default']
                if isinstance(custom_tunner_default, str):
                    custom_tunner_default = [custom_tunner_default]

            if now_pipeline in self.default_choices[
                    'controllers'] and control_mode in self.default_choices[
                        'controllers'][now_pipeline]:
                controller_choices = self.default_choices['controllers'][
                    now_pipeline][control_mode]['choices']
                controller_default = self.default_choices['controllers'][
                    now_pipeline][control_mode]['default']
            else:
                controller_choices = []
                controller_default = ''

            default_resolutions = self.pipe_manager.pipeline_level_modules[
                now_pipeline].paras.RESOLUTIONS
            h_level_dict, default_res = diffusion_ui.merge_resolutions(
                diffusion_ui.h_level_dict, default_resolutions)
            diffusion_ui.cur_h_level_dict = h_level_dict

            default_input = self.pipe_manager.pipeline_level_modules[
                now_pipeline].input
            cur_paras = diffusion_ui.get_default(diffusion_ui.diffusion_paras,
                                                 default_input)
            diffusion_ui.cur_paras = cur_paras
            return (
                diffusion_model,
                gr.Dropdown(value=all_module_name['first_stage_model']),
                gr.Dropdown(value=all_module_name['cond_stage_model']),
                gr.Dropdown(choices=tunner_choices, value=[]),
                gr.Dropdown(choices=custom_tunner_choices,
                            value=custom_tunner_default),
                gr.Dropdown(choices=controller_choices,
                            value=controller_default),
                gr.Dropdown(choices=mantra_ui.all_styles.get(now_pipeline, []),
                            value=[]),
                gr.Textbox(choices=cur_paras.NEGATIVE_PROMPT.get('VALUES', []),
                           value=cur_paras.NEGATIVE_PROMPT.get('DEFAULT', '')),
                gr.Textbox(choices=cur_paras.PROMPT_PREFIX.get('VALUES', []),
                           value=cur_paras.PROMPT_PREFIX.get('DEFAULT', '')),
                gr.Dropdown(choices=[key for key in h_level_dict.keys()],
                            value=default_res[0]),
                gr.Dropdown(choices=cur_paras.SAMPLE.get('VALUES', []),
                            value=cur_paras.SAMPLE.get('DEFAULT', '')),
                gr.Dropdown(choices=cur_paras.DISCRETIZATION.get('VALUES', []),
                            value=cur_paras.DISCRETIZATION.get('DEFAULT', '')),
                gr.Slider(value=cur_paras.SAMPLE_STEPS.get('DEFAULT', 30)),
                gr.Slider(value=cur_paras.GUIDE_SCALE.get('DEFAULT', 7.5)),
                gr.Slider(value=cur_paras.GUIDE_RESCALE.get('DEFAULT', 0.5)))

        self.diffusion_model.change(
            diffusion_model_change,
            inputs=[
                self.diffusion_state, self.diffusion_model,
                control_ui.control_mode
            ],
            outputs=[
                self.diffusion_state, self.first_stage_model,
                self.cond_stage_model, tuner_ui.tuner_model,
                tuner_ui.custom_tuner_model, control_ui.control_model,
                mantra_ui.style, diffusion_ui.negative_prompt,
                diffusion_ui.prompt_prefix, diffusion_ui.output_height,
                diffusion_ui.sampler, diffusion_ui.discretization,
                diffusion_ui.sample_steps, diffusion_ui.guide_scale,
                diffusion_ui.guide_rescale
            ],
            queue=True)
