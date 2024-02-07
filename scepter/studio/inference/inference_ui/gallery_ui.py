# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import gradio as gr
import numpy as np
from PIL import Image

from scepter.studio.inference.inference_ui.component_names import GalleryUIName
from scepter.studio.utils.uibase import UIBase


class GalleryUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.pipe_manager = pipe_manager
        self.component_names = GalleryUIName(language)
        self.cfg = cfg

    def create_ui(self, *args, **kwargs):
        with gr.Group():
            gr.Markdown(value=self.component_names.gallery_block_name)
            with gr.Row(variant='panel', equal_height=True):
                with gr.Column(scale=2, min_width=0,
                               visible=False) as self.before_refine_panel:
                    self.before_refine_gallery = gr.Gallery(
                        label=self.component_names.
                        gallery_before_refine_output,
                        value=[])
                with gr.Column(scale=2, min_width=0):
                    self.output_gallery = gr.Gallery(
                        label=self.component_names.gallery_diffusion_output,
                        value=[],
                        allow_preview=True,
                        preview=True)
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=17):
                    self.prompt = gr.Textbox(
                        show_label=False,
                        placeholder=self.component_names.prompt_input,
                        elem_id='positive_prompt',
                        container=False,
                        autofocus=True,
                        elem_classes='type_row',
                        lines=1)

                with gr.Column(scale=3, min_width=0):
                    self.generate_button = gr.Button(
                        label='Generate',
                        value=self.component_names.generate,
                        elem_classes='type_row',
                        elem_id='generate_button',
                        visible=True)

    def generate_gallery(self,
                         prompt,
                         mantra_state,
                         tuner_state,
                         control_state,
                         refine_state,
                         diffusion_model,
                         first_stage_model,
                         cond_stage_model,
                         refiner_cond_model,
                         refiner_diffusion_model,
                         tuner_model,
                         tuner_scale,
                         custom_tuner_model,
                         control_model,
                         control_scale,
                         crop_type,
                         control_cond_image,
                         negative_prompt,
                         prompt_prefix,
                         sample,
                         discretization,
                         output_height,
                         output_width,
                         image_number,
                         sample_steps,
                         guide_scale,
                         guide_rescale,
                         refine_strength,
                         refine_sampler,
                         refine_discretization,
                         refine_guide_scale,
                         refine_guide_rescale,
                         style_template,
                         style_negative_template,
                         image_seed,
                         show_jpeg_image=True):
        if control_state and control_cond_image is None:
            raise gr.Error(self.component_names.control_err1)

        current_pipeline = self.pipe_manager.get_pipeline_given_modules({
            'diffusion_model':
            diffusion_model,
            'first_stage_model':
            first_stage_model,
            'cond_stage_model':
            cond_stage_model,
            'refiner_cond_model':
            refiner_cond_model,
            'refiner_diffusion_model':
            refiner_diffusion_model
        })
        now_pipeline = self.pipe_manager.model_level_info[diffusion_model][
            'pipeline'][0]
        used_tuner_model = []
        if not isinstance(tuner_model, list):
            tuner_model = [tuner_model]
        for tuner_m in tuner_model:
            if tuner_m is None or tuner_m == '':
                continue
            if (now_pipeline in self.pipe_manager.model_level_info['tuners']
                    and tuner_m in self.pipe_manager.model_level_info['tuners']
                [now_pipeline]):
                tuner_m = self.pipe_manager.model_level_info['tuners'][
                    now_pipeline][tuner_m]['model_info']
                used_tuner_model.append(tuner_m)
        used_custom_tuner_model = []
        if not isinstance(custom_tuner_model, list):
            custom_tuner_model = [custom_tuner_model]
        for tuner_m in custom_tuner_model:
            if tuner_m is None or tuner_m == '':
                continue
            if (now_pipeline
                    in self.pipe_manager.model_level_info['customized_tuners']
                    and tuner_m in self.pipe_manager.
                    model_level_info['customized_tuners'][now_pipeline]):
                tuner_m = self.pipe_manager.model_level_info[
                    'customized_tuners'][now_pipeline][tuner_m]['model_info']
                used_custom_tuner_model.append(tuner_m)

        if (now_pipeline in self.pipe_manager.model_level_info['controllers']
                and control_model in self.pipe_manager.
                model_level_info['controllers'][now_pipeline]):
            control_model = self.pipe_manager.model_level_info['controllers'][
                now_pipeline][control_model]['model_info']

        prompt_rephrased = style_template.replace(
            '{prompt}',
            prompt) if not style_template == '' and mantra_state else prompt
        prompt_rephrased = f'{prompt_prefix}{prompt_rephrased}' if not prompt_prefix == '' else prompt_rephrased
        negative_prompt_rephrased = negative_prompt + style_negative_template if mantra_state else negative_prompt
        pipeline_input = {
            'prompt': prompt_rephrased,
            'negative_prompt': negative_prompt_rephrased,
            'sample': sample,
            'sample_steps': sample_steps,
            'discretization': discretization,
            'original_size_as_tuple': [int(output_height),
                                       int(output_width)],
            'target_size_as_tuple': [int(output_height),
                                     int(output_width)],
            'crop_coords_top_left': [0, 0],
            'guide_scale': guide_scale,
            'guide_rescale': guide_rescale,
        }
        if refine_state:
            pipeline_input['refine_sampler'] = refine_sampler
            pipeline_input['refine_discretization'] = refine_discretization
            pipeline_input['refine_guide_scale'] = refine_guide_scale
            pipeline_input['refine_guide_rescale'] = refine_guide_rescale
        else:
            refine_strength = 0
        results = current_pipeline(
            pipeline_input,
            num_samples=image_number,
            intermediate_callback=None,
            refine_strength=refine_strength,
            img_to_img_strength=0,
            tuner_model=used_tuner_model +
            used_custom_tuner_model if tuner_state else None,
            tuner_scale=tuner_scale if tuner_state or control_state else None,
            control_model=control_model if control_state else None,
            control_scale=control_scale
            if tuner_state or control_state else None,
            control_cond_image=control_cond_image if control_state else None,
            crop_type=crop_type if control_state else None,
            seed=int(image_seed))
        images = []
        before_images = []
        if 'images' in results:
            images_tensor = results['images'] * 255
            images = [
                Image.fromarray(images_tensor[idx].permute(
                    1, 2, 0).cpu().numpy().astype(np.uint8))
                for idx in range(images_tensor.shape[0])
            ]
        if 'before_refine_images' in results and results[
                'before_refine_images'] is not None:
            before_refine_images_tensor = results['before_refine_images'] * 255
            before_images = [
                Image.fromarray(before_refine_images_tensor[idx].permute(
                    1, 2, 0).cpu().numpy().astype(np.uint8))
                for idx in range(before_refine_images_tensor.shape[0])
            ]
        if 'seed' in results:
            print(results['seed'])
        print(images, before_images)
        if show_jpeg_image:
            save_list = []
            for i, img in enumerate(images):
                save_image = os.path.join(self.cfg.WORK_DIR,
                                          f'cur_gallery_{i}.jpg')
                img.save(save_image)
                save_list.append(save_image)
            images = save_list
        return (
            gr.Column(visible=len(before_images) > 0),
            before_images,
            images,
        )

    def generate_image(self, *args, **kwargs):
        gallery_result = self.generate_gallery(*args, **kwargs)
        before_refine_panel, before_refine_gallery, output_gallery = gallery_result
        return (before_refine_panel, before_refine_gallery, output_gallery[0])

    def set_callbacks(self, inference_ui, model_manage_ui, diffusion_ui,
                      mantra_ui, tuner_ui, refiner_ui, control_ui, **kwargs):

        self.gen_inputs = [
            self.prompt, mantra_ui.state, tuner_ui.state, control_ui.state,
            refiner_ui.state, model_manage_ui.diffusion_model,
            model_manage_ui.first_stage_model,
            model_manage_ui.cond_stage_model, refiner_ui.refiner_cond_model,
            refiner_ui.refiner_diffusion_model, tuner_ui.tuner_model,
            tuner_ui.tuner_scale, tuner_ui.custom_tuner_model,
            control_ui.control_model, control_ui.control_scale,
            control_ui.crop_type, control_ui.cond_image,
            diffusion_ui.negative_prompt, diffusion_ui.prompt_prefix,
            diffusion_ui.sampler, diffusion_ui.discretization,
            diffusion_ui.output_height, diffusion_ui.output_width,
            diffusion_ui.image_number, diffusion_ui.sample_steps,
            diffusion_ui.guide_scale, diffusion_ui.guide_rescale,
            refiner_ui.refine_strength, refiner_ui.refine_sampler,
            refiner_ui.refine_discretization, refiner_ui.refine_guide_scale,
            refiner_ui.refine_guide_rescale, mantra_ui.style_template,
            mantra_ui.style_negative_template, diffusion_ui.image_seed
        ]

        self.gen_outputs = [
            self.before_refine_panel, self.before_refine_gallery,
            self.output_gallery
        ]

        self.generate_button.click(self.generate_gallery,
                                   inputs=self.gen_inputs,
                                   outputs=self.gen_outputs,
                                   queue=True)

        self.prompt.submit(self.generate_gallery,
                           inputs=self.gen_inputs,
                           outputs=self.gen_outputs,
                           queue=True)
