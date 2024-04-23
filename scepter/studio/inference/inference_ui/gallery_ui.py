# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict

import gradio as gr
import numpy as np
from PIL import Image

from scepter.modules.utils.file_system import FS
from scepter.studio.inference.inference_ui.component_names import GalleryUIName
from scepter.studio.utils.uibase import UIBase


class GalleryUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.pipe_manager = pipe_manager
        self.component_names = GalleryUIName(language)
        self.cfg = cfg
        self.work_dir = cfg.WORK_DIR
        self.local_work_dir, _ = FS.map_to_local(self.work_dir)
        os.makedirs(self.local_work_dir, exist_ok=True)
        self.component_mapping = OrderedDict()
        self.manager = None

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
                        submit_on_enter=True,
                        lines=1)

                with gr.Column(scale=3, min_width=0):
                    self.generate_button = gr.Button(
                        label='Generate',
                        value=self.component_names.generate,
                        elem_classes='type_row',
                        elem_id='generate_button',
                        visible=True)
        self.register_components({'prompt': self.prompt})

    def register_components(self, components):
        common_keys = self.component_mapping.keys() & components.keys()
        assert len(
            common_keys) == 0, f'Component key already exist: {common_keys}'
        self.component_mapping.update(components)

    def generate_gallery(self, *args, show_jpeg_image=True):
        # unload all image preprocess model.
        if self.manager is not None:
            if (hasattr(self.manager, 'preprocess') and hasattr(
                    self.manager.preprocess.dataset_gallery.processors_manager,
                    'dynamic_unload')):
                self.manager.preprocess.dataset_gallery.processors_manager.dynamic_unload(
                )

        def pipeline_init(args):
            return self.pipe_manager.get_pipeline_given_modules(args)

        def control_init(args):
            control_state, tuner_state = args['control_state'], args[
                'tuner_state']
            control_cond_image = args.pop('control_cond_image')
            control_model = args.pop('control_model')

            if control_state and control_cond_image is None:
                raise gr.Error(self.component_names.control_err1)

            now_pipeline = self.pipe_manager.model_level_info[
                args['diffusion_model']]['pipeline'][0]
            if (now_pipeline
                    in self.pipe_manager.model_level_info['controllers']
                    and control_model in self.pipe_manager.
                    model_level_info['controllers'][now_pipeline]):
                control_model = self.pipe_manager.model_level_info[
                    'controllers'][now_pipeline][control_model]['model_info']

            args.update({
                'control_model':
                control_model if control_state else None,
                'control_scale':
                args.pop('control_scale')
                if tuner_state or control_state else None,
                'control_cond_image':
                control_cond_image if control_state else None,
                'crop_type':
                args.pop('crop_type') if control_state else None
            })

        def tuner_init(args):
            control_state, tuner_state = args['control_state'], args[
                'tuner_state']
            tuner_model = args.pop('tuner_model')
            custom_tuner_model = args.pop('custom_tuner_model')

            now_pipeline = self.pipe_manager.model_level_info[
                args['diffusion_model']]['pipeline'][0]
            used_tuner_model = []
            if not isinstance(tuner_model, list):
                tuner_model = [tuner_model]
            for tuner_m in tuner_model:
                if tuner_m is None or tuner_m == '':
                    continue
                if now_pipeline in self.pipe_manager.model_level_info['tuners'] and \
                   tuner_m in self.pipe_manager.model_level_info['tuners'][now_pipeline]:
                    tuner_m = self.pipe_manager.model_level_info['tuners'][
                        now_pipeline][tuner_m]['model_info']
                    used_tuner_model.append(tuner_m)
            used_custom_tuner_model = []
            if not isinstance(custom_tuner_model, list):
                custom_tuner_model = [custom_tuner_model]
            for tuner_m in custom_tuner_model:
                if tuner_m is None or tuner_m == '':
                    continue
                if (now_pipeline in
                        self.pipe_manager.model_level_info['customized_tuners']
                        and tuner_m in self.pipe_manager.
                        model_level_info['customized_tuners'][now_pipeline]):
                    tuner_m = self.pipe_manager.model_level_info[
                        'customized_tuners'][now_pipeline][tuner_m][
                            'model_info']
                    used_custom_tuner_model.append(tuner_m)
            args.update({
                'tuner_model':
                used_tuner_model +
                used_custom_tuner_model if tuner_state else None,
                'tuner_scale':
                args.pop('tuner_scale')
                if tuner_state or control_state else None,
            })

        def input_init(args):
            mantra_state = args['mantra_state']
            prompt = args.pop('prompt')
            negative_prompt = args.pop('negative_prompt')
            prompt_prefix = args.pop('prompt_prefix')
            style_template = args.pop('style_template')
            size = [
                int(args.pop('output_height')),
                int(args.pop('output_width'))
            ]

            prompt_rephrased = style_template.replace(
                '{prompt}', prompt
            ) if not style_template == '' and mantra_state else prompt

            pipeline_input = {
                'prompt':
                f'{prompt_prefix}{prompt_rephrased}'
                if not prompt_prefix == '' else prompt_rephrased,
                'negative_prompt':
                negative_prompt + args.pop('style_negative_template')
                if mantra_state else negative_prompt,
                'sample':
                args.pop('sample'),
                'sample_steps':
                args.pop('sample_steps'),
                'discretization':
                args.pop('discretization'),
                'original_size_as_tuple':
                size,
                'target_size_as_tuple':
                size,
                'crop_coords_top_left': [0, 0],
                'guide_scale':
                args.pop('guide_scale'),
                'guide_rescale':
                args.pop('guide_rescale')
            }
            args.update({'input': pipeline_input})

        def appedix_init(args):
            args.update({
                'num_samples': args.pop('image_number'),
                'intermediate_callback': None,
                'img_to_img_strength': 0,
                'seed': int(args.pop('image_seed')),
            })

        def load_init(args):
            cur_pipe_name = self.pipe_manager.model_level_info[
                args['diffusion_model']]['pipeline'][0]
            for sub_name, sub_pipe in self.pipe_manager.pipeline_level_modules.items(
            ):
                if sub_name == cur_pipe_name:
                    continue
                if len(sub_pipe.loaded_model) > 0:
                    sub_pipe.dynamic_unload(name='all')
                    print(f'Unloading {sub_name} modules')

        args = dict(zip(self.component_mapping.keys(), args))
        largen_history = args.pop('largen_history')

        current_pipeline = pipeline_init(args)
        control_init(args)
        tuner_init(args)
        input_init(args)
        appedix_init(args)
        load_init(args)
        results = current_pipeline(**args)

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

        if args['largen_state']:
            largen_history.extend(images)
            if len(largen_history) > 5:
                largen_history = largen_history[-5:]

        if show_jpeg_image:
            save_list = []
            for i, img in enumerate(images):
                save_image = os.path.join(self.local_work_dir,
                                          f'cur_gallery_{i}.jpg')
                img.save(save_image)
                save_list.append(save_image)
            images = save_list

        return (
            gr.Column(visible=len(before_images) > 0),
            before_images,
            images,
            largen_history,
            gr.update(value=largen_history),
        )

    def generate_image(self, *args, **kwargs):
        gallery_result = self.generate_gallery(*args, **kwargs)
        before_refine_panel, before_refine_gallery, output_gallery, _ = gallery_result
        return (before_refine_panel, before_refine_gallery, output_gallery[0])

    def set_callbacks(self,
                      inference_ui,
                      model_manage_ui,
                      diffusion_ui,
                      mantra_ui,
                      tuner_ui,
                      refiner_ui,
                      control_ui,
                      largen_ui,
                      manager=None,
                      **kwargs):
        self.manager = manager
        self.gen_inputs = list(self.component_mapping.values())
        print(self.gen_inputs, len(self.gen_inputs))
        self.gen_outputs = [
            self.before_refine_panel,
            self.before_refine_gallery,
            self.output_gallery,
            largen_ui.image_history,
            largen_ui.gallery,
        ]

        self.generate_button.click(self.generate_gallery,
                                   inputs=self.gen_inputs,
                                   outputs=self.gen_outputs,
                                   queue=True)

        self.prompt.submit(self.generate_gallery,
                           inputs=self.gen_inputs,
                           outputs=self.gen_outputs,
                           queue=True)
