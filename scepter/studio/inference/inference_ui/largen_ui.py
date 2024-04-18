# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

import albumentations as A
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.model.utils.data_utils import (box2squre, expand_bbox,
                                                    get_bbox_from_mask,
                                                    pad_to_square)
from scepter.modules.utils.distribute import we
from scepter.studio.inference.inference_ui.component_names import LargenUIName
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


class LargenUI(UIBase):
    def __init__(self, cfg, pipe_manager, is_debug=False, language='en'):
        self.cfg = cfg
        self.pipe_manager = pipe_manager

        self.component_names = LargenUIName(language)

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
        self.tar_image = gr.State(value=None)
        self.tar_mask = gr.State(value=None)
        self.ref_image = gr.State(value=None)
        self.ref_mask = gr.State(value=None)
        self.ref_clip = gr.State(value=None)
        self.task = gr.State(value=self.component_names.tasks[0])
        self.masked_image = gr.State(value=None)
        self.base_image = gr.State(value=None)
        self.extra_sizes = gr.State(value=None)
        self.bbox_yyxx = gr.State(value=None)
        self.image_history = gr.State(value=[])

        with gr.Column(visible=False) as self.tab:
            with gr.Row():
                self.select_app = gr.Dropdown(
                    label=self.component_names.dropdown_name,
                    choices=self.component_names.apps,
                    value=self.component_names.apps[0],
                    type='index')
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=0):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=0):
                            self.scene_image = gr.Image(
                                label=self.component_names.scene_image,
                                type='pil',
                                tool='sketch',
                                source='upload',
                                height=400,
                                interactive=True)
                            self.cache_button = gr.Button(
                                value='Use Last Generated Image', visible=True)
                        with gr.Column(scale=1, min_width=0):
                            self.subject_image = gr.Image(
                                label=self.component_names.subject_image,
                                type='pil',
                                tool='sketch',
                                source='upload',
                                interactive=True,
                                height=400,
                                visible=False)

                    self.gallery = gr.Gallery(label='Image History',
                                              value=[],
                                              columns=1,
                                              rows=1,
                                              height=500)
                    self.clear_button = gr.Button(value='Clear History',
                                                  visible=True)

                with gr.Column(scale=1, min_width=0):
                    self.image_scale = gr.Slider(label='Image Strength',
                                                 minimum=0.0,
                                                 maximum=1.0,
                                                 value=1.0,
                                                 visible=False)
                    self.image_ratio = gr.Slider(minimum=0.5,
                                                 maximum=1.0,
                                                 value=0.75,
                                                 label='Image Resize Ratio',
                                                 visible=True)
                    self.out_direction = gr.Dropdown(
                        label=self.component_names.out_direction_label,
                        choices=self.component_names.out_directions,
                        value=self.component_names.out_directions[0],
                        visible=True)

                    self.proc_button = gr.Button(
                        value=self.component_names.button_name)
                    self.proc_status = gr.Markdown(value='', visible=False)
                    self.task_desc = gr.Markdown(
                        self.component_names.direction, visible=True)

                    self.eg = gr.Column(visible=True)
        gallery_ui = kwargs.pop('gallery_ui', None)
        gallery_ui.register_components({
            'largen_state': self.state,
            'largen_task': self.task,
            'largen_image_scale': self.image_scale,
            'largen_tar_image': self.tar_image,
            'largen_tar_mask': self.tar_mask,
            'largen_masked_image': self.masked_image,
            'largen_ref_image': self.ref_image,
            'largen_ref_mask': self.ref_mask,
            'largen_ref_clip': self.ref_clip,
            'largen_base_image': self.base_image,
            'largen_extra_sizes': self.extra_sizes,
            'largen_bbox_yyxx': self.bbox_yyxx,
            'largen_history': self.image_history
        })

    def set_callbacks(self, model_manage_ui, diffusion_ui, **kwargs):
        def example_data_process(select_app_id, prompt, scene_image,
                                 scene_mask, subject_image, subject_mask,
                                 image_scale, image_ratio, out_direction,
                                 output_height, output_width):
            task = self.component_names.tasks[select_app_id]
            if scene_mask is not None:
                scene_mask = (scene_mask > 128).astype(np.uint8)
            if subject_mask is not None:
                subject_mask = (subject_mask > 128).astype(np.uint8)

            if task == 'Text_Guided_Inpainting':
                data = self.data_preprocess_inpaint(scene_image, scene_mask,
                                                    None, None, False, 1.3,
                                                    output_height,
                                                    output_width)
            elif task == 'Subject_Guided_Inpainting':
                data = self.data_preprocess_inpaint(scene_image, scene_mask,
                                                    subject_image,
                                                    subject_mask, False, 1.3,
                                                    output_height,
                                                    output_width)
            elif task == 'Text_Subject_Guided_Inpainting':
                data = self.data_preprocess_inpaint(scene_image, scene_mask,
                                                    subject_image,
                                                    subject_mask, True, 1.3,
                                                    output_height,
                                                    output_width)
            elif task == 'Text_Guided_Outpainting':
                data = self.data_preprocess_outpaint(scene_image,
                                                     out_direction,
                                                     image_ratio,
                                                     output_height,
                                                     output_width)

            subject_image_show = None if subject_image is None else Image.fromarray(
                subject_image.astype(np.uint8))
            return *data, gr.update(value='Data Process Succeed!', visible=True), \
                gr.update(value=Image.fromarray(scene_image.astype(np.uint8))), \
                gr.update(value=subject_image_show), \
                task, gr.update(value=self.component_names.apps[select_app_id]), \
                gr.update(value=prompt), gr.update(value=image_scale), gr.update(value=image_ratio)

        gallery_ui = kwargs.pop('gallery_ui')
        with self.eg:
            self.scene_image_eg = gr.Image(
                label=self.component_names.scene_image,
                type='numpy',
                visible=False)  # noqa
            self.scene_mask_eg = gr.Image(
                label=self.component_names.scene_mask,
                type='numpy',
                image_mode='L',
                visible=False)  # noqa
            self.subject_image_eg = gr.Image(
                label=self.component_names.subject_image,
                type='numpy',
                visible=False)  # noqa
            self.subject_mask_eg = gr.Image(
                label=self.component_names.subject_mask,
                type='numpy',
                image_mode='L',
                visible=False)  # noqa
            self.prompt = gr.Textbox(label=self.component_names.prompt,
                                     visible=False)
            self.examples = gr.Examples(examples=self.component_names.examples,
                                        inputs=[
                                            self.select_app,
                                            self.prompt,
                                            self.scene_image_eg,
                                            self.scene_mask_eg,
                                            self.subject_image_eg,
                                            self.subject_mask_eg,
                                            self.image_scale,
                                            self.image_ratio,
                                            self.out_direction,
                                            diffusion_ui.output_height,
                                            diffusion_ui.output_width,
                                        ],
                                        outputs=[
                                            self.tar_image,
                                            self.tar_mask,
                                            self.masked_image,
                                            self.ref_image,
                                            self.ref_mask,
                                            self.ref_clip,
                                            self.base_image,
                                            self.extra_sizes,
                                            self.bbox_yyxx,
                                            self.proc_status,
                                            self.scene_image,
                                            self.subject_image,
                                            self.task,
                                            self.select_app,
                                            gallery_ui.prompt,
                                            self.image_scale,
                                            self.image_ratio,
                                        ],
                                        fn=example_data_process,
                                        cache_examples=False,
                                        run_on_click=True)

        def change_app(select_app_id):
            select_task = self.component_names.tasks[select_app_id]
            return gr.update(visible=('Subject' in select_task)), \
                gr.update(visible=('Subject' in select_task)), \
                gr.update(visible=('Outpainting' in select_task)), \
                gr.update(visible=('Outpainting' in select_task)), select_task

        self.select_app.change(change_app,
                               inputs=[self.select_app],
                               outputs=[
                                   self.subject_image, self.image_scale,
                                   self.image_ratio, self.out_direction,
                                   self.task
                               ],
                               queue=False)

        def read_gallery_image(gallery):
            if len(gallery) == 0:
                last_image = None
            else:
                last_image = gallery[-1]['name']
            return gr.update(value=last_image)

        self.cache_button.click(read_gallery_image,
                                inputs=[self.gallery],
                                outputs=[self.scene_image])

        def clear_gallery(image_history, gallery):
            image_history.clear()
            gallery.clear()
            return image_history, gallery

        self.clear_button.click(fn=clear_gallery,
                                inputs=[self.image_history, self.gallery],
                                outputs=[self.image_history, self.gallery])

        def data_process(scene_image, subject_image, task, image_ratio,
                         out_direction, output_height, output_width):
            tar_image = scene_image['image'].convert('RGB')
            tar_mask = scene_image['mask'].convert('L')
            tar_image = np.asarray(tar_image)
            tar_mask = np.asarray(tar_mask)
            tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

            if task == 'Text_Guided_Inpainting':
                data = self.data_preprocess_inpaint(tar_image, tar_mask, None,
                                                    None, False, 1.3,
                                                    output_height,
                                                    output_width)
            elif task == 'Subject_Guided_Inpainting':
                ref_image = subject_image['image'].convert('RGB')
                ref_mask = subject_image['mask'].convert('L')
                ref_image = np.asarray(ref_image)
                ref_mask = np.asarray(ref_mask)
                ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)
                data = self.data_preprocess_inpaint(tar_image, tar_mask,
                                                    ref_image, ref_mask, False,
                                                    1.3, output_height,
                                                    output_width)
            elif task == 'Text_Subject_Guided_Inpainting':
                ref_image = subject_image['image'].convert('RGB')
                ref_mask = subject_image['mask'].convert('L')
                ref_image = np.asarray(ref_image)
                ref_mask = np.asarray(ref_mask)
                ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)
                data = self.data_preprocess_inpaint(tar_image, tar_mask,
                                                    ref_image, ref_mask, True,
                                                    1.3, output_height,
                                                    output_width)
            elif task == 'Text_Guided_Outpainting':
                data = self.data_preprocess_outpaint(tar_image, out_direction,
                                                     image_ratio,
                                                     output_height,
                                                     output_width)

            return *data, gr.update(value='Data Process Succeed!',
                                    visible=True)

        self.proc_button.click(data_process,
                               inputs=[
                                   self.scene_image, self.subject_image,
                                   self.task, self.image_ratio,
                                   self.out_direction,
                                   diffusion_ui.output_height,
                                   diffusion_ui.output_width
                               ],
                               outputs=[
                                   self.tar_image,
                                   self.tar_mask,
                                   self.masked_image,
                                   self.ref_image,
                                   self.ref_mask,
                                   self.ref_clip,
                                   self.base_image,
                                   self.extra_sizes,
                                   self.bbox_yyxx,
                                   self.proc_status,
                               ])

    def data_preprocess_inpaint(self, tar_image, tar_mask, ref_image, ref_mask,
                                use_rectangle_mask, tar_crop_ratio,
                                output_height, output_width):
        tar_mask = np.expand_dims(tar_mask, 2).astype(np.float32)

        # Zoom-In
        tar_yyxx = get_bbox_from_mask(tar_mask)
        tar_yyxx_crop = expand_bbox(tar_mask, tar_yyxx, ratio=tar_crop_ratio)
        tar_yyxx_crop = box2squre(tar_mask, tar_yyxx_crop)
        y1, y2, x1, x2 = tar_yyxx_crop
        crop_tar_image = tar_image[y1:y2, x1:x2, :]
        crop_tar_mask = tar_mask[y1:y2, x1:x2, :]
        H1, W1 = crop_tar_image.shape[:2]

        if use_rectangle_mask:
            tar_bbox_yyxx = get_bbox_from_mask(crop_tar_mask)
            y1, y2, x1, x2 = tar_bbox_yyxx
            crop_tar_mask[y1:y2, x1:x2] = 1

        crop_tar_image, pad1, pad2 = pad_to_square(crop_tar_image.astype(
            np.uint8),
                                                   pad_value=0)
        crop_tar_mask, _, _ = pad_to_square(crop_tar_mask, pad_value=0)
        H2, W2 = crop_tar_image.shape[:2]

        aug_tar_image = cv2.resize(crop_tar_image.astype(np.uint8),
                                   (output_width, output_height))
        aug_tar_mask = cv2.resize(crop_tar_mask, (output_width, output_height))

        final_tar_image = TF.to_tensor(aug_tar_image)
        final_tar_image = TF.normalize(final_tar_image,
                                       mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])
        final_tar_mask = TF.to_tensor((aug_tar_mask > 0.5).astype(np.float32))

        masked_image = final_tar_image.clone()
        masked_image = masked_image * (1 - final_tar_mask)

        final_tar_image = final_tar_image.unsqueeze(0)
        final_tar_mask = final_tar_mask.unsqueeze(0)
        masked_image = masked_image.unsqueeze(0)

        if ref_image is not None and ref_mask is not None:
            ref_mask = np.expand_dims(ref_mask, 2).astype(np.float32)
            # background-free
            ref_image = ref_image * ref_mask + np.ones_like(
                ref_image) * 255. * (1 - ref_mask)

            ref_yyxx = get_bbox_from_mask(ref_mask)
            y1, y2, x1, x2 = ref_yyxx

            crop_ref_image_i = ref_image[y1:y2, x1:x2, :]
            crop_ref_mask_i = ref_mask[y1:y2, x1:x2, :]

            h, w = crop_ref_mask_i.shape[:2]
            ref_expand_size = int(max(h, w) * 1.02)
            pad_op = A.PadIfNeeded(ref_expand_size,
                                   ref_expand_size,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255),
                                   mask_value=0)
            out = pad_op(image=crop_ref_image_i, mask=crop_ref_mask_i)
            crop_ref_image = out['image']

            to_clip_input = T.Compose([
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                            std=(0.26862954, 0.26130258, 0.27577711)),
            ])
            ref_clip = to_clip_input(crop_ref_image.astype(np.uint8))

            output_size = max(output_height, output_width)
            ref_resize_op = A.Compose([
                A.LongestMaxSize(output_size),
                A.PadIfNeeded(output_size,
                              output_size,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=(255, 255, 255),
                              mask_value=0),
            ])
            aug_out = ref_resize_op(image=crop_ref_image_i.astype(np.uint8),
                                    mask=crop_ref_mask_i)
            aug_ref_image = aug_out['image']
            aug_ref_mask = aug_out['mask']

            final_ref_image = TF.to_tensor(aug_ref_image)
            final_ref_image = TF.normalize(final_ref_image,
                                           mean=[0.5, 0.5, 0.5],
                                           std=[0.5, 0.5, 0.5])
            final_ref_mask = TF.to_tensor(aug_ref_mask)

            final_ref_image = final_ref_image.unsqueeze(0)
            final_ref_mask = final_ref_mask.unsqueeze(0)
            ref_clip = ref_clip.unsqueeze(0)
        else:
            final_ref_image = None
            final_ref_mask = None
            ref_clip = None

        return final_tar_image, final_tar_mask, masked_image, final_ref_image, final_ref_mask, ref_clip, \
            TF.to_tensor(tar_image), torch.LongTensor([H1, W1, H2, W2, pad1, pad2]), torch.LongTensor(tar_yyxx_crop)

    def data_preprocess_outpaint(self, tar_image, direction, img_ratio,
                                 output_height, output_width):
        oh, ow = output_height, output_width
        h, w = tar_image.shape[:2]
        ratio = max(h / (oh * img_ratio), w / (ow * img_ratio))

        ih, iw = int(h / ratio), int(w / ratio)

        masked_image = np.zeros((oh, ow, 3), dtype=np.uint8)
        mask = np.zeros((oh, ow, 1))

        if direction in ['CenterAround', '中心向外']:
            y1, x1 = (oh - ih) // 2, (ow - iw) // 2
        elif direction in ['RightDown', '右下']:
            y1, x1 = 0, 0
        elif direction in ['LeftDown', '左下']:
            y1, x1 = 0, ow - iw
        elif direction in ['RightUp', '右上']:
            y1, x1 = oh - ih, 0
        elif direction in ['LeftUp', '左上']:
            y1, x1 = oh - ih, ow - iw
        else:
            y1, x1 = 0, 0

        tar_image = cv2.resize(tar_image.astype(np.uint8), (iw, ih))
        masked_image[y1:y1 + ih, x1:x1 + iw] = tar_image
        mask[y1 + 5:y1 + ih - 5, x1 + 5:x1 + iw - 5] = 1

        final_tar_image = TF.to_tensor(masked_image.astype(np.uint8))
        final_tar_image = TF.normalize(final_tar_image,
                                       mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])
        final_tar_mask = TF.to_tensor(((1.0 - mask) > 0.5).astype(np.float32))

        masked_image = final_tar_image.clone()
        masked_image = masked_image * (1 - final_tar_mask)

        final_tar_image = final_tar_image.unsqueeze(0)
        final_tar_mask = final_tar_mask.unsqueeze(0)
        masked_image = masked_image.unsqueeze(0)

        return final_tar_image, final_tar_mask, masked_image, None, None, None, None, None, None
