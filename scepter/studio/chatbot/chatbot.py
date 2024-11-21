# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import base64
import copy
import csv
import glob
import io
import os
import random
import re
import string
import sys
import threading
import warnings

import cv2
import gradio as gr
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from scepter.modules.inference.ace_inference import ACEInference
from scepter.modules.utils.config import Config
from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS
from scepter.studio.utils.env import init_env
from importlib.metadata import version

from .example import get_examples
from .utils import load_image

csv.field_size_limit(sys.maxsize)

refresh_sty = '\U0001f504'  # 🔄
clear_sty = '\U0001f5d1'  # 🗑️
upload_sty = '\U0001f5bc'  # 🖼️
sync_sty = '\U0001f4be'  # 💾
chat_sty = '\U0001F4AC'  # 💬
video_sty = '\U0001f3a5'  # 🎥

lock = threading.Lock()


class ChatBotUI(object):
    def __init__(self,
                 cfg_general_file,
                 is_debug=False,
                 language='en',
                 root_work_dir='./'):
        try:
            from diffusers import CogVideoXImageToVideoPipeline
            from diffusers.utils import export_to_video
        except Exception as e:
            print(f"Import diffusers failed, please install or upgrade diffusers. Error information: {e}")
        if isinstance(cfg_general_file, str):
            cfg = Config(cfg_file=cfg_general_file)
        else:
            cfg = cfg_general_file
        cfg.WORK_DIR = os.path.join(root_work_dir, cfg.WORK_DIR)
        if not FS.exists(cfg.WORK_DIR):
            FS.make_dir(cfg.WORK_DIR)
        cfg = init_env(cfg)
        self.cache_dir = cfg.WORK_DIR
        self.chatbot_examples = get_examples(self.cache_dir) if not cfg.get('SKIP_EXAMPLES', False) else []
        self.model_cfg_dir = cfg.MODEL.EDIT_MODEL.MODEL_CFG_DIR
        self.model_yamls = glob.glob(os.path.join(self.model_cfg_dir,
                                                  '*.yaml'))
        self.model_choices = dict()
        self.default_model_name = ''
        for i in self.model_yamls:
            model_cfg = Config(load=True, cfg_file=i)
            model_name = model_cfg.NAME
            if model_cfg.IS_DEFAULT: self.default_model_name = model_name
            self.model_choices[model_name] = model_cfg
        print('Models: ', self.model_choices.keys())
        assert len(self.model_choices) > 0
        if self.default_model_name == "": self.default_model_name = list(self.model_choices.keys())[0]
        self.model_name = self.default_model_name
        self.pipe = ACEInference()
        self.pipe.init_from_cfg(self.model_choices[self.default_model_name])
        self.max_msgs = 20
        self.enable_i2v = cfg.get('ENABLE_I2V', False)
        self.gradio_version = version('gradio')

        if self.enable_i2v:
            self.i2v_model_dir = cfg.MODEL.I2V.MODEL_DIR
            self.i2v_model_name = cfg.MODEL.I2V.MODEL_NAME
            if self.i2v_model_name == 'CogVideoX-5b-I2V':
                with FS.get_dir_to_local_dir(self.i2v_model_dir) as local_dir:
                    self.i2v_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                        local_dir, torch_dtype=torch.bfloat16).cuda()
            else:
                raise NotImplementedError

            with FS.get_dir_to_local_dir(
                    cfg.MODEL.CAPTIONER.MODEL_DIR) as local_dir:
                self.captioner = AutoModel.from_pretrained(
                    local_dir,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True).eval().cuda()
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    local_dir, trust_remote_code=True, use_fast=False)
                self.llm_generation_config = dict(max_new_tokens=1024,
                                                  do_sample=True)
                self.llm_prompt = cfg.LLM.PROMPT
                self.llm_max_num = 2

            with FS.get_dir_to_local_dir(
                    cfg.MODEL.ENHANCER.MODEL_DIR) as local_dir:
                self.enhancer = transformers.pipeline(
                    'text-generation',
                    model=local_dir,
                    model_kwargs={'torch_dtype': torch.bfloat16},
                    device_map='auto',
                )

            sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

            For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
            There are a few rules to follow:

            You will only ever output a single video description per user request.

            When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
            Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

            Video descriptions must have the same num of words as examples below. Extra words will be ignored.
            """
            self.enhance_ctx = [
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {
                    'role':
                    'user',
                    'content':
                    'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    'role':
                    'assistant',
                    'content':
                    "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    'role':
                    'user',
                    'content':
                    'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    'role':
                    'assistant',
                    'content':
                    "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    'role':
                    'user',
                    'content':
                    'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    'role':
                    'assistant',
                    'content':
                    'A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.',
                },
            ]

    def create_ui(self):

        css = '.chatbot.prose.md {opacity: 1.0 !important} #chatbot {opacity: 1.0 !important}'
        with gr.Blocks(css=css,
                       title='Chatbot',
                       head='Chatbot',
                       analytics_enabled=False):
            self.history = gr.State(value=[])
            self.images = gr.State(value={})
            self.history_result = gr.State(value={})
            self.retry_msg = gr.State(value='')
            with gr.Group():
                self.ui_mode = gr.State(value='legacy')
                with gr.Row(equal_height=True, visible=False) as self.chat_group:
                    with gr.Column(visible=True) as self.chat_page:
                        self.chatbot = gr.Chatbot(
                            height=600,
                            value=[],
                            bubble_full_width=False,
                            show_copy_button=True,
                            container=False,
                            placeholder='<strong>Chat Box</strong>')
                        with gr.Row():
                            self.clear_btn = gr.Button(clear_sty +
                                                       ' Clear Chat',
                                                       size='sm')

                    with gr.Column(visible=False) as self.editor_page:
                        with gr.Tabs(visible=False) as self.upload_tabs:
                            with gr.Tab(id='ImageUploader',
                                        label='Image Uploader',
                                        visible=True) as self.upload_tab:
                                self.image_uploader = gr.Image(
                                    height=550,
                                    interactive=True,
                                    type='pil',
                                    image_mode='RGB',
                                    sources=['upload'],
                                    elem_id='image_uploader',
                                    format='png')
                                with gr.Row():
                                    self.sub_btn_1 = gr.Button(
                                        value='Submit',
                                        elem_id='upload_submit')
                                    self.ext_btn_1 = gr.Button(value='Exit')
                        with gr.Tabs(visible=False) as self.edit_tabs:
                            with gr.Tab(id='ImageEditor',
                                        label='Image Editor') as self.edit_tab:
                                self.mask_type = gr.Dropdown(
                                    label='Mask Type',
                                    choices=[
                                        'Background', 'Composite',
                                        'Outpainting'
                                    ],
                                    value='Background')
                                self.mask_type_info = gr.HTML(
                                    value=
                                    "<div style='background-color: white; padding-left: 15px; color: grey;'>Background mode will not erase the visual content in the mask area</div>"
                                )
                                with gr.Accordion(
                                        label='Outpainting Setting',
                                        open=True,
                                        visible=False) as self.outpaint_tab:
                                    with gr.Row(variant='panel'):
                                        self.top_ext = gr.Slider(
                                            show_label=True,
                                            label='Top Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                        self.bottom_ext = gr.Slider(
                                            show_label=True,
                                            label='Bottom Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                    with gr.Row(variant='panel'):
                                        self.left_ext = gr.Slider(
                                            show_label=True,
                                            label='Left Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                        self.right_ext = gr.Slider(
                                            show_label=True,
                                            label='Right Extend Ratio',
                                            minimum=0.0,
                                            maximum=2.0,
                                            step=0.1,
                                            value=0.25)
                                    with gr.Row(variant='panel'):
                                        self.img_pad_btn = gr.Button(
                                            value='Pad Image')

                                self.image_editor = gr.ImageMask(
                                    value=None,
                                    sources=[],
                                    layers=False,
                                    label='Edit Image',
                                    elem_id='image_editor',
                                    format='png')
                                with gr.Row():
                                    self.sub_btn_2 = gr.Button(
                                        value='Submit', elem_id='edit_submit')
                                    self.ext_btn_2 = gr.Button(value='Exit')

                            with gr.Tab(id='ImageViewer',
                                        label='Image Viewer') as self.image_view_tab:
                                if self.gradio_version >= '5.0.0':
                                    self.image_viewer = gr.Image(
                                        label='Image',
                                        type='pil',
                                        show_download_button=True,
                                        elem_id='image_viewer')
                                else:
                                    try:
                                        from gradio_imageslider import ImageSlider
                                    except Exception as e:
                                        print(f"Import gradio_imageslider failed, please install.")
                                    self.image_viewer = ImageSlider(
                                        label='Image',
                                        type='pil',
                                        show_download_button=True,
                                        elem_id='image_viewer')

                                self.ext_btn_3 = gr.Button(value='Exit')

                            with gr.Tab(id='VideoViewer',
                                        label='Video Viewer',
                                        visible=False) as self.video_view_tab:
                                self.video_viewer = gr.Video(
                                    label='Video',
                                    interactive=False,
                                    sources=[],
                                    format='mp4',
                                    show_download_button=True,
                                    elem_id='video_viewer',
                                    loop=True,
                                    autoplay=True)

                                self.ext_btn_4 = gr.Button(value='Exit')

                with gr.Row(equal_height=True, visible=True) as self.legacy_group:
                    with gr.Column():
                        self.legacy_image_uploader = gr.Image(
                            height=550,
                            interactive=True,
                            type='pil',
                            image_mode='RGB',
                            elem_id='legacy_image_uploader',
                            format='png')
                    with gr.Column():
                        self.legacy_image_viewer = gr.Image(
                            label='Image',
                            height=550,
                            type='pil',
                            interactive=False,
                            show_download_button=True,
                            elem_id='image_viewer')


                with gr.Accordion(label='Setting', open=False):
                    with gr.Row():
                        self.model_name_dd = gr.Dropdown(
                            choices=self.model_choices,
                            value=self.default_model_name,
                            label='Model Version')

                    with gr.Row():
                        self.negative_prompt = gr.Textbox(
                            value='',
                            placeholder=
                            'Negative prompt used for Classifier-Free Guidance',
                            label='Negative Prompt',
                            container=False)

                    with gr.Row():
                        # REFINER_PROMPT
                        self.refiner_prompt = gr.Textbox(
                            value=self.pipe.input.get("refiner_prompt", ""),
                            visible=self.pipe.input.get("refiner_prompt", None) is not None,
                            placeholder=
                            'Prompt used for refiner',
                            label='Refiner Prompt',
                            container=False)


                    with gr.Row():
                        with gr.Column(scale=8, min_width=500):
                            with gr.Row():
                                self.step = gr.Slider(minimum=1,
                                                      maximum=1000,
                                                      value=self.pipe.input.get("sample_steps", 20),
                                                      visible=self.pipe.input.get("sample_steps", None) is not None,
                                                      label='Sample Step')
                                self.cfg_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=self.pipe.input.get("guide_scale", 4.5),
                                    visible=self.pipe.input.get("guide_scale", None) is not None,
                                    label='Guidance Scale')
                                self.rescale = gr.Slider(minimum=0.0,
                                                         maximum=1.0,
                                                         value=self.pipe.input.get("guide_rescale", 0.5),
                                                         visible=self.pipe.input.get("guide_rescale", None) is not None,
                                                         label='Rescale')
                                self.refiner_scale = gr.Slider(minimum=-0.1,
                                                         maximum=1.0,
                                                         value=self.pipe.input.get("refiner_scale", -1),
                                                         visible=self.pipe.input.get("refiner_scale", None) is not None,
                                                         label='Refiner Scale')
                                self.seed = gr.Slider(minimum=-1,
                                                      maximum=10000000,
                                                      value=-1,
                                                      label='Seed')
                                self.output_height = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=self.pipe.input.get("output_height", 1024),
                                    visible=self.pipe.input.get("output_height", None) is not None,
                                    label='Output Height')
                                self.output_width = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=self.pipe.input.get("output_width", 1024),
                                    visible=self.pipe.input.get("output_width", None) is not None,
                                    label='Output Width')
                        with gr.Column(scale=1, min_width=50):
                            self.use_history = gr.Checkbox(value=False,
                                                           label='Use History')
                            self.use_ace = gr.Checkbox(value=self.pipe.input.get("use_ace", True),
                                                       visible=self.pipe.input.get("use_ace", None) is not None,
                                                       label='Use ACE')
                            self.video_auto = gr.Checkbox(
                                value=False,
                                label='Auto Gen Video',
                                visible=self.enable_i2v)

                    with gr.Row(variant='panel',
                                equal_height=True,
                                visible=self.enable_i2v):
                        self.video_fps = gr.Slider(minimum=1,
                                                   maximum=16,
                                                   value=8,
                                                   label='Video FPS',
                                                   visible=True)
                        self.video_frames = gr.Slider(minimum=8,
                                                      maximum=49,
                                                      value=49,
                                                      label='Video Frame Num',
                                                      visible=True)
                        self.video_step = gr.Slider(minimum=1,
                                                    maximum=1000,
                                                    value=50,
                                                    label='Video Sample Step',
                                                    visible=True)
                        self.video_cfg_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=6.0,
                            label='Video Guidance Scale',
                            visible=True)
                        self.video_seed = gr.Slider(minimum=-1,
                                                    maximum=10000000,
                                                    value=-1,
                                                    label='Video Seed',
                                                    visible=True)

                with gr.Row():
                    self.chatbot_inst = """
                       **Instruction**:

                       1. Click 'Upload' button to upload one or more images as input images.
                       2. Enter '@' in the text box will exhibit all images in the gallery.
                       3. Select the image you wish to edit from the gallery, and its Image ID will be displayed in the text box.
                       4. Compose the editing instruction for the selected image, incorporating image id '@xxxxxx' into your instruction.
                       For example, you might say, "Change the girl's skirt in @123456 to blue." The '@xxxxx' token will facilitate the identification of the specific image, and will be automatically replaced by a special token '{image}' in the instruction. Furthermore, it is also possible to engage in text-to-image generation without any initial image input.
                       5. Once your instructions are prepared, please click the "Chat" button to view the edited result in the chat window.
                       6. **Important** To render text on an image, please ensure to include a space between each letter. For instance, "add text 'g i r l' on the mask area of @xxxxx".
                       7. To implement local editing based on a specified mask, simply click on the image within the chat window to access the image editor. Here, you can draw a mask and then click the 'Submit' button to upload the edited image along with the mask. For inpainting tasks, select the 'Composite' mask type, while for outpainting tasks, choose the 'Outpainting' mask type. For all other local editing tasks, please select the 'Background' mask type.
                       8. If you find our work valuable, we invite you to refer to the [ACE Page](https://ali-vilab.github.io/ace-page/) for comprehensive information.

                    """

                    self.legacy_inst = """
                       **Instruction**:

                       1. You can edit the image by uploading it; if no image is uploaded, an image will be generated from text..
                       2. Enter '@' in the text box will exhibit all images in the gallery.
                       3. Select the image you wish to edit from the gallery, and its Image ID will be displayed in the text box.
                       4. **Important** To render text on an image, please ensure to include a space between each letter. For instance, "add text 'g i r l' on the mask area of @xxxxx".
                       5. To perform multi-step editing, partial editing, inpainting, outpainting, and other operations, please click the Chatbot Checkbox to enable the conversational editing mode and follow the relevant instructions..
                       6. If you find our work valuable, we invite you to refer to the [ACE Page](https://ali-vilab.github.io/ace-page/) for comprehensive information.

                    """

                    self.instruction = gr.Markdown(value=self.legacy_inst)

                with gr.Row(variant='panel',
                            equal_height=True,
                            show_progress=False):
                    with gr.Column(scale=1, min_width=100, visible=False) as self.upload_panel:
                        self.upload_btn = gr.Button(value=upload_sty +
                                                    ' Upload',
                                                    variant='secondary')
                    with gr.Column(scale=5, min_width=500):
                        self.text = gr.Textbox(
                            placeholder='Input "@" find history of image',
                            label='Instruction',
                            container=False)
                    with gr.Column(scale=1, min_width=100):
                        self.chat_btn = gr.Button(value='Generate',
                                                  variant='primary')
                    with gr.Column(scale=1, min_width=100):
                        self.retry_btn = gr.Button(value=refresh_sty +
                                                   ' Retry',
                                                   variant='secondary')
                    with gr.Column(scale=1, min_width=100):
                        self.mode_checkbox = gr.Checkbox(
                            value=False,
                            label='ChatBot')
                    with gr.Column(scale=(1 if self.enable_i2v else 0),
                                   min_width=0):
                        self.video_gen_btn = gr.Button(value=video_sty +
                                                       ' Gen Video',
                                                       variant='secondary',
                                                       visible=self.enable_i2v)
                    with gr.Column(scale=(1 if self.enable_i2v else 0),
                                   min_width=0):
                        self.extend_prompt = gr.Checkbox(
                            value=True,
                            label='Extend Prompt',
                            visible=self.enable_i2v)

                with gr.Row():
                    self.gallery = gr.Gallery(visible=False,
                                              label='History',
                                              columns=10,
                                              allow_preview=False,
                                              interactive=False)

                self.eg = gr.Column(visible=True)

    def set_callbacks(self, *args, **kwargs):

        ########################################
        def change_model(model_name):
            if model_name not in self.model_choices:
                gr.Info('The provided model name is not a valid choice!')
                return model_name, gr.update(), gr.update()

            if model_name != self.model_name:
                lock.acquire()
                del self.pipe
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                self.pipe = ACEInference()
                self.pipe.init_from_cfg(self.model_choices[model_name])
                self.model_name = model_name
                lock.release()

            return (model_name, gr.update(), gr.update(),
                    gr.Slider(
                              value=self.pipe.input.get("sample_steps", 20),
                              visible=self.pipe.input.get("sample_steps", None) is not None),
                    gr.Slider(
                        value=self.pipe.input.get("guide_scale", 4.5),
                        visible=self.pipe.input.get("guide_scale", None) is not None),
                    gr.Slider(
                              value=self.pipe.input.get("guide_rescale", 0.5),
                              visible=self.pipe.input.get("guide_rescale", None) is not None),
                    gr.Slider(
                        value=self.pipe.input.get("output_height", 1024),
                        visible=self.pipe.input.get("output_height", None) is not None),
                    gr.Slider(
                        value=self.pipe.input.get("output_width", 1024),
                        visible=self.pipe.input.get("output_width", None) is not None),
                    gr.Textbox(
                        value=self.pipe.input.get("refiner_prompt", ""),
                        visible=self.pipe.input.get("refiner_prompt", None) is not None),
                    gr.Slider(
                              value=self.pipe.input.get("refiner_scale", -1),
                              visible=self.pipe.input.get("refiner_scale", None) is not None
                        ),
                    gr.Checkbox(
                        value=self.pipe.input.get("use_ace", True),
                        visible=self.pipe.input.get("use_ace", None) is not None
                    )
                    )

        self.model_name_dd.change(
            change_model,
            inputs=[self.model_name_dd],
            outputs=[
                self.model_name_dd, self.chatbot, self.text,
                self.step,
                self.cfg_scale, self.rescale, self.output_height,
                self.output_width, self.refiner_prompt, self.refiner_scale,
                self.use_ace])


        def mode_change(mode_check):
            if mode_check:
                # ChatBot
                return (
                    gr.Row(visible=False),
                    gr.Row(visible=True),
                    gr.Button(value='Generate'),
                    gr.State(value='chatbot'),
                    gr.Column(visible=True),
                    gr.Markdown(value=self.chatbot_inst)
                )
            else:
                # Legacy
                return (
                    gr.Row(visible=True),
                    gr.Row(visible=False),
                    gr.Button(value=chat_sty + ' Chat'),
                    gr.State(value='legacy'),
                    gr.Column(visible=False),
                    gr.Markdown(value=self.legacy_inst)
                )
        self.mode_checkbox.change(mode_change, inputs=[self.mode_checkbox],
                                  outputs=[self.legacy_group, self.chat_group,
                                           self.chat_btn, self.ui_mode,
                                           self.upload_panel, self.instruction])


        ########################################
        def generate_gallery(text, images):
            if text.endswith(' '):
                return gr.update(), gr.update(visible=False)
            elif text.endswith('@'):
                gallery_info = []
                for image_id, image_meta in images.items():
                    thumbnail_path = image_meta['thumbnail']
                    gallery_info.append((thumbnail_path, image_id))
                return gr.update(), gr.update(visible=True, value=gallery_info)
            else:
                gallery_info = []
                match = re.search('@([^@ ]+)$', text)
                if match:
                    prefix = match.group(1)
                    for image_id, image_meta in images.items():
                        if not image_id.startswith(prefix):
                            continue
                        thumbnail_path = image_meta['thumbnail']
                        gallery_info.append((thumbnail_path, image_id))

                    if len(gallery_info) > 0:
                        return gr.update(), gr.update(visible=True,
                                                      value=gallery_info)
                    else:
                        return gr.update(), gr.update(visible=False)
                else:
                    return gr.update(), gr.update(visible=False)

        self.text.input(generate_gallery,
                        inputs=[self.text, self.images],
                        outputs=[self.text, self.gallery],
                        show_progress='hidden')

        ########################################
        def select_image(text, evt: gr.SelectData):
            image_id = evt.value['caption']
            text = '@'.join(text.split('@')[:-1]) + f'@{image_id} '
            return gr.update(value=text), gr.update(visible=False, value=None)

        self.gallery.select(select_image,
                            inputs=self.text,
                            outputs=[self.text, self.gallery])

        ########################################
        def generate_video(message,
                           extend_prompt,
                           history,
                           images,
                           num_steps,
                           num_frames,
                           cfg_scale,
                           fps,
                           seed,
                           progress=gr.Progress(track_tqdm=True)):

            from diffusers.utils import export_to_video

            generator = torch.Generator(device='cuda').manual_seed(seed)
            img_ids = re.findall('@(.*?)[ ,;.?$]', message)
            if len(img_ids) == 0:
                history.append((
                    message,
                    'Sorry, no images were found in the prompt to be used as the first frame of the video.'
                ))
                while len(history) >= self.max_msgs:
                    history.pop(0)
                return history, self.get_history(
                    history), gr.update(), gr.update(visible=False)

            img_id = img_ids[0]
            prompt = re.sub(f'@{img_id}\s+', '', message)

            if extend_prompt:
                messages = copy.deepcopy(self.enhance_ctx)
                messages.append({
                    'role':
                    'user',
                    'content':
                    f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{prompt}"',
                })
                lock.acquire()
                outputs = self.enhancer(
                    messages,
                    max_new_tokens=200,
                )

                prompt = outputs[0]['generated_text'][-1]['content']
                print(prompt)
                lock.release()

            img_meta = images[img_id]
            img_path = img_meta['image']
            image = Image.open(img_path).convert('RGB')

            lock.acquire()
            video = self.i2v_pipe(
                prompt=prompt,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=num_steps,
                num_frames=num_frames,
                guidance_scale=cfg_scale,
                generator=generator,
            ).frames[0]
            lock.release()

            out_video_path = export_to_video(video, fps=fps)
            history.append((
                f"Based on first frame @{img_id} and description '{prompt}', generate a video",
                'This is generated video:'))
            history.append((None, out_video_path))
            while len(history) >= self.max_msgs:
                history.pop(0)

            return history, self.get_history(history), gr.update(
                value=''), gr.update(visible=False)

        self.video_gen_btn.click(
            generate_video,
            inputs=[
                self.text, self.extend_prompt, self.history, self.images,
                self.video_step, self.video_frames, self.video_cfg_scale,
                self.video_fps, self.video_seed
            ],
            outputs=[self.history, self.chatbot, self.text, self.gallery])

        ########################################
        def run_chat(
                     message,
                     legacy_image,
                     ui_mode,
                     use_ace,
                     extend_prompt,
                     history,
                     images,
                     use_history,
                     history_result,
                     negative_prompt,
                     cfg_scale,
                     rescale,
                     refiner_prompt,
                     refiner_scale,
                     step,
                     seed,
                     output_h,
                     output_w,
                     video_auto,
                     video_steps,
                     video_frames,
                     video_cfg_scale,
                     video_fps,
                     video_seed,
                     progress=gr.Progress(track_tqdm=True)):
            legacy_img_ids = []
            if ui_mode == 'legacy':
                if legacy_image is not None:
                    history, images, img_id = self.add_uploaded_image_to_history(
                        legacy_image, history, images)
                    legacy_img_ids.append(img_id)
            retry_msg = message
            gen_id = get_md5(message)[:12]
            save_path = os.path.join(self.cache_dir, f'{gen_id}.png')

            img_ids = re.findall('@(.*?)[ ,;.?$]', message)
            history_io = None

            if len(img_ids) < 1:
                img_ids = legacy_img_ids
                for img_id in img_ids:
                    if f'@{img_id}' not in message:
                        message = f'@{img_id} ' + message

            new_message = message

            if len(img_ids) > 0:
                edit_image, edit_image_mask, edit_task = [], [], []
                for i, img_id in enumerate(img_ids):
                    if img_id not in images:
                        gr.Info(
                            f'The input image ID {img_id} is not exist... Skip loading image.'
                        )
                        continue
                    placeholder = '{image}' if i == 0 else '{' + f'image{i}' + '}'
                    new_message = re.sub(f'@{img_id}', placeholder,
                                         new_message)
                    img_meta = images[img_id]
                    img_path = img_meta['image']
                    img_mask = img_meta['mask']
                    img_mask_type = img_meta['mask_type']
                    if img_mask_type is not None and img_mask_type == 'Composite':
                        task = 'inpainting'
                    else:
                        task = ''
                    edit_image.append(Image.open(img_path).convert('RGB'))
                    edit_image_mask.append(
                        Image.open(img_mask).
                        convert('L') if img_mask is not None else None)
                    edit_task.append(task)

                    if use_history and (img_id in history_result):
                        history_io = history_result[img_id]

                buffered = io.BytesIO()
                edit_image[0].save(buffered, format='PNG')
                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'
                pre_info = f'Received one or more images, so image editing is conducted.\n The first input image @{img_ids[0]} is:\n {img_str}'
            else:
                pre_info = 'No image ids were found in the provided text prompt, so text-guided image generation is conducted. \n'
                edit_image = None
                edit_image_mask = None
                edit_task = ''

            print(new_message)
            imgs = self.pipe(
                image=edit_image,
                mask=edit_image_mask,
                task=edit_task,
                prompt=[new_message] *
                len(edit_image) if edit_image is not None else [new_message],
                negative_prompt=[negative_prompt] * len(edit_image)
                if edit_image is not None else [negative_prompt],
                history_io=history_io,
                output_height=output_h,
                output_width=output_w,
                sampler='ddim',
                sample_steps=step,
                guide_scale=cfg_scale,
                guide_rescale=rescale,
                seed=seed,
                refiner_prompt=refiner_prompt,
                refiner_scale=refiner_scale,
                use_ace=use_ace
            )

            img = imgs[0]
            img.save(save_path, format='PNG')

            if history_io:
                history_io_new = copy.deepcopy(history_io)
                history_io_new['image'] += edit_image[:1]
                history_io_new['mask'] += edit_image_mask[:1]
                history_io_new['task'] += edit_task[:1]
                history_io_new['prompt'] += [new_message]
                history_io_new['image'] = history_io_new['image'][-5:]
                history_io_new['mask'] = history_io_new['mask'][-5:]
                history_io_new['task'] = history_io_new['task'][-5:]
                history_io_new['prompt'] = history_io_new['prompt'][-5:]
                history_result[gen_id] = history_io_new
            elif edit_image is not None and len(edit_image) > 0:
                history_io_new = {
                    'image': edit_image[:1],
                    'mask': edit_image_mask[:1],
                    'task': edit_task[:1],
                    'prompt': [new_message]
                }
                history_result[gen_id] = history_io_new

            w, h = img.size
            if w > h:
                tb_w = 128
                tb_h = int(h * tb_w / w)
            else:
                tb_h = 128
                tb_w = int(w * tb_h / h)

            thumbnail_path = os.path.join(self.cache_dir,
                                          f'{gen_id}_thumbnail.jpg')
            thumbnail = img.resize((tb_w, tb_h))
            thumbnail.save(thumbnail_path, format='JPEG')

            images[gen_id] = {
                'image': save_path,
                'mask': None,
                'mask_type': None,
                'thumbnail': thumbnail_path
            }

            buffered = io.BytesIO()
            img.convert('RGB').save(buffered, format='PNG')
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'

            history.append(
                (message,
                 f'{pre_info} The generated image @{gen_id} is:\n {img_str}'))

            if video_auto:
                if video_seed is None or video_seed == -1:
                    video_seed = random.randint(0, 10000000)

                lock.acquire()
                generator = torch.Generator(
                    device='cuda').manual_seed(video_seed)
                pixel_values = load_image(img.convert('RGB'),
                                          max_num=self.llm_max_num).to(
                                              torch.bfloat16).cuda()
                prompt = self.captioner.chat(self.llm_tokenizer, pixel_values,
                                             self.llm_prompt,
                                             self.llm_generation_config)
                print(prompt)
                lock.release()

                if extend_prompt:
                    messages = copy.deepcopy(self.enhance_ctx)
                    messages.append({
                        'role':
                        'user',
                        'content':
                        f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{prompt}"',
                    })
                    lock.acquire()
                    outputs = self.enhancer(
                        messages,
                        max_new_tokens=200,
                    )
                    prompt = outputs[0]['generated_text'][-1]['content']
                    print(prompt)
                    lock.release()

                lock.acquire()
                video = self.i2v_pipe(
                    prompt=prompt,
                    image=img,
                    num_videos_per_prompt=1,
                    num_inference_steps=video_steps,
                    num_frames=video_frames,
                    guidance_scale=video_cfg_scale,
                    generator=generator,
                ).frames[0]
                lock.release()

                out_video_path = export_to_video(video, fps=video_fps)
                history.append((
                    f"Based on first frame @{gen_id} and description '{prompt}', generate a video",
                    'This is generated video:'))
                history.append((None, out_video_path))

            while len(history) >= self.max_msgs:
                history.pop(0)

            return (history, images, gr.Image(value=save_path),
                    history_result, self.get_history(
                history), gr.update(), gr.update(
                    visible=False), retry_msg)

        chat_inputs = [
            self.legacy_image_uploader, self.ui_mode, self.use_ace,
            self.extend_prompt, self.history, self.images, self.use_history,
            self.history_result, self.negative_prompt, self.cfg_scale,
            self.rescale, self.refiner_prompt, self.refiner_scale,
            self.step, self.seed, self.output_height,
            self.output_width, self.video_auto, self.video_step,
            self.video_frames, self.video_cfg_scale, self.video_fps,
            self.video_seed
        ]

        chat_outputs = [
            self.history, self.images, self.legacy_image_viewer,
            self.history_result, self.chatbot,
            self.text, self.gallery, self.retry_msg
        ]

        self.chat_btn.click(run_chat,
                            inputs=[self.text] + chat_inputs,
                            outputs=chat_outputs)

        self.text.submit(run_chat,
                         inputs=[self.text] + chat_inputs,
                         outputs=chat_outputs)

        def retry_fn(*args):
            return run_chat(*args)

        self.retry_btn.click(retry_fn,
                             inputs=[self.retry_msg] + chat_inputs,
                             outputs=chat_outputs)

        ########################################
        def run_example(task, img, img_mask, ref1, prompt, seed):
            edit_image, edit_image_mask, edit_task = [], [], []
            if img is not None:
                w, h = img.size
                if w > 2048:
                    ratio = w / 2048.
                    w = 2048
                    h = int(h / ratio)
                if h > 2048:
                    ratio = h / 2048.
                    h = 2048
                    w = int(w / ratio)
                img = img.resize((w, h))
                edit_image.append(img)
                if img_mask is not None:
                    img_mask = img_mask if np.sum(np.array(img_mask)) > 0 else None
                edit_image_mask.append(
                    img_mask if img_mask is not None else None)
                edit_task.append(task)
                if ref1 is not None:
                    ref1 = ref1 if np.sum(np.array(ref1)) > 0 else None
                if ref1 is not None:
                    edit_image.append(ref1)
                    edit_image_mask.append(None)
                    edit_task.append('')

                buffered = io.BytesIO()
                img.save(buffered, format='PNG')
                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'
                pre_info = f'Received one or more images, so image editing is conducted.\n The first input image is:\n {img_str}'
            else:
                pre_info = 'No image ids were found in the provided text prompt, so text-guided image generation is conducted. \n'
                edit_image = None
                edit_image_mask = None
                edit_task = ''

            img_num = len(edit_image) if edit_image is not None else 1
            imgs = self.pipe(
                image=edit_image,
                mask=edit_image_mask,
                task=edit_task,
                prompt=[prompt] * img_num,
                negative_prompt=[''] * img_num,
                seed=seed,
                refiner_prompt=self.pipe.input.get("refiner_prompt", ""),
                refiner_scale=self.pipe.input.get("refiner_scale", 0.0),
            )

            img = imgs[0]
            buffered = io.BytesIO()
            img.convert('RGB').save(buffered, format='PNG')
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'
            history = [(prompt,
                        f'{pre_info} The generated image is:\n {img_str}')]

            img_id = get_md5(img_b64)[:12]
            save_path = os.path.join(self.cache_dir, f'{img_id}.png')
            img.convert('RGB').save(save_path)

            return self.get_history(history), gr.update(value=''), gr.update(
                visible=False), gr.Image(value=save_path), gr.update(value=-1)

        with self.eg:
            self.example_task = gr.Text(label='Task Name',
                                        value='',
                                        visible=False)
            self.example_image = gr.Image(label='Edit Image',
                                          type='pil',
                                          image_mode='RGB',
                                          visible=False)
            self.example_mask = gr.Image(label='Edit Image Mask',
                                         type='pil',
                                         image_mode='L',
                                         visible=False)
            self.example_ref_im1 = gr.Image(label='Ref Image',
                                            type='pil',
                                            image_mode='RGB',
                                            visible=False)

            self.examples = gr.Examples(
                fn=run_example,
                examples=self.chatbot_examples,
                inputs=[
                    self.example_task, self.example_image, self.example_mask,
                    self.example_ref_im1, self.text, self.seed
                ],
                outputs=[self.chatbot, self.text, self.gallery, self.legacy_image_viewer, self.seed],
                examples_per_page=4,
                cache_examples=False,
                run_on_click=True)

        ########################################
        def upload_image():
            return (gr.update(visible=True,
                              scale=1), gr.update(visible=True, scale=1),
                    gr.update(visible=True), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True))

        self.upload_btn.click(upload_image,
                              inputs=[],
                              outputs=[
                                  self.chat_page, self.editor_page,
                                  self.upload_tab, self.edit_tab,
                                  self.image_view_tab, self.video_view_tab,
                                  self.upload_tabs
                              ])

        ########################################
        def edit_image(evt: gr.SelectData):
            if isinstance(evt.value, str):
                img_b64s = re.findall(
                    '<img src="data:image/png;base64,(.*?)" style="pointer-events: none;">',
                    evt.value)
                imgs = [
                    Image.open(io.BytesIO(base64.b64decode(copy.deepcopy(i))))
                    for i in img_b64s
                ]
                if len(imgs) > 0:
                    if len(imgs) == 2:
                        if self.gradio_version >= '5.0.0':
                            view_img = copy.deepcopy(imgs[-1])
                        else:
                            view_img = copy.deepcopy(imgs)
                        edit_img = copy.deepcopy(imgs[-1])
                    else:
                        if self.gradio_version >= '5.0.0':
                            view_img = copy.deepcopy(imgs[-1])
                        else:
                            view_img = [
                                copy.deepcopy(imgs[-1]),
                                copy.deepcopy(imgs[-1])
                            ]
                        edit_img = copy.deepcopy(imgs[-1])

                    return (gr.update(visible=True,
                                      scale=1), gr.update(visible=True,
                                                          scale=1),
                            gr.update(visible=False), gr.update(visible=True),
                            gr.update(visible=True), gr.update(visible=False),
                            gr.update(value=edit_img),
                            gr.update(value=view_img), gr.update(value=None),
                            gr.update(visible=True))
                else:
                    return (gr.update(), gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update())
            elif isinstance(evt.value, dict) and evt.value.get(
                    'component', '') == 'video':
                value = evt.value['value']['video']['path']
                return (gr.update(visible=True,
                                  scale=1), gr.update(visible=True, scale=1),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=True),
                        gr.update(), gr.update(), gr.update(value=value),
                        gr.update())
            else:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update())

        self.chatbot.select(edit_image,
                            outputs=[
                                self.chat_page, self.editor_page,
                                self.upload_tab, self.edit_tab,
                                self.image_view_tab, self.video_view_tab,
                                self.image_editor, self.image_viewer,
                                self.video_viewer, self.edit_tabs
                            ])

        if self.gradio_version < '5.0.0':
            self.image_viewer.change(lambda x: x,
                                     inputs=self.image_viewer,
                                     outputs=self.image_viewer)

        ########################################
        def submit_upload_image(image, history, images):
            history, images, _ = self.add_uploaded_image_to_history(
                image, history, images)
            return gr.update(visible=False), gr.update(
                visible=True), gr.update(
                    value=self.get_history(history)), history, images

        self.sub_btn_1.click(
            submit_upload_image,
            inputs=[self.image_uploader, self.history, self.images],
            outputs=[
                self.editor_page, self.chat_page, self.chatbot, self.history,
                self.images
            ])

        ########################################
        def submit_edit_image(imagemask, mask_type, history, images):
            history, images = self.add_edited_image_to_history(
                imagemask, mask_type, history, images)
            return gr.update(visible=False), gr.update(
                visible=True), gr.update(
                    value=self.get_history(history)), history, images

        self.sub_btn_2.click(submit_edit_image,
                             inputs=[
                                 self.image_editor, self.mask_type,
                                 self.history, self.images
                             ],
                             outputs=[
                                 self.editor_page, self.chat_page,
                                 self.chatbot, self.history, self.images
                             ])

        ########################################
        def exit_edit():
            return gr.update(visible=False), gr.update(visible=True, scale=3)

        self.ext_btn_1.click(exit_edit,
                             outputs=[self.editor_page, self.chat_page])
        self.ext_btn_2.click(exit_edit,
                             outputs=[self.editor_page, self.chat_page])
        self.ext_btn_3.click(exit_edit,
                             outputs=[self.editor_page, self.chat_page])
        self.ext_btn_4.click(exit_edit,
                             outputs=[self.editor_page, self.chat_page])

        ########################################
        def update_mask_type_info(mask_type):
            if mask_type == 'Background':
                info = 'Background mode will not erase the visual content in the mask area'
                visible = False
            elif mask_type == 'Composite':
                info = 'Composite mode will erase the visual content in the mask area'
                visible = False
            elif mask_type == 'Outpainting':
                info = 'Outpaint mode is used for preparing input image for outpainting task'
                visible = True
            return (gr.update(
                visible=True,
                value=
                f"<div style='background-color: white; padding-left: 15px; color: grey;'>{info}</div>"
            ), gr.update(visible=visible))

        self.mask_type.change(update_mask_type_info,
                              inputs=self.mask_type,
                              outputs=[self.mask_type_info, self.outpaint_tab])

        ########################################
        def extend_image(top_ratio, bottom_ratio, left_ratio, right_ratio,
                         image):
            img = cv2.cvtColor(image['background'], cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            new_h = int(h * (top_ratio + bottom_ratio + 1))
            new_w = int(w * (left_ratio + right_ratio + 1))
            start_h = int(h * top_ratio)
            start_w = int(w * left_ratio)
            new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            new_mask = np.ones((new_h, new_w, 1), dtype=np.uint8) * 255
            new_img[start_h:start_h + h, start_w:start_w + w, :] = img
            new_mask[start_h:start_h + h, start_w:start_w + w] = 0
            layer = np.concatenate([new_img, new_mask], axis=2)
            value = {
                'background': new_img,
                'composite': new_img,
                'layers': [layer]
            }
            return gr.update(value=value)

        self.img_pad_btn.click(extend_image,
                               inputs=[
                                   self.top_ext, self.bottom_ext,
                                   self.left_ext, self.right_ext,
                                   self.image_editor
                               ],
                               outputs=self.image_editor)

        ########################################
        def clear_chat(history, images, history_result):
            history.clear()
            images.clear()
            history_result.clear()
            return history, images, history_result, self.get_history(history)

        self.clear_btn.click(
            clear_chat,
            inputs=[self.history, self.images, self.history_result],
            outputs=[
                self.history, self.images, self.history_result, self.chatbot
            ])

    def get_history(self, history):
        info = []
        for item in history:
            new_item = [None, None]
            if isinstance(item[0], str) and item[0].endswith('.mp4'):
                new_item[0] = gr.Video(item[0], format='mp4')
            else:
                new_item[0] = item[0]
            if isinstance(item[1], str) and item[1].endswith('.mp4'):
                new_item[1] = gr.Video(item[1], format='mp4')
            else:
                new_item[1] = item[1]
            info.append(new_item)
        return info

    def generate_random_string(self, length=20):
        letters_and_digits = string.ascii_letters + string.digits
        random_string = ''.join(
            random.choice(letters_and_digits) for i in range(length))
        return random_string

    def add_edited_image_to_history(self, image, mask_type, history, images):
        if mask_type == 'Composite':
            img = Image.fromarray(image['composite'])
        else:
            img = Image.fromarray(image['background'])

        img_id = get_md5(self.generate_random_string())[:12]
        save_path = os.path.join(self.cache_dir, f'{img_id}.png')
        img.convert('RGB').save(save_path)

        mask = image['layers'][0][:, :, 3]
        mask = Image.fromarray(mask).convert('RGB')
        mask_path = os.path.join(self.cache_dir, f'{img_id}_mask.png')
        mask.save(mask_path)

        w, h = img.size
        if w > h:
            tb_w = 128
            tb_h = int(h * tb_w / w)
        else:
            tb_h = 128
            tb_w = int(w * tb_h / h)

        if mask_type == 'Background':
            comp_mask = np.array(mask, dtype=np.uint8)
            mask_alpha = (comp_mask[:, :, 0:1].astype(np.float32) *
                          0.6).astype(np.uint8)
            comp_mask = np.concatenate([comp_mask, mask_alpha], axis=2)
            thumbnail = Image.alpha_composite(
                img.convert('RGBA'),
                Image.fromarray(comp_mask).convert('RGBA')).convert('RGB')
        else:
            thumbnail = img.convert('RGB')

        thumbnail_path = os.path.join(self.cache_dir,
                                      f'{img_id}_thumbnail.jpg')
        thumbnail = thumbnail.resize((tb_w, tb_h))
        thumbnail.save(thumbnail_path, format='JPEG')

        buffered = io.BytesIO()
        img.convert('RGB').save(buffered, format='PNG')
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'

        buffered = io.BytesIO()
        mask.convert('RGB').save(buffered, format='PNG')
        mask_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        mask_str = f'<img src="data:image/png;base64,{mask_b64}" style="pointer-events: none;">'

        images[img_id] = {
            'image': save_path,
            'mask': mask_path,
            'mask_type': mask_type,
            'thumbnail': thumbnail_path
        }
        history.append((
            None,
            f'This is edited image and mask:\n {img_str} {mask_str} image ID is: {img_id}'
        ))
        return history, images

    def add_uploaded_image_to_history(self, img, history, images):
        img_id = get_md5(self.generate_random_string())[:12]
        save_path = os.path.join(self.cache_dir, f'{img_id}.png')
        w, h = img.size
        if w > 2048:
            ratio = w / 2048.
            w = 2048
            h = int(h / ratio)
        if h > 2048:
            ratio = h / 2048.
            h = 2048
            w = int(w / ratio)
        img = img.resize((w, h))
        img.save(save_path)

        w, h = img.size
        if w > h:
            tb_w = 128
            tb_h = int(h * tb_w / w)
        else:
            tb_h = 128
            tb_w = int(w * tb_h / h)
        thumbnail_path = os.path.join(self.cache_dir,
                                      f'{img_id}_thumbnail.jpg')
        thumbnail = img.resize((tb_w, tb_h))
        thumbnail.save(thumbnail_path, format='JPEG')

        images[img_id] = {
            'image': save_path,
            'mask': None,
            'mask_type': None,
            'thumbnail': thumbnail_path
        }

        buffered = io.BytesIO()
        img.convert('RGB').save(buffered, format='PNG')
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_str = f'<img src="data:image/png;base64,{img_b64}" style="pointer-events: none;">'

        history.append(
            (None,
             f'This is uploaded image:\n {img_str} image ID is: {img_id}'))
        return history, images, img_id


def run_gr(cfg):
    with gr.Blocks() as demo:
        chatbot = ChatBotUI(cfg)
        chatbot.create_ui()
        chatbot.set_callbacks()
        demo.launch(server_name='0.0.0.0',
                    server_port=cfg.args.server_port,
                    root_path=cfg.args.root_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--server_port',
                        dest='server_port',
                        help='',
                        type=int,
                        default=2345)
    parser.add_argument('--root_path', dest='root_path', help='', default='')
    cfg = Config(load=True, parser_ins=parser)
    run_gr(cfg)
