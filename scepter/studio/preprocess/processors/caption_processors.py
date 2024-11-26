# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import re
import time
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
import numpy as np
from scepter.studio.preprocess.processors.base_processor import \
    BaseCaptionProcessor
import io
from decord import cpu, VideoReader, bridge

__all__ = ['BlipImageBase', 'QWVL', 'QWVLQuantize', 'InternVL15', 'CogVLM2Llama3Caption',
           'OpusMtZhEn', 'OpusMtEnZh']

def get_region(image, mask, mask_id):
    locs = np.where(np.array(mask) == mask_id)
    if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
        return None
    left, right = np.min(locs[1]), np.max(locs[1])
    top, bottom = np.min(locs[0]), np.max(locs[0])
    box = [left, top, right, bottom]
    region_image = image.crop(box)
    region_image.save("1.jpg")
    return region_image

class BlipImageBase(BaseCaptionProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)
        self.model_path = cfg.MODEL_PATH
        self.model_info = {
            'device': 'offline',
            'model': None,
            'tokenizer': None
        }

    def load_model(self):
        is_flg, msg = super().load_model()
        if not is_flg:
            return is_flg, msg
        if self.model_info['device'] == 'offline':
            model = None
            processor = None
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                local_model_dir = FS.get_dir_to_local_dir(self.model_path)
                processor = BlipProcessor.from_pretrained(local_model_dir)
                model = BlipForConditionalGeneration.from_pretrained(
                    local_model_dir).to(we.device_id)
            except Exception as e:
                if model is not None:
                    del model
                if processor is not None:
                    del model
                return False, f"Load model error '{e}'"
            self.model_info['device'] = model.device
            self.model_info['model'] = model
            self.model_info['processor'] = processor
        elif self.model_info['device'] == 'cpu':
            try:
                self.model_info['model'].to(we.device_id)
                self.model_info['device'] = we.device_id
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                del self.model_info['model']
                self.model_info['model'] = None
                self.model_info['device'] = 'offline'
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                return False, f"Load model error '{e}'"

        return True, ''

    def unload_model(self):
        super().unload_model()
        if self.delete_instance:
            self.model_info['device'] = 'offline'
            if self.model_info['model'] is not None:
                self.model_info['model'] = self.model_info['model'].to('cpu')
                del self.model_info['model']
            self.model_info['model'] = None
        elif (isinstance(self.model_info['device'], numbers.Number)
              or str(self.model_info['device']).startswith('cuda')):
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = self.model_info['model'].to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True, ''

    def get_caption(self, image, use_local, mask):
        image = image.convert('RGB')
        if use_local:
            image = get_region(image,
                               mask.convert('L'),
                               255)

        if image is None:
            return ""
        inputs = self.model_info['processor'](image, return_tensors='pt').to(we.device_id)
        out = self.model_info['model'].generate(**inputs)
        caption = self.model_info['processor'].decode(out[0], skip_special_tokens=True)
        return caption
    def __call__(self, **kwargs):
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        use_local = kwargs.get('use_local', False)
        cache = kwargs.get('cache', None)
        preview_target_image = kwargs.get('preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.get('preview_src_mask', None) if use_preview else src_mask
        preview_src_image = kwargs.get('preview_src_image', None) if use_preview else src_image
        preview_caption = kwargs.get('preview_caption', None) if use_preview else caption
        response = ""

        if preview_src_image is not None:
            response += ("src_caption" + ": " +
                         self.get_caption(preview_src_image, use_local, preview_src_mask) + "\n")
        if preview_target_image is not None:
            response += ("target_caption" + ": " +
                         self.get_caption(preview_target_image, use_local, preview_src_mask) + "\n")
        return response


class QWVL(BaseCaptionProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)
        self.model_path = cfg.MODEL_PATH
        self.model_info = {
            'device': 'offline',
            'model': None,
            'tokenizer': None
        }

    def load_model(self):
        is_flg, msg = super().load_model()
        if not is_flg:
            return is_flg, msg
        if self.model_info['device'] == 'offline':
            model = None
            try:
                from modelscope import (AutoModelForCausalLM, AutoTokenizer,
                                        GenerationConfig)
                local_model_dir = FS.get_dir_to_local_dir(self.model_path)
                # without quantization using 19.52G memory
                # with quantization using 7.7G memory
                tokenizer = AutoTokenizer.from_pretrained(
                    local_model_dir, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_dir,
                    device_map='auto',
                    trust_remote_code=True,
                    fp16=True).eval()
                model.generation_config = GenerationConfig.from_pretrained(
                    local_model_dir, trust_remote_code=True)
            except Exception as e:
                if model is not None:
                    del model
                return False, f"Load model error '{e}'"
            self.model_info['device'] = model.device
            self.model_info['model'] = model
            self.model_info['tokenizer'] = tokenizer
        elif self.model_info['device'] == 'cpu':
            try:
                self.model_info['model'].to(we.device_id)
                self.model_info['device'] = we.device_id
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                del self.model_info['model']
                self.model_info['model'] = None
                self.model_info['device'] = 'offline'
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                return False, f"Load model error '{e}'"

        return True, ''

    def unload_model(self):
        super().unload_model()
        if self.delete_instance:
            if self.model_info['model'] is not None:
                self.model_info['model'] = self.model_info['model'].to('cpu')
                del self.model_info['model']
            self.model_info['model'] = None
            self.model_info['device'] = 'offline'
        elif (isinstance(self.model_info['device'], numbers.Number)
              or str(self.model_info['device']).startswith('cuda')):
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = self.model_info['model'].to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True, ''

    def get_caption(self, image, prompt, kwargs, use_local, mask):
        image = image.convert('RGB')
        if use_local:
            image = get_region(image,
                               mask.convert('L'),
                               255)
        if image is None:
            return ""
        torch.manual_seed(int(time.time()) % 100000)
        query = self.model_info['tokenizer'].from_list_format([
            {
                'image': image
            },
            {
                'text': prompt
            },
        ])
        inputs = self.model_info['tokenizer'](query, return_tensors='pt')
        inputs = inputs.to(self.model_info['device'])
        pred = self.model_info['model'].generate(**inputs, **kwargs)
        response = self.model_info['tokenizer'].decode(
            pred.cpu()[0], skip_special_tokens=True)
        ret_caption = response.split(prompt)[-1]
        if ret_caption.startswith(','):
            ret_caption = ret_caption[1:]
        regex = re.compile(r'[' + '#®•©™&@·º½¾¿¡§~' + ')' + '(' + ']' + '[' +
                           '}' + '{' + '|' + '\\' + '/' + '*' +
                           r']{1,}')  # noqa: E501
        ret_caption = re.sub(regex, r' ', ret_caption)
        regex = re.compile(r'^[\-\_]+')
        ret_caption = re.sub(regex, r'', ret_caption)
        return ret_caption
    def __call__(self, **kwargs):
        target_image = kwargs.pop('target_image', None)
        src_mask = kwargs.pop('src_mask', None)
        src_image = kwargs.pop('src_image', None)
        use_preview = kwargs.pop('use_preview', True)
        sys_prompt = kwargs.pop('sys_prompt', 'Generate the caption in English')
        caption = kwargs.pop('caption', None)
        use_local = kwargs.pop('use_local', False)
        cache = kwargs.pop('cache', None)
        preview_target_image = kwargs.pop('preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.pop('preview_src_mask', None) if use_preview else src_mask
        preview_src_image = kwargs.pop('preview_src_image', None) if use_preview else src_image
        preview_caption = kwargs.pop('preview_caption', None) if use_preview else caption
        kwargs_keys = ["max_new_tokens", "min_new_tokens", "num_beams", "repetition_penalty", "temperature"]
        process_kwargs = {}
        for key in kwargs_keys:
            if key in kwargs:
                process_kwargs[key] = kwargs[key]
        response = ""
        if preview_src_image is not None:
            response += ("src_caption" + ": " +
                          self.get_caption(preview_src_image, sys_prompt, process_kwargs, use_local, preview_src_mask)  + "\n")
        if preview_target_image is not None:
            response += ("target_caption" + ": " +
                         self.get_caption(preview_target_image, sys_prompt, process_kwargs, use_local, preview_src_mask)  + "\n")
        return response


class QWVLQuantize(QWVL):
    def load_model(self):
        is_flg, msg = super(QWVL, self).load_model()
        if not is_flg:
            return is_flg, msg
        self.model = None
        if self.model_info['device'] == 'offline':
            try:
                from transformers import BitsAndBytesConfig
                torch.manual_seed(int(time.time()))
                from modelscope import (AutoModelForCausalLM, AutoTokenizer,
                                        GenerationConfig)
                local_model_dir = FS.get_dir_to_local_dir(self.model_path)
                # without quantization using 19.52G memory
                # with quantization using 7.7G memory
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=['lm_head', 'attn_pool.attn'])
                tokenizer = AutoTokenizer.from_pretrained(
                    local_model_dir, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_dir,
                    device_map='auto',
                    trust_remote_code=True,
                    fp16=True,
                    quantization_config=quantization_config).eval()
                model.generation_config = GenerationConfig.from_pretrained(
                    local_model_dir, trust_remote_code=True)
                # model.to(we.device_id)
            except Exception as e:
                if self.model is not None:
                    del self.model
                return False, f"Load model error '{e}'"
            self.model_info['device'] = model.device
            self.model_info['model'] = model
            self.model_info['tokenizer'] = tokenizer
        elif self.model_info['device'] == 'cpu':
            try:
                self.model_info['model'].to(we.device_id)
                self.model_info['device'] = we.device_id
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                del self.model_info['model']
                self.model_info['model'] = None
                self.model_info['device'] = 'offline'
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                return False, f"Load model error '{e}'"

        return True, ''

    def unload_model(self):
        print(self.model_info['device'])
        if (isinstance(self.model_info['device'], numbers.Number)
                or str(self.model_info['device']).startswith('cuda')):
            del self.model_info['model']
            self.model_info['model'] = None
            self.model_info['device'] = 'offline'
        torch.cuda.empty_cache()
        return True, ''


class InternVL15(QWVL):
    # AI-ModelScope/InternVL-Chat-V1-5
    def load_model(self):
        is_flg, msg = super(QWVL, self).load_model()
        if not is_flg:
            return is_flg, msg
        self.model = None
        if self.model_info['device'] == 'offline':
            try:
                local_path = FS.get_dir_to_local_dir(self.model_path)
                # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
                model = AutoModel.from_pretrained(
                    local_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True).eval().to(we.device_id)
                tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
                # model.to(we.device_id)
            except Exception as e:
                if self.model is not None:
                    del self.model
                return False, f"Load model error '{e}'"
            self.model_info['device'] = model.device
            self.model_info['model'] = model
            self.model_info['tokenizer'] = tokenizer
        elif self.model_info['device'] == 'cpu':
            try:
                self.model_info['model'].to(we.device_id)
                self.model_info['device'] = we.device_id
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                del self.model_info['model']
                self.model_info['model'] = None
                self.model_info['device'] = 'offline'
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                return False, f"Load model error '{e}'"

        return True, ''

    def unload_model(self):
        print(self.model_info['device'])
        if (isinstance(self.model_info['device'], numbers.Number)
                or str(self.model_info['device']).startswith('cuda')):
            del self.model_info['model']
            self.model_info['model'] = None
            self.model_info['device'] = 'offline'
        torch.cuda.empty_cache()
        return True, ''
    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=6):
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_caption(self, image, prompt, kwargs, use_local, mask):
        if use_local:
            image = get_region(image,
                               mask.convert('L'),
                               255)
        if image is None:
            return ""
        torch.manual_seed(int(time.time()) % 100000)
        generation_config = dict(
            num_beams=1,
            max_new_tokens=4096,
            do_sample=True
        )
        image = self.load_image(image, max_num=6).to(torch.bfloat16).cuda(we.device_id)
        response = self.model_info['model'].chat(self.model_info['tokenizer'], image, prompt, generation_config=generation_config)
        response = response.replace("\n", "").strip()
        return response


class CogVLM2Llama3Caption(BaseCaptionProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)
        self.model_path = cfg.MODEL_PATH
        self.model_info = {
            'device': 'offline',
            'model': None,
            'tokenizer': None
        }
        self.prompt = cfg.PROMPT
        self.temperature = cfg.TEMPERATURE
        self.max_new_tokens = cfg.MAX_NEW_TOKENS
        self.pad_token_id = cfg.PAD_TOKEN_ID
        self.top_k = cfg.TOP_K
        self.top_p = cfg.TOP_P
        self.TORCH_TYPE = torch.bfloat16 if (torch.cuda.is_available() and
                                             torch.cuda.get_device_capability()
                                             [0] >= 8) else torch.float16

    def load_model(self):
        is_flg, msg = super().load_model()
        if not is_flg:
            return is_flg, msg
        if self.model_info['device'] == 'offline':
            model = None
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                local_model_dir = FS.get_dir_to_local_dir(self.model_path)
                tokenizer = AutoTokenizer.from_pretrained(
                    local_model_dir,
                    trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_dir,
                    device_map='auto',
                    torch_dtype=self.TORCH_TYPE,
                    trust_remote_code=True
                ).eval().to(we.device_id)
            except Exception as e:
                if model is not None:
                    del model
                return False, f"Load model error '{e}'"
            self.model_info['device'] = model.device
            self.model_info['model'] = model
            self.model_info['tokenizer'] = tokenizer
        elif self.model_info['device'] == 'cpu':
            try:
                self.model_info['model'].to(we.device_id)
                self.model_info['device'] = we.device_id
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                del self.model_info['model']
                self.model_info['model'] = None
                self.model_info['device'] = 'offline'
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                return False, f"Load model error '{e}'"
        return True, ''

    def unload_model(self):
        super().unload_model()
        if self.delete_instance:
            self.model_info['device'] = 'offline'
            if self.model_info['model'] is not None:
                self.model_info['model'] = self.model_info['model'].to('cpu')
                del self.model_info['model']
            self.model_info['model'] = None
        elif (isinstance(self.model_info['device'], numbers.Number)
              or str(self.model_info['device']).startswith('cuda')):
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = self.model_info['model'].to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True, ''

    def load_video(self, video_data, strategy='chat'):
        bridge.set_bridge('torch')
        mp4_stream = video_data
        num_frames = 24
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

        frame_id_list = None
        total_frames = len(decord_vr)
        if strategy == 'base':
            clip_end_sec = 60
            clip_start_sec = 0
            start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
            end_frame = min(total_frames,
                            int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
            frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        elif strategy == 'chat':
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_id_list = []
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data

    def get_caption(self, prompt, video_data, temperature):
        strategy = 'chat'
        video = self.load_video(video_data, strategy=strategy)

        history = []
        query = prompt
        model = self.model_info['model']
        tokenizer = self.model_info['tokenizer']
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(we.device_id),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(we.device_id),
            'images': [[inputs['images'][0].to(we.device_id).to(self.TORCH_TYPE)]],
        }
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.pad_token_id,
            "top_k": self.top_k,
            "do_sample": True,
            "top_p": self.top_p,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

    def __call__(self, *args, **kwargs):
        video_path = kwargs.pop('video_path', None)
        with open(video_path, 'rb') as f:
            video_data = f.read()
        response = (self.get_caption(self.prompt, video_data, self.temperature))
        return response


class OpusMtZhEn(BaseCaptionProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)
        self.model_path = cfg.MODEL_PATH
        self.model_info = {
            'device': 'offline',
            'model': None,
            'tokenizer': None
        }

    def load_model(self):
        is_flg, msg = super().load_model()
        if not is_flg:
            return is_flg, msg
        if self.model_info['device'] == 'offline':
            model = None
            try:
                from transformers import MarianMTModel, AutoTokenizer
                local_model_dir = FS.get_dir_to_local_dir(self.model_path)
                tokenizer = AutoTokenizer.from_pretrained(
                    local_model_dir
                )
                model = MarianMTModel.from_pretrained(
                    local_model_dir
                ).to(we.device_id)
            except Exception as e:
                if model is not None:
                    del model
                return False, f"Load model error '{e}'"
            self.model_info['device'] = model.device
            self.model_info['model'] = model
            self.model_info['tokenizer'] = tokenizer
        elif self.model_info['device'] == 'cpu':
            try:
                self.model_info['model'].to(we.device_id)
                self.model_info['device'] = we.device_id
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                del self.model_info['model']
                self.model_info['model'] = None
                self.model_info['device'] = 'offline'
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                return False, f"Load model error '{e}'"
        return True, ''

    def unload_model(self):
        super().unload_model()
        if self.delete_instance:
            self.model_info['device'] = 'offline'
            if self.model_info['model'] is not None:
                self.model_info['model'] = self.model_info['model'].to('cpu')
                del self.model_info['model']
            self.model_info['model'] = None
        elif (isinstance(self.model_info['device'], numbers.Number)
              or str(self.model_info['device']).startswith('cuda')):
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = self.model_info['model'].to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True, ''

    def get_caption(self, data):
        model = self.model_info['model']
        tokenizer = self.model_info['tokenizer']
        batch = tokenizer(data, return_tensors="pt").to(we.device_id)
        generated_ids = model.generate(**batch)
        translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return translated_text

    def __call__(self, *args, **kwargs):
        caption = kwargs.pop('caption', None)
        response = (self.get_caption(caption))
        return response


class OpusMtEnZh(BaseCaptionProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)
        self.model_path = cfg.MODEL_PATH
        self.model_info = {
            'device': 'offline',
            'model': None,
            'tokenizer': None
        }

    def load_model(self):
        is_flg, msg = super().load_model()
        if not is_flg:
            return is_flg, msg
        if self.model_info['device'] == 'offline':
            model = None
            try:
                from transformers import MarianMTModel, AutoTokenizer
                local_model_dir = FS.get_dir_to_local_dir(self.model_path)
                tokenizer = AutoTokenizer.from_pretrained(
                    local_model_dir
                )
                model = MarianMTModel.from_pretrained(
                    local_model_dir
                ).to(we.device_id)
            except Exception as e:
                if model is not None:
                    del model
                return False, f"Load model error '{e}'"
            self.model_info['device'] = model.device
            self.model_info['model'] = model
            self.model_info['tokenizer'] = tokenizer
        elif self.model_info['device'] == 'cpu':
            try:
                self.model_info['model'].to(we.device_id)
                self.model_info['device'] = we.device_id
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                del self.model_info['model']
                self.model_info['model'] = None
                self.model_info['device'] = 'offline'
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                return False, f"Load model error '{e}'"
        return True, ''

    def unload_model(self):
        super().unload_model()
        if self.delete_instance:
            self.model_info['device'] = 'offline'
            if self.model_info['model'] is not None:
                self.model_info['model'] = self.model_info['model'].to('cpu')
                del self.model_info['model']
            self.model_info['model'] = None
        elif (isinstance(self.model_info['device'], numbers.Number)
              or str(self.model_info['device']).startswith('cuda')):
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = self.model_info['model'].to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True, ''

    def get_caption(self, data):
        model = self.model_info['model']
        tokenizer = self.model_info['tokenizer']
        batch = tokenizer(data, return_tensors="pt").to(we.device_id)
        generated_ids = model.generate(**batch)
        translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return translated_text

    def __call__(self, *args, **kwargs):
        caption = kwargs.pop('caption', None)
        response = (self.get_caption(caption))
        return response