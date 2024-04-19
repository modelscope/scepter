# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import re
import time

import torch
from PIL import Image

from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.processors.base_processor import \
    BaseCaptionProcessor

__all__ = ['BlipImageBase', 'QWVL', 'QWVLQuantize']


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

    def __call__(self, image, prompt=None, **kwargs):
        raw_image = Image.open(image).convert('RGB')
        inputs = self.model_info['processor'](
            raw_image, return_tensors='pt').to(we.device_id)
        out = self.model_info['model'].generate(**inputs)
        return self.model_info['processor'].decode(out[0],
                                                   skip_special_tokens=True)


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

    def __call__(self,
                 image,
                 prompt='Generate the caption in English',
                 **kwargs):

        torch.manual_seed(int(time.time()) % 100000)
        query = self.model_info['tokenizer'].from_list_format([
            {
                'image': image
            },
            {
                'text': prompt
            },
        ])
        print(kwargs)

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
