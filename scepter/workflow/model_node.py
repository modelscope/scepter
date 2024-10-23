# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os

from .constant import WORKFLOW_CONFIG, WORKFLOW_MODEL_PREFIX

class ModelNode:
    def __init__(self):
        from scepter.modules.utils.logger import get_logger
        self.pipeline = {}
        self.diff_infer = None
        self.cfg = WORKFLOW_CONFIG.workflow_config
        self.model_file = WORKFLOW_CONFIG.model_info
        self.logger = get_logger('scepter', level=logging.WARNING)

    CATEGORY = '🪄 ComfyUI-Scepter'

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'model': (list(s().model_file.keys()), ),
                "model_source": (list(s().cfg['MODEL_SOURCE']), ),
                'prompt': ('STRING', {
                    'multiline': True
                }),
                'negative_prompt': ('STRING', {
                    'multiline': True
                })
            },
            'optional': {
                'parameters': ('CONDITIONING', ),
                'mantras': ('CONDITIONING', ),
                'tuners': ('CONDITIONING', ),
                'controls': ('CONDITIONING', )
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('IMAGE', )
    FUNCTION = 'execute'

    def execute(self,
                model,
                model_source,
                prompt,
                negative_prompt,
                parameters=None,
                mantras=None,
                tuners=None,
                controls=None):
        data = self.format_parameters(model, model_source, prompt, negative_prompt,
                                      parameters, mantras, tuners, controls)
        cfg = self.model_file.get(model)['config']
        cfg = self.source_mapping(cfg, model_source)
        self.init_infer(model, cfg)
        output = self.diff_infer(data[0], **data[1])

        x = output['images'].permute(0, 2, 3, 1)
        output_image = x.unsqueeze(0)

        return output_image

    def source_mapping(self, cfg, source, type='model'):
        def mapping(str):
            if source == "Local":
                str = os.path.join(WORKFLOW_MODEL_PREFIX, str.split('/', 3)[-1].replace('@', '/'))
            elif source == "HuggingFace":
                str = str.replace('ms://iic/', 'hf://scepter-studio/')
            return str

        if type == 'model':
            if source == 'ModelScope':
                return cfg
            elif source == 'Local':
                cfg_new = copy.deepcopy(cfg)
                cfg_new.MODEL = cfg_new.MODEL_LOCAL
                return cfg_new
            elif source == 'HuggingFace':
                cfg_new = copy.deepcopy(cfg)
                cfg_new.MODEL = cfg_new.MODEL_HF
                return cfg_new
            else:
                raise NotImplementedError(f"Unknown model source: {source}")
        elif type in ['mantra', 'tuner', 'control']:
            if 'MODEL_PATH' in cfg and cfg.MODEL_PATH is not None:
                cfg.MODEL_PATH = mapping(cfg.MODEL_PATH)
            if 'IMAGE_PATH' in cfg and cfg.IMAGE_PATH is not None:
                cfg.IMAGE_PATH = mapping(cfg.IMAGE_PATH)
            return cfg
        else:
            raise NotImplementedError(f"Unknown model source: {source}")

    def init_infer(self, model_name, cfg):
        from scepter.modules.inference.diffusion_inference import DiffusionInference
        from scepter.modules.inference.sd3_inference import SD3Inference
        from scepter.modules.inference.pixart_inference import PixArtInference
        from scepter.modules.inference.flux_inference import FluxInference

        if model_name.startswith('PIXART'):
            infer_func = PixArtInference
        elif model_name.startswith('SD3'):
            infer_func = SD3Inference
        elif model_name.startswith('FLUX'):
            infer_func = FluxInference
        else:
            infer_func = DiffusionInference

        if model_name in self.pipeline:
            if not isinstance(self.diff_infer, infer_func):
                self.diff_infer.dynamic_unload(name='all')
                diff_infer = self.pipeline[model_name]
                diff_infer.dynamic_load(name='all')
            else:
                diff_infer = self.pipeline[model_name]
        else:
            if self.diff_infer is not None:
                self.diff_infer.dynamic_unload(name='all')
            diff_infer = infer_func(logger=self.logger)
            diff_infer.init_from_cfg(cfg)
            self.pipeline[model_name] = diff_infer
        self.diff_infer = diff_infer

    def format_parameters(self,
                          model,
                          model_source,
                          prompt,
                          negative_prompt,
                          parameters,
                          mantras,
                          tuners,
                          controls):
        input_data = {'prompt': prompt, 'negative_prompt': negative_prompt}
        input_params = {
            'diffusion_model': self.model_file.get(model)['diffusion_model'],
            'first_stage_model': self.model_file.get(model)['first_stage_model'],
            'cond_stage_model': self.model_file.get(model)['cond_stage_model']
        }

        if parameters:
            seed = parameters.get('seed', -1)
            input_params.update({'seed': seed})
            input_data.update(parameters)

        if mantras:
            prompt_template = mantras['prompt_template']
            negative_prompt_template = mantras['negative_prompt_template']
            if prompt_template != "":
                prompt = prompt_template.replace('{prompt}', prompt)
            if negative_prompt_template != "":
                negative_prompt = negative_prompt + ',' + negative_prompt_template if negative_prompt != "" else negative_prompt_template
            input_data['prompt'] = prompt
            input_data['negative_prompt'] = negative_prompt
            input_params.update({'mantra_state': True})

        if tuners:
            tuner_info = tuners['tuner_info']
            tuner_info = self.source_mapping(tuner_info, model_source, type='tuner')
            tuner_scale = tuners['tuner_scale']
            assert model == tuner_info['BASE_MODEL'], (
                'The tuner model is inconsistent with the base model, '
                'please ensure that the selected model is consistent')
            input_params.update({
                'tuner_state': True,
                'tuner_model': tuner_info,
                'tuner_scale': tuner_scale
            })

        if controls:
            controls['control_model'] = self.source_mapping(controls['control_model'], model_source, type='control')
            assert model == controls['control_model']['BASE_MODEL'], (
                'The control model is inconsistent with the base model, '
                'please ensure that the selected model is consistent')
            input_params.update(controls)
            input_params.update({'control_state': True})
        return [input_data, input_params]
