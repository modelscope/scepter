# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os.path
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL.Image import Image
from scepter.modules.model.network.diffusion.diffusion import GaussianDiffusion
from scepter.modules.model.network.diffusion.schedules import noise_schedule
from scepter.modules.model.registry import (BACKBONES, EMBEDDERS, MODELS,
                                            TOKENIZERS)
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.studio.utils.env import get_available_memory

from .control_inference import ControlInference
from .diffusion_inference import DiffusionInference, get_model
from .tuner_inference import TunerInference


class PixArtInference(DiffusionInference):
    '''
        define vae, unet, text-encoder, tuner, refiner components
        support to load the components dynamicly.
        create and load model when run this model at the first time.
    '''
    def __init__(self, logger=None):
        self.logger = logger
        self.is_redefine_paras = False
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model'
        ]
        self.diffusion_insclass = GaussianDiffusion
        self.tuner_infer = TunerInference(self.logger)
        self.control_infer = ControlInference(self.logger)

    @torch.no_grad()
    def __call__(self,
                 input,
                 num_samples=1,
                 intermediate_callback=None,
                 refine_strength=0,
                 img_to_img_strength=0,
                 cat_uc=True,
                 tuner_model=None,
                 control_model=None,
                 **kwargs):

        value_input = copy.deepcopy(self.input)
        value_input.update(input)
        print(value_input)
        height, width = value_input['target_size_as_tuple']
        value_output = copy.deepcopy(self.output)

        # register tuner
        if tuner_model is not None and tuner_model != '' and len(
                tuner_model) > 0:
            if not isinstance(tuner_model, list):
                tuner_model = [tuner_model]
            self.dynamic_load(self.diffusion_model, 'diffusion_model')
            self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
            self.tuner_infer.register_tuner(tuner_model, self.diffusion_model,
                                            self.cond_stage_model)
            self.dynamic_unload(self.diffusion_model,
                                'diffusion_model',
                                skip_loaded=True)
            self.dynamic_unload(self.cond_stage_model,
                                'cond_stage_model',
                                skip_loaded=True)

        # cond stage
        self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
        function_name, dtype = self.get_function_info(self.cond_stage_model)
        with torch.autocast('cuda',
                            enabled=dtype == 'float16',
                            dtype=getattr(torch, dtype)):
            context, null_context = {}, {}
            cont_mask = None
            cont, cont_mask = getattr(get_model(self.cond_stage_model),
                                      function_name)(value_input['prompt'],
                                                     return_mask=True)
            context['crossattn'] = cont.float()
            self.dynamic_load(self.diffusion_model, 'diffusion_model')
            null_context['crossattn'] = get_model(
                self.diffusion_model).y_embedder.y_embedding[None].repeat(
                    num_samples, 1, 1)
            self.dynamic_unload(self.diffusion_model, 'diffusion_model')
        self.dynamic_unload(self.cond_stage_model,
                            'cond_stage_model',
                            skip_loaded=True)

        # get noise
        seed = kwargs.pop('seed', -1)
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        if 'seed' in value_output:
            value_output['seed'] = seed
        for sample_id in range(num_samples):
            if self.diffusion_model is not None:
                noise = torch.empty(
                    1,
                    4,
                    height // self.first_stage_model['paras']['size_factor'],
                    width // self.first_stage_model['paras']['size_factor'],
                    device=we.device_id).normal_(generator=g)

                self.dynamic_load(self.diffusion_model, 'diffusion_model')
                # UNet use input n_prompt
                function_name, dtype = self.get_function_info(
                    self.diffusion_model)
                with torch.autocast('cuda',
                                    enabled=dtype == 'float16',
                                    dtype=getattr(torch, dtype)):
                    latent = self.diffusion.sample(
                        solver=value_input.get('sample', 'ddim'),
                        noise=noise,
                        model=get_model(self.diffusion_model),
                        model_kwargs=[{
                            'cond': context,
                            'mask': cont_mask,
                            'data_info': {
                                'img_hw':
                                torch.tensor([[height, width]],
                                             dtype=torch.float,
                                             device=we.device_id).repeat(
                                                 num_samples, 1),
                                'aspect_ratio':
                                torch.tensor([[1.]],
                                             device=we.device_id).repeat(
                                                 num_samples, 1)
                            }
                        }, {
                            'cond': null_context,
                            'mask': cont_mask,
                            'data_info': {
                                'img_hw':
                                torch.tensor([[height, width]],
                                             dtype=torch.float,
                                             device=we.device_id).repeat(
                                                 num_samples, 1),
                                'aspect_ratio':
                                torch.tensor([[1.]],
                                             device=we.device_id).repeat(
                                                 num_samples, 1)
                            }
                        }],
                        cat_uc=False,
                        steps=value_input.get('sample_steps', 50),
                        guide_scale=value_input.get('guide_scale', 7.5),
                        guide_rescale=value_input.get('guide_rescale', 0.5),
                        discretization=value_input.get('discretization',
                                                       'trailing'),
                        show_progress=True,
                        seed=seed,
                        condition_fn=None,
                        clamp=None,
                        percentile=None,
                        t_max=None,
                        t_min=None,
                        discard_penultimate_step=None,
                        return_intermediate=None,
                        **kwargs)

                self.dynamic_unload(self.diffusion_model,
                                    'diffusion_model',
                                    skip_loaded=True)

            if 'latent' in value_output:
                if value_output['latent'] is None or (
                        isinstance(value_output['latent'], list)
                        and len(value_output['latent']) < 1):
                    value_output['latent'] = []
                value_output['latent'].append(latent)

            self.dynamic_load(self.first_stage_model, 'first_stage_model')
            x_samples = self.decode_first_stage(latent).float()
            self.dynamic_unload(self.first_stage_model,
                                'first_stage_model',
                                skip_loaded=True)
            images = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            if 'images' in value_output:
                if value_output['images'] is None or (
                        isinstance(value_output['images'], list)
                        and len(value_output['images']) < 1):
                    value_output['images'] = []
                value_output['images'].append(images)

        for k, v in value_output.items():
            if isinstance(v, list):
                value_output[k] = torch.cat(v, dim=0)
            if isinstance(v, torch.Tensor):
                value_output[k] = v.cpu()

        # unregister tuner
        if tuner_model is not None and tuner_model != '' and len(
                tuner_model) > 0:
            self.tuner_infer.unregister_tuner(tuner_model,
                                              self.diffusion_model,
                                              self.cond_stage_model)

        # unregister control
        if control_model is not None and control_model != '':
            self.control_infer.unregister_controllers(control_model,
                                                      self.diffusion_model)

        return value_output
