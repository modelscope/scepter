# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import numpy as np
from typing import Tuple
import random

import torch

from scepter.modules.utils.file_system import FS
from scepter.modules.utils.distribute import we
from scepter.modules.model.backbone.cogvideox.utils import get_3d_rotary_pos_embed, get_resize_crop_region_for_grid
from .diffusion_inference import DiffusionInference, get_model
from .tuner_inference import TunerInference

class CogVideoXInference(DiffusionInference):
    def __init__(self, logger=None):
        self.logger = logger
        self.is_redefine_paras = False
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model'
        ]
        self.tuner_infer = TunerInference(self.logger)

    @torch.no_grad()
    def decode_first_stage(self, latents):
        _, dtype = self.get_function_info(self.first_stage_model, 'decode')
        with torch.autocast('cuda',
                            enabled=dtype in ('bfloat16'),
                            dtype=getattr(torch, dtype)):
            latents = latents.permute(0, 2, 1, 3, 4)
            latents = 1 / self.first_stage_model['paras']['scaling_factor_image'] * latents
            frames = get_model(self.first_stage_model).decode(latents)
        return frames

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        grid_height = height // (self.diffusion_model['paras']['scale_factor_spatial'] * self.diffusion_model['paras']['patch_size'])
        grid_width = width // (self.diffusion_model['paras']['scale_factor_spatial'] * self.diffusion_model['paras']['patch_size'])
        base_size_width = self.diffusion_model['paras']['sample_width'] // (self.diffusion_model['paras']['scale_factor_spatial'] * self.diffusion_model['paras']['patch_size'])
        base_size_height = self.diffusion_model['paras']['sample_height'] // (self.diffusion_model['paras']['scale_factor_spatial'] * self.diffusion_model['paras']['patch_size'])

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.diffusion_model['paras']['attention_head_dim'],
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def __call__(self,
                 input,
                 num_samples=1,
                 cat_uc=True,
                 tuner_model=None,
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
            self.tuner_infer.register_tuner(tuner_model, self.diffusion_model,
                                            cond_stage_model=None)
            self.dynamic_unload(self.diffusion_model,
                                'diffusion_model',
                                skip_loaded=True)

        # cond stage
        self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
        function_name, dtype = self.get_function_info(self.cond_stage_model)
        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            cont = getattr(get_model(self.cond_stage_model),
                                  function_name)(value_input['prompt'], return_mask=False, use_mask=False)
            null_cont = getattr(get_model(self.cond_stage_model),
                                        function_name)(value_input['negative_prompt'] * num_samples, return_mask=False, use_mask=False)
        self.dynamic_unload(self.cond_stage_model,
                            'cond_stage_model',
                            skip_loaded=True)

        # get noise
        seed = kwargs.pop('seed', -1)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator().manual_seed(seed)
        if 'seed' in value_output:
            value_output['seed'] = seed
        for sample_id in range(num_samples):
            if self.diffusion_model is not None:
                noise_shape = (1,
                         (value_input['num_frames'] - 1) // self.diffusion_model['paras']['scale_factor_temporal'] + 1,
                         self.diffusion_model['paras']['latent_channels'],
                         height // self.diffusion_model['paras']['scale_factor_spatial'],
                         width // self.diffusion_model['paras']['scale_factor_spatial']
                         )
                noise = torch.randn(noise_shape, generator=generator, dtype=getattr(torch, dtype), device='cpu').to(we.device_id)

                self.dynamic_load(self.diffusion_model, 'diffusion_model')

                image_rotary_emb = (
                    self._prepare_rotary_positional_embeddings(height, width, noise.size(1), we.device_id)
                    if self.diffusion_model['paras']['use_rotary_positional_embeddings']
                    else None
                )
                function_name, dtype = self.get_function_info(
                    self.diffusion_model)

                with torch.autocast('cuda',
                                    enabled=dtype in ('float16', 'bfloat16'),
                                    dtype=getattr(torch, dtype)):
                    solver_sample = value_input.get('sample', 'ddim')
                    sample_steps = value_input.get('sample_steps', 50)
                    guide_scale = value_input.get('guide_scale', 7.5)
                    guide_rescale = value_input.get('guide_rescale', 0.5)

                    latent = self.diffusion.sample(noise=noise,
                                        sampler=solver_sample,
                                        model=get_model(self.diffusion_model),
                                        model_kwargs=[{
                                            'cond': cont,
                                            'image_latent': None,
                                            'image_rotary_emb': image_rotary_emb,
                                        }, {
                                            'cond': null_cont,
                                            'image_latent': None,
                                            'image_rotary_emb': image_rotary_emb,
                                        }],
                                        steps=sample_steps,
                                        show_progress=True,
                                        guide_scale=guide_scale,
                                        guide_rescale=guide_rescale,
                                        return_intermediate=None,
                                        **kwargs).float()
                self.dynamic_unload(self.diffusion_model,
                                    'diffusion_model',
                                    skip_loaded=True)
            self.dynamic_load(self.first_stage_model, 'first_stage_model')
            x_samples = self.decode_first_stage(latent).float()  # [B, C, F, H, W]
            self.dynamic_unload(self.first_stage_model,
                                'first_stage_model',
                                skip_loaded=True)

            x_frames = torch.clamp(x_samples / 2 + 0.5, min=0.0, max=1.0)
            if 'videos' in value_output:
                if value_output['videos'] is None or (
                        isinstance(value_output['videos'], list)
                        and len(value_output['videos']) < 1):
                    value_output['videos'] = []
                value_output['videos'].append(x_frames)

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
                                              cond_stage_model=None)
        return value_output
