# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math
import random
import torch
from scepter.modules.utils.distribute import we
from .control_inference import ControlInference
from .diffusion_inference import DiffusionInference, get_model
from .tuner_inference import TunerInference
from scepter.modules.model.registry import DIFFUSIONS, TOKENIZERS


class FluxInference(DiffusionInference):
    def __init__(self, logger=None):
        self.logger = logger
        self.is_redefine_paras = False
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model'
        ]
        self.tuner_infer = TunerInference(self.logger)
        self.control_infer = ControlInference(self.logger)

    def init_from_cfg(self, cfg):
        self.name = cfg.NAME
        self.is_default = cfg.get('IS_DEFAULT', False)
        module_paras = self.load_default(cfg.get('DEFAULT_PARAS', None))
        assert cfg.have('MODEL')
        if self.is_redefine_paras:
            cfg.MODEL = self.redefine_paras(cfg.MODEL)
        self.diffusion_model = self.infer_model(
            cfg.MODEL.DIFFUSION_MODEL, module_paras.get(
                'DIFFUSION_MODEL',
                None)) if cfg.MODEL.have('DIFFUSION_MODEL') else None
        self.first_stage_model = self.infer_model(
            cfg.MODEL.FIRST_STAGE_MODEL,
            module_paras.get(
                'FIRST_STAGE_MODEL',
                None)) if cfg.MODEL.have('FIRST_STAGE_MODEL') else None
        self.cond_stage_model = self.infer_model(
            cfg.MODEL.COND_STAGE_MODEL,
            module_paras.get(
                'COND_STAGE_MODEL',
                None)) if cfg.MODEL.have('COND_STAGE_MODEL') else None
        self.refiner_cond_model = self.infer_model(
            cfg.MODEL.REFINER_COND_MODEL,
            module_paras.get(
                'REFINER_COND_MODEL',
                None)) if cfg.MODEL.have('REFINER_COND_MODEL') else None
        self.refiner_diffusion_model = self.infer_model(
            cfg.MODEL.REFINER_MODEL, module_paras.get(
                'REFINER_MODEL',
                None)) if cfg.MODEL.have('REFINER_MODEL') else None
        self.tokenizer = TOKENIZERS.build(
            cfg.MODEL.TOKENIZER,
            logger=self.logger) if cfg.MODEL.have('TOKENIZER') else None

        if self.tokenizer is not None:
            self.cond_stage_model['cfg'].KWARGS = {
                'vocab_size': self.tokenizer.vocab_size
            }
        self.diffusion = DIFFUSIONS.build(cfg.MODEL.DIFFUSION, logger=self.logger) \
            if cfg.MODEL.have('DIFFUSION') else None
        assert self.diffusion is not None

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        _, dtype = self.get_function_info(self.first_stage_model, 'encode')
        with torch.autocast('cuda',
                            enabled= dtype in ('float16', 'bfloat16'),
                            dtype=getattr(torch, dtype)):
            z = get_model(self.first_stage_model).encode(x)
            if isinstance(z, (tuple, list)):
                z = z[0]
            return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        _, dtype = self.get_function_info(self.first_stage_model, 'decode')
        with torch.autocast('cuda',
                            enabled=dtype in ('float16', 'bfloat16'),
                            dtype=getattr(torch, dtype)):
            return get_model(self.first_stage_model).decode(z)

    @torch.no_grad()
    def __call__(self,
                 input,
                 num_samples=1,
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
            ctx = getattr(get_model(self.cond_stage_model),
                                  function_name)(value_input['prompt'])
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
                noise = torch.randn(
                    num_samples,
                    16,
                    # allow for packing
                    2 * math.ceil(height / 16),
                    2 * math.ceil(width / 16),
                    device=we.device_id,
                    dtype=getattr(torch, dtype),
                    generator=torch.Generator(device=we.device_id).manual_seed(seed),
                )
                self.dynamic_load(self.diffusion_model, 'diffusion_model')
                # UNet use input n_prompt
                function_name, dtype = self.get_function_info(
                    self.diffusion_model)
                with torch.autocast('cuda',
                                    enabled= dtype in ('float16', 'bfloat16'),
                                    dtype=getattr(torch, dtype)):
                    solver_sample = value_input.get('sample', 'flow_eluer')
                    sample_steps = value_input.get('sample_steps', 20)
                    guide_scale = value_input.get('guide_scale', 3.5)
                    if guide_scale is not None:
                        guide_scale = torch.full((noise.shape[0],), guide_scale, device=noise.device,
                                                 dtype=noise.dtype)
                    else:
                        guide_scale = None

                    latent = self.diffusion.sample(
                        noise=noise,
                        sampler=solver_sample,
                        model=get_model(self.diffusion_model),
                        model_kwargs={"cond": ctx, "guidance": guide_scale},
                        steps=sample_steps,
                        show_progress=True,
                        guide_scale=guide_scale,
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
