# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers
import random

import torch
import torch.nn.functional as F
from scepter.modules.model.network.diffusion.diffusion import \
    GaussianDiffusionRF
from scepter.modules.model.network.diffusion.schedules import noise_schedule
from scepter.modules.model.network.ldm import LatentDiffusion
from scepter.modules.model.registry import MODELS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we


@MODELS.register_class()
class LatentDiffusionSD3(LatentDiffusion):
    para_dict = LatentDiffusion.para_dict

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

        self.shift_factor = cfg.get('SHIFT_FACTOR', 0)
        self.t_weight_type = cfg.get('T_WEIGHT', 'logit_normal')
        self.logit_mean = cfg.get('LOGIT_MEAN', 0.0)
        self.logit_std = cfg.get('LOGIT_STD', 1.0)

    def init_params(self):
        self.parameterization = self.cfg.get('PARAMETERIZATION', 'rf')
        assert self.parameterization in [
            'eps', 'x0', 'v', 'rf'
        ], 'currently only supporting "eps" and "x0" and "v" and "rf"'
        self.num_timesteps = self.cfg.get('TIMESTEPS', 1000)

        self.schedule_args = {
            k.lower(): v
            for k, v in self.cfg.get('SCHEDULE_ARGS', {
                'NAME': 'logsnr_cosine_interp',
                'SCALE_MIN': 2.0,
                'SCALE_MAX': 4.0
            }).items()
        }

        self.min_snr_gamma = self.cfg.get('MIN_SNR_GAMMA', None)

        self.zero_terminal_snr = self.cfg.get('ZERO_TERMINAL_SNR', False)
        if self.zero_terminal_snr:
            assert self.parameterization == 'v', 'Now zero_terminal_snr only support v-prediction mode.'

        self.sigmas = noise_schedule(schedule=self.schedule_args.pop('name'),
                                     n=self.num_timesteps,
                                     zero_terminal_snr=self.zero_terminal_snr,
                                     **self.schedule_args)

        self.diffusion = GaussianDiffusionRF(
            sigmas=self.sigmas, prediction_type=self.parameterization)

        self.pretrained_model = self.cfg.get('PRETRAINED_MODEL', None)
        self.ignore_keys = self.cfg.get('IGNORE_KEYS', [])

        self.model_config = self.cfg.DIFFUSION_MODEL
        self.first_stage_config = self.cfg.FIRST_STAGE_MODEL
        self.cond_stage_config = self.cfg.COND_STAGE_MODEL
        self.tokenizer_config = self.cfg.get('TOKENIZER', None)
        self.loss_config = self.cfg.get('LOSS', None)

        self.scale_factor = self.cfg.get('SCALE_FACTOR', 0.18215)
        self.size_factor = self.cfg.get('SIZE_FACTOR', 8)
        self.default_n_prompt = self.cfg.get('DEFAULT_N_PROMPT', '')
        self.default_n_prompt = '' if self.default_n_prompt is None else self.default_n_prompt
        self.p_zero = self.cfg.get('P_ZERO', 0.0)
        self.train_n_prompt = self.cfg.get('TRAIN_N_PROMPT', '')
        if self.default_n_prompt is None:
            self.default_n_prompt = ''
        if self.train_n_prompt is None:
            self.train_n_prompt = ''
        self.use_ema = self.cfg.get('USE_EMA', False)
        self.model_ema_config = self.cfg.get('DIFFUSION_MODEL_EMA', None)

    def noise_sample(self, batch_size, h, w, g, c=4):
        noise = torch.empty(batch_size, c, h, w,
                            device=we.device_id).normal_(generator=g)
        return noise

    def forward_train(self, image=None, noise=None, prompt=None, **kwargs):
        n, c, h, w = image.shape
        x_start = self.encode_first_stage(image, **kwargs)
        if self.t_weight_type == 'uniform':
            t = torch.randint(0,
                              self.num_timesteps, (n, ),
                              device=x_start.device).long()
        elif self.t_weight_type == 'logit_normal':
            density = F.sigmoid(
                torch.normal(mean=self.logit_mean,
                             std=self.logit_std,
                             size=(n, ),
                             device=x_start.device))
            t = (density * (self.num_timesteps - 1)).round().long()
            sigma = (t + 1) / self.num_timesteps
            shift = self.schedule_args['shift']
            if shift > 1.:
                sigma = shift * sigma / (1 + (shift - 1) * sigma)
            t = sigma * self.num_timesteps

        context = {}
        if prompt and self.cond_stage_model:
            ctx, pooled = getattr(self.cond_stage_model, 'encode')(prompt)
            context['crossattn'] = ctx.float()
            context['y'] = pooled
        else:
            assert False

        if self.min_snr_gamma is not None:
            alphas = self.diffusion.alphas.to(we.device_id)[t]
            sigmas = self.diffusion.sigmas.pow(2).to(we.device_id)[t]
            snrs = (alphas / sigmas).clamp(min=1e-20)
            min_snrs = snrs.clamp(max=self.min_snr_gamma)
            weights = min_snrs / snrs
        else:
            weights = 1
        self.register_probe({'snrs_weights': weights})

        loss = self.diffusion.loss(x0=x_start,
                                   t=t,
                                   model=self.model,
                                   model_kwargs={
                                       'cond': context,
                                   },
                                   noise=noise,
                                   **kwargs)
        loss = loss * weights
        loss = loss.mean()
        ret = {'loss': loss, 'probe_data': {'prompt': prompt}}
        return ret

    @torch.no_grad()
    def forward_test(self,
                     image=None,
                     prompt=None,
                     sampler='ddim',
                     sample_steps=20,
                     seed=2023,
                     guide_scale=4.5,
                     guide_rescale=0.0,
                     **kwargs):
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        num_samples = len(prompt)
        context = {}
        null_context = {}

        if prompt and self.cond_stage_model:
            ctx, pooled = getattr(self.cond_stage_model, 'encode')(prompt)
            null_ctx, null_pooled = getattr(self.cond_stage_model,
                                            'encode')([''] * len(prompt))
            context['crossattn'] = ctx.float()
            context['y'] = pooled.float()
            null_context['crossattn'] = null_ctx.float()
            null_context['y'] = null_pooled.float()
        else:
            assert False

        if 'index' in kwargs:
            kwargs.pop('index')
        image_size = None
        if 'meta' in kwargs:
            meta = kwargs.pop('meta')
            if 'image_size' in meta:
                h = int(meta['image_size'][0][0])
                w = int(meta['image_size'][1][0])
                image_size = [h, w]
        if 'image_size' in kwargs:
            image_size = kwargs.pop('image_size')
        if isinstance(image_size, numbers.Number):
            image_size = [image_size, image_size]
        if image_size is None:
            image_size = [1024, 1024]
        height, width = image_size
        noise = self.noise_sample(num_samples,
                                  height // self.size_factor,
                                  width // self.size_factor,
                                  g,
                                  c=16)
        # UNet use input n_prompt
        samples = self.diffusion.sample(
            solver=sampler,
            noise=noise,
            model=self.model,
            model_kwargs=[{
                'cond': context
            }, {
                'cond': null_context
            }] if guide_scale is not None and guide_scale > 0 else {
                'cond': context,
            },
            cat_uc=False,
            steps=sample_steps,
            guide_scale=guide_scale,
            guide_rescale=guide_rescale,
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
        x_samples = self.decode_first_stage(samples).float()
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        outputs = list()

        for i, (p, img) in enumerate(zip(prompt, x_samples)):
            one_tup = {'prompt': str(p), 'n_prompt': '', 'image': img}
            outputs.append(one_tup)

        return outputs

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            LatentDiffusionSD3.para_dict,
                            set_name=True)

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        z = self.first_stage_model.encode(x)
        return self.scale_factor * (z - self.shift_factor)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z + self.shift_factor
        return self.first_stage_model.decode(z)
