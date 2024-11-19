# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

import torch

from scepter.modules.model.registry import DIFFUSION_SAMPLERS
from scepter.modules.utils.config import dict_to_yaml

from .util import _i


@dataclass
class SamplerOutput(object):
    callback_fn: callable
    prediction_type: str
    alphas: torch.Tensor
    alphas_bar: torch.Tensor
    betas: torch.Tensor
    sigmas: torch.Tensor
    alphas_init: torch.Tensor
    alphas_bar_init: torch.Tensor
    betas_init: torch.Tensor
    sigmas_init: torch.Tensor
    ts: torch.Tensor
    x_t: torch.Tensor
    x_0: torch.Tensor
    step: int
    steps: int
    msg: str

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


@DIFFUSION_SAMPLERS.register_class('base')
class BaseDiffusionSampler(object):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super(BaseDiffusionSampler, self).__init__()
        self.logger = logger
        self.cfg = cfg
        self.init_params()

    def init_params(self):
        self.discretization_type = self.cfg.get('DISCRETIZATION_TYPE',
                                                'linspace')
        self.discard_penultimate_step = self.cfg.get(
            'DISCARD_PENULTIMATE_STEP', False)
        self.free_steps = self.cfg.get('FREE_STEPS', None)
        self.t_max = self.cfg.get('T_MAX', None)
        self.t_min = self.cfg.get('T_MIN', None)

    def discretization(self, steps=20, num_timesteps=1000, reverse_scale = -1., **kwargs):
        # get timesteps
        if isinstance(steps, int):
            steps += 1 if self.discard_penultimate_step else 0
            t_max = num_timesteps - 1 if self.t_max is None else self.t_max
            t_min = 0 if self.t_min is None else self.t_min

            # discretize timesteps
            if self.discretization_type == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)
            elif self.discretization_type == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif self.discretization_type == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))
            elif self.discretization_type == 'free':
                steps = torch.tensor(self.free_steps)
            else:
                raise NotImplementedError(
                    f'{self.discretization_type} discretization not implemented'
                )
            steps = steps.clamp_(t_min, t_max)
        elif isinstance(steps, list):
            steps = torch.tensor(steps)
        if reverse_scale >=0:
            img2img_step = int((1 - reverse_scale) * len(steps))
            timesteps = torch.as_tensor(steps[img2img_step:], dtype=torch.float32)
            return timesteps
        return torch.as_tensor(steps, dtype=torch.float32)

    def preprare_sampler(self,
                         noise,
                         x=None,
                         steps=20,
                         reverse_scale=-1.,
                         scheduler_ins=None,
                         prediction_type='',
                         sigmas=None,
                         betas=None,
                         alphas=None,
                         alphas_bar=None,
                         callback_fn=None,
                         **kwargs):
        '''
            1. Control the model's inputs and outputs externally in the solver by callback_fn,
            and perform the conversion between x0 and xt internally within the solver.
            2. The function callback_fn use the x0 and xt as the default inputs and also give me
            the x0 and xt as output. The other inputs will be set in kwargs.
            3. The basic parameters of the schedule should be set manually.
            4. To ensure the safety of threading, use the instance of SamplerOutput as the manager,
            which manage all necessary information.
        '''
        if reverse_scale >= 0:
            assert x is not None
        num_timesteps = scheduler_ins.num_timesteps if scheduler_ins is not None else 1000
        timestamps = self.discretization(steps,
                                         num_timesteps=num_timesteps,
                                         reverse_scale=reverse_scale,
                                         **kwargs)
        alphas = scheduler_ins.t_to_alpha(
            timestamps, **kwargs) if scheduler_ins is not None else alphas
        alphas_bar = scheduler_ins.t_to_alpha_bar(
            timestamps, **kwargs) if scheduler_ins is not None else alphas_bar
        betas = scheduler_ins.t_to_beta(
            timestamps, **kwargs) if scheduler_ins is not None else betas
        sigmas = scheduler_ins.t_to_sigma(
            timestamps, **kwargs) if scheduler_ins is not None else sigmas
        alphas_init = scheduler_ins.t_to_alpha_init(
            timestamps, **kwargs) if scheduler_ins is not None else alphas

        alphas_bar_init = scheduler_ins.t_to_alpha_bar_init(
            timestamps, **kwargs) if scheduler_ins is not None else alphas_bar

        betas_init = scheduler_ins.t_to_beta_init(
            timestamps, **kwargs) if scheduler_ins is not None else betas
        sigmas_init = scheduler_ins.t_to_sigma_init(
            timestamps, **kwargs) if scheduler_ins is not None else sigmas
        if reverse_scale >= 0:
            x_t = x_0 = scheduler_ins.add_noise(x, noise=noise, t=timestamps[0].repeat(x.size(0)).to(x.device)).x_t if len(timestamps) > 0 else x
        else:
            x_t = x_0 = noise
        # Consider the sigma's list is from sigma_ to zero. the steps equal to len(timestamps)
        output = SamplerOutput(callback_fn=callback_fn,
                               prediction_type=prediction_type,
                               alphas=alphas,
                               alphas_bar=alphas_bar,
                               betas=betas,
                               sigmas=sigmas,
                               alphas_init=alphas_init,
                               alphas_bar_init=alphas_bar_init,
                               betas_init=betas_init,
                               sigmas_init=sigmas_init,
                               ts=timestamps,
                               x_t=x_t,
                               x_0=x_0,
                               step=0,
                               msg='step 0',
                               steps=len(timestamps) - 1)
        return output

    def step(self, sampler_ouput):
        raise NotImplementedError(
            'DiffusionSampler step function not implemented')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' + ' ' + super().__repr__()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DIFFUSION_SAMPLERS',
                            __class__.__name__,
                            BaseDiffusionSampler.para_dict,
                            set_name=True)


@DIFFUSION_SAMPLERS.register_class('eluer')
class EulerSampler(BaseDiffusionSampler):
    def step(self, sampler_ouput):
        pass


@DIFFUSION_SAMPLERS.register_class('ddim')
class DDIMSampler(BaseDiffusionSampler):
    def init_params(self):
        super().init_params()
        self.eta = self.cfg.get('ETA', 0.)
        self.discretization_type = self.cfg.get('DISCRETIZATION_TYPE',
                                                'trailing')

    def preprare_sampler(self,
                         noise,
                         x=None,
                         steps=20,
                         reverse_scale = -1.,
                         scheduler_ins=None,
                         prediction_type='',
                         sigmas=None,
                         betas=None,
                         alphas=None,
                         alphas_bar=None,
                         callback_fn=None,
                         **kwargs):
        output = super().preprare_sampler(noise,
                                          x = x,
                                          steps = steps,
                                          reverse_scale = reverse_scale,
                                          scheduler_ins = scheduler_ins,
                                          prediction_type = prediction_type,
                                          sigmas = sigmas,
                                          betas = betas,
                                          alphas = alphas,
                                          alphas_bar = alphas_bar,
                                          callback_fn = callback_fn,
                                          **kwargs)
        sigmas = output.sigmas
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        sigmas_vp = (sigmas**2 / (1 + sigmas**2))**0.5
        sigmas_vp[sigmas == float('inf')] = 1.
        output.add_custom_field('sigmas_vp', sigmas_vp)
        output.steps += 1
        return output

    def step(self, sampler_output):
        x_t = sampler_output.x_t
        step = sampler_output.step
        t = sampler_output.ts[step]
        sigmas_vp = sampler_output.sigmas_vp.to(x_t.device)
        alpha_bar_init = _i(sampler_output.alphas_bar_init, step, x_t[:1])
        sigma_init = _i(sampler_output.sigmas_init, step, x_t[:1])

        x = sampler_output.callback_fn(x_t, t, sigma_init, alpha_bar_init)
        noise_factor = self.eta * (sigmas_vp[step + 1]**2 /
                                   sigmas_vp[step]**2 *
                                   (1 - (1 - sigmas_vp[step]**2) /
                                    (1 - sigmas_vp[step + 1]**2)))
        d = (x_t - (1 - sigmas_vp[step]**2)**0.5 * x) / sigmas_vp[step]
        x = (1 - sigmas_vp[step + 1] ** 2) ** 0.5 * x + \
            (sigmas_vp[step + 1] ** 2 - noise_factor ** 2) ** 0.5 * d
        sampler_output.x_0 = x
        if sigmas_vp[step + 1] > 0:
            x += noise_factor * torch.randn_like(x)
        sampler_output.x_t = x
        sampler_output.step += 1
        sampler_output.msg = f'step {step}'
        return sampler_output


@DIFFUSION_SAMPLERS.register_class('flow_euler')
class FlowEluerSampler(BaseDiffusionSampler):
    def preprare_sampler(self,
                         noise,
                         x=None,
                         steps=20,
                         reverse_scale = -1.,
                         scheduler_ins=None,
                         prediction_type='',
                         sigmas=None,
                         betas=None,
                         alphas=None,
                         alphas_bar=None,
                         callback_fn=None,
                         **kwargs):
        if noise.ndim == 3:
            seq_len = noise.shape[2] // 4
        else:
            n, _, h, w = noise.shape
            seq_len = (h // 2 * w // 2)
        kwargs['seq_len'] = seq_len
        output = super().preprare_sampler(noise,
                                          x = x,
                                          steps = steps,
                                          reverse_scale = reverse_scale,
                                          scheduler_ins = scheduler_ins,
                                          prediction_type = prediction_type,
                                          sigmas = sigmas,
                                          betas = betas,
                                          alphas = alphas,
                                          alphas_bar = alphas_bar,
                                          callback_fn = callback_fn,
                                          **kwargs)
        return output

    def step(self, sampler_output):
        step = sampler_output.step
        x_t = sampler_output.x_t
        sigma_curr, sigma_prev = sampler_output.sigmas[
            step], sampler_output.sigmas[step + 1]
        prediction_type = sampler_output.prediction_type
        assert prediction_type in ('raw', 'sigma_scaled')
        t = sampler_output.ts[step]
        x_0 = sampler_output.callback_fn(x_t, t, sigma_curr)
        x_t = x_t + (sigma_prev - sigma_curr) * x_0
        sampler_output.x_0 = x_0
        sampler_output.x_t = x_t
        sampler_output.step += 1
        sampler_output.msg = f'step {step}, sigma_curr: {sigma_curr}, sigma_prev: {sigma_prev}'
        return sampler_output

    def discretization(self, steps=20, num_timesteps=1000, reverse_scale=-1., **kwargs):
        # extra step for zero
        timesteps = torch.linspace(num_timesteps, 0, steps + 1)
        if reverse_scale >= 0:
            img2img_step = int((1 - reverse_scale) * len(timesteps))
            timesteps = timesteps[img2img_step:]
            return timesteps
        return timesteps

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DIFFUSION_SAMPLERS',
                            __class__.__name__,
                            FlowEluerSampler.para_dict,
                            set_name=True)
