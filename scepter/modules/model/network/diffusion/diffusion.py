# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
GaussianDiffusion wraps operators for denoising diffusion models, including the
diffusion and denoising processes, as well as the loss evaluation.
"""
import copy
import random

import torch

from .schedules import karras_schedule
from .solvers import (sample_ddim, sample_dpm_2, sample_dpm_2_ancestral,
                      sample_dpmpp_2m, sample_dpmpp_2m_sde,
                      sample_dpmpp_2s_ancestral, sample_dpmpp_sde,
                      sample_euler, sample_euler_ancestral, sample_heun,
                      sample_img2img_euler, sample_img2img_euler_ancestral)

__all__ = ['GaussianDiffusion']


def _i(tensor, t, x):
    """
    Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).to(x.device)


class GaussianDiffusion(object):
    def __init__(self, sigmas, prediction_type='eps'):
        assert prediction_type in {'x0', 'eps', 'v'}
        self.sigmas = sigmas  # noise coefficients
        self.alphas = torch.sqrt(1 - sigmas**2)  # signal coefficients
        self.num_timesteps = len(sigmas)
        self.prediction_type = prediction_type

    def diffuse(self, x0, t, noise=None):
        """
        Add Gaussian noise to signal x0 according to:
        q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I).
        """
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt

    def denoise(self,
                xt,
                t,
                s,
                model,
                model_kwargs={},
                guide_scale=None,
                guide_rescale=None,
                clamp=None,
                percentile=None,
                cat_uc=False,
                **kwargs):
        """
        Apply one step of denoising from the posterior distribution q(x_s | x_t, x0).
        Since x0 is not available, estimate the denoising results using the learned
        distribution p(x_s | x_t, \hat{x}_0 == f(x_t)). # noqa
        """
        s = t - 1 if s is None else s

        # hyperparams
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.
        sigmas_s = torch.sqrt(1 - alphas_s**2)

        # precompute variables
        betas = 1 - (alphas / alphas_s)**2
        coef1 = betas * alphas_s / sigmas**2
        coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var = betas * (sigmas_s / sigmas)**2
        log_var = torch.log(var).clamp_(-20, 20)

        # prediction
        if guide_scale is None:
            assert isinstance(model_kwargs, dict)
            out = model(xt, t=t, **model_kwargs, **kwargs)
        else:
            # classifier-free guidance (arXiv:2207.12598)
            # model_kwargs[0]: conditional kwargs
            # model_kwargs[1]: non-conditional kwargs
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2

            if guide_scale == 1.:
                out = model(xt, t=t, **model_kwargs[0], **kwargs)
            else:
                if cat_uc:

                    def parse_model_kwargs(prev_value, value):
                        if isinstance(value, torch.Tensor):
                            prev_value = torch.cat([prev_value, value], dim=0)
                        elif isinstance(value, dict):
                            for k, v in value.items():
                                prev_value[k] = parse_model_kwargs(
                                    prev_value[k], v)
                        elif isinstance(value, list):
                            for idx, v in enumerate(value):
                                prev_value[idx] = parse_model_kwargs(
                                    prev_value[idx], v)
                        return prev_value

                    all_model_kwargs = copy.deepcopy(model_kwargs[0])
                    for model_kwarg in model_kwargs[1:]:
                        for key, value in model_kwarg.items():
                            all_model_kwargs[key] = parse_model_kwargs(
                                all_model_kwargs[key], value)
                    all_out = model(xt.repeat(2, 1, 1, 1),
                                    t=t.repeat(2),
                                    **all_model_kwargs,
                                    **kwargs)
                    y_out, u_out = all_out.chunk(2)
                else:
                    y_out = model(xt, t=t, **model_kwargs[0], **kwargs)
                    u_out = model(xt, t=t, **model_kwargs[1], **kwargs)
                out = u_out + guide_scale * (y_out - u_out)

                # rescale the output according to arXiv:2305.08891
                if guide_rescale is not None:
                    assert guide_rescale >= 0 and guide_rescale <= 1
                    ratio = (y_out.flatten(1).std(dim=1) /
                             (out.flatten(1).std(dim=1) +
                              1e-12)).view((-1, ) + (1, ) * (y_out.ndim - 1))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0
        # compute x0
        if self.prediction_type == 'x0':
            x0 = out
        elif self.prediction_type == 'eps':
            x0 = (xt - sigmas * out) / alphas
        elif self.prediction_type == 'v':
            x0 = alphas * xt - sigmas * out
        else:
            raise NotImplementedError(
                f'prediction_type {self.prediction_type} not implemented')

        # restrict the range of x0
        if percentile is not None:
            # NOTE: percentile should only be used when data is within range [-1, 1]
            assert percentile > 0 and percentile <= 1
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1, ) + (1, ) * (xt.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        # recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps

    def loss(self,
             x0,
             t,
             model,
             model_kwargs={},
             reduction='mean',
             noise=None,
             **kwargs):
        # hyperparams
        sigmas = _i(self.sigmas, t, x0)
        alphas = _i(self.alphas, t, x0)

        # diffuse and denoise
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.diffuse(x0, t, noise)
        out = model(xt, t=t, **model_kwargs, **kwargs)

        # mse loss
        target = {
            'eps': noise,
            'x0': x0,
            'v': alphas * noise - sigmas * x0
        }[self.prediction_type]
        loss = (out - target).pow(2)
        if reduction == 'mean':
            loss = loss.flatten(1).mean(dim=1)
        return loss

    @torch.no_grad()
    def sample(self,
               noise,
               model,
               x=None,
               denoising_strength=1.0,
               refine_stage=False,
               refine_strength=0.0,
               model_kwargs={},
               condition_fn=None,
               guide_scale=None,
               guide_rescale=None,
               clamp=None,
               percentile=None,
               solver='euler_a',
               steps=20,
               t_max=None,
               t_min=None,
               discretization=None,
               discard_penultimate_step=None,
               return_intermediate=None,
               show_progress=False,
               seed=-1,
               intermediate_callback=None,
               cat_uc=False,
               **kwargs):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')

        # function of diffusion solver
        solver_fn = {
            'ddim': sample_ddim,
            'euler_ancestral': sample_euler_ancestral,
            'euler': sample_euler,
            'heun': sample_heun,
            'dpm2': sample_dpm_2,
            'dpm2_ancestral': sample_dpm_2_ancestral,
            'dpmpp_2s_ancestral': sample_dpmpp_2s_ancestral,
            'dpmpp_2m': sample_dpmpp_2m,
            'dpmpp_sde': sample_dpmpp_sde,
            'dpmpp_2m_sde': sample_dpmpp_2m_sde,
            'dpm2_karras': sample_dpm_2,
            'dpm2_ancestral_karras': sample_dpm_2_ancestral,
            'dpmpp_2s_ancestral_karras': sample_dpmpp_2s_ancestral,
            'dpmpp_2m_karras': sample_dpmpp_2m,
            'dpmpp_sde_karras': sample_dpmpp_sde,
            'dpmpp_2m_sde_karras': sample_dpmpp_2m_sde
        }[solver]

        # options
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2', 'dpm2_ancestral', 'dpmpp_2m_sde', 'dpm2_karras',
                'dpm2_ancestral_karras', 'dpmpp_2m_sde_karras') else False

        # function for denoising xt to get x0
        intermediates = []

        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            x0 = self.denoise(xt,
                              t,
                              None,
                              model,
                              model_kwargs,
                              guide_scale,
                              guide_rescale,
                              clamp,
                              percentile,
                              cat_uc=cat_uc,
                              **kwargs)[-2]

            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                intermediates.append(x0)
            if intermediate_callback is not None:
                intermediate_callback(intermediates[-1])
            return x0

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented')
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(steps,
                                dtype=torch.float32,
                                device=noise.device)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        t_enc = int(min(denoising_strength, 0.999) * len(steps))
        sigmas = sigmas[len(steps) - t_enc - 1:]
        if refine_strength > 0:
            t_refine = int(min(refine_strength, 0.999) * len(steps))
            if refine_stage:
                sigmas = sigmas[-t_refine:]
            else:
                sigmas = sigmas[:-t_refine + 1]
        # print(sigmas)
        if x is not None:
            noise = (x + noise * sigmas[0]) / torch.sqrt(1.0 + sigmas[0]**2.0)

        if schedule == 'karras':
            if sigmas[0] == float('inf'):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float('inf')].max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([
                    sigmas.new_tensor([float('inf')]), sigmas,
                    sigmas.new_zeros([1])
                ])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        kwargs['seed'] = seed
        # sampling
        x0 = solver_fn(noise,
                       model_fn,
                       sigmas,
                       show_progress=show_progress,
                       **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0

    def _sigma_to_t(self, sigma):
        if sigma == float('inf'):
            t = torch.full_like(sigma, len(self.sigmas) - 1)
        else:
            log_sigmas = torch.sqrt(self.sigmas**2 /
                                    (1 - self.sigmas**2)).log().to(sigma)
            log_sigma = sigma.log()
            dists = log_sigma - log_sigmas[:, None]
            low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(
                max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
            low, high = log_sigmas[low_idx], log_sigmas[high_idx]
            w = (low - log_sigma) / (low - high)
            w = w.clamp(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            t = t.view(sigma.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t

    def _t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = torch.sqrt(self.sigmas**2 /
                                (1 - self.sigmas**2)).log().to(t)
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        log_sigma[torch.isnan(log_sigma)
                  | torch.isinf(log_sigma)] = float('inf')
        return log_sigma.exp()

    @torch.no_grad()
    def stochastic_encode(self, x0, t, steps):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas

        t_max = None
        t_min = None

        # discretization method
        discretization = 'trailing' if self.prediction_type == 'v' else 'leading'

        # timesteps
        if isinstance(steps, int):
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min
            steps = discretize_timesteps(t_max, t_min, steps, discretization)
        steps = torch.as_tensor(steps).round().long().flip(0).to(x0.device)
        # steps = torch.as_tensor(steps).round().long().to(x0.device)

        # self.alphas_bar = torch.cumprod(1 - self.sigmas ** 2, dim=0)
        # print('sigma: ', self.sigmas, len(self.sigmas))
        # print('alpha_bar: ', self.alphas_bar, len(self.alphas_bar))
        # print('steps: ', steps, len(steps))
        # sqrt_alphas_cumprod = torch.sqrt(self.alphas_bar).to(x0.device)[steps]
        # sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_bar).to(x0.device)[steps]

        sqrt_alphas_cumprod = self.alphas.to(x0.device)[steps]
        sqrt_one_minus_alphas_cumprod = self.sigmas.to(x0.device)[steps]
        # print('sigma: ', self.sigmas, len(self.sigmas))
        # print('alpha: ', self.alphas, len(self.alphas))
        # print('steps: ', steps, len(steps))

        noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) *
            noise)

    @torch.no_grad()
    def sample_img2img(self,
                       x,
                       noise,
                       model,
                       denoising_strength=1,
                       model_kwargs={},
                       condition_fn=None,
                       guide_scale=None,
                       guide_rescale=None,
                       clamp=None,
                       percentile=None,
                       solver='euler_a',
                       steps=20,
                       t_max=None,
                       t_min=None,
                       discretization=None,
                       discard_penultimate_step=None,
                       return_intermediate=None,
                       show_progress=False,
                       seed=-1,
                       **kwargs):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')
        # function of diffusion solver
        solver_fn = {
            'euler_ancestral': sample_img2img_euler_ancestral,
            'euler': sample_img2img_euler,
        }[solver]
        # options
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2', 'dpm2_ancestral', 'dpmpp_2m_sde', 'dpm2_karras',
                'dpm2_ancestral_karras', 'dpmpp_2m_sde_karras') else False

        # function for denoising xt to get x0
        intermediates = []

        def get_scalings(sigma):
            c_out = -sigma
            c_in = 1 / (sigma**2 + 1.**2)**0.5
            return c_out, c_in

        def model_fn(xt, sigma):
            # denoising
            c_out, c_in = get_scalings(sigma)
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()

            x0 = self.denoise(xt * c_in, t, None, model, model_kwargs,
                              guide_scale, guide_rescale, clamp, percentile,
                              **kwargs)[-2]
            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                intermediates.append(x0)
            return xt + x0 * c_out

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min
            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented')
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(steps, dtype=torch.float32, device=x.device)
        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        t_enc = int(min(denoising_strength, 0.999) * len(steps))
        sigmas = sigmas[len(steps) - t_enc - 1:]
        noise = x + noise * sigmas[0]

        if schedule == 'karras':
            if sigmas[0] == float('inf'):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float('inf')].max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([
                    sigmas.new_tensor([float('inf')]), sigmas,
                    sigmas.new_zeros([1])
                ])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        # sampling
        x0 = solver_fn(noise,
                       model_fn,
                       sigmas,
                       seed=seed,
                       show_progress=show_progress,
                       **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def discretize_timesteps(t_max, t_min, steps, discretization):
    """
    Implementation of timestep discretization methods.
    """
    if discretization == 'leading':
        steps = torch.arange(t_min, t_max + 1,
                             (t_max - t_min + 1) / steps).flip(0)
    elif discretization == 'linspace':
        steps = torch.linspace(t_max, t_min, steps)
    elif discretization == 'trailing':
        steps = torch.arange(t_max, t_min - 1, -((t_max - t_min + 1) / steps))
    else:
        raise NotImplementedError(
            f'{discretization} discretization not implemented')
    return steps.clamp_(t_min, t_max)
