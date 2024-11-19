# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from collections import OrderedDict

import torch
from tqdm import trange

from scepter.modules.model.registry import (DIFFUSION_SAMPLERS, DIFFUSIONS,
                                            NOISE_SCHEDULERS)
from scepter.modules.utils.config import Config, dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


@DIFFUSIONS.register_class()
class BaseDiffusion(object):
    para_dict = {
        'NOISE_SCHEDULER': {},
        'SAMPLER_SCHEDULER': {},
        'PREDICTION_TYPE': {
            'value': 'eps',
            'description':
            'The type of prediction to use for the loss function.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(BaseDiffusion, self).__init__()
        self.logger = logger
        self.cfg = cfg
        self.init_params()

    def init_params(self):
        self.prediction_type = self.cfg.get('PREDICTION_TYPE', 'eps')
        self.noise_scheduler = NOISE_SCHEDULERS.build(self.cfg.NOISE_SCHEDULER,
                                                      logger=self.logger)
        self.sampler_scheduler = NOISE_SCHEDULERS.build(self.cfg.get(
            'SAMPLER_SCHEDULER', self.cfg.NOISE_SCHEDULER),
                                                        logger=self.logger)
        self.num_timesteps = self.noise_scheduler.num_timesteps
        if self.cfg.have('WORK_DIR') and we.rank == 0:
            schedule_visualization = os.path.join(self.cfg.WORK_DIR,
                                                  'noise_schedule.png')
            with FS.put_to(schedule_visualization) as local_path:
                self.noise_scheduler.plot_noise_sampling_map(local_path)
            schedule_visualization = os.path.join(self.cfg.WORK_DIR,
                                                  'sampler_schedule.png')
            with FS.put_to(schedule_visualization) as local_path:
                self.sampler_scheduler.plot_noise_sampling_map(local_path)

    def sample(self,
               noise,
               model,
               model_kwargs={},
               steps=20,
               sampler=None,
               use_dynamic_cfg=False,
               guide_scale=None,
               guide_rescale=None,
               show_progress=False,
               return_intermediate=None,
               intermediate_callback=None,
               reverse_scale = -1.,
               x = None,
               **kwargs):
        assert isinstance(steps, (int, torch.LongTensor))
        assert return_intermediate in (None, 'x0', 'xt')
        assert isinstance(sampler, (str, dict, Config))
        intermediates = []

        def callback_fn(x_t, t, sigma=None, alpha_bar=None):
            timestamp = t
            t = t.repeat(len(x_t)).round().long().to(x_t.device)
            sigma = sigma.repeat(len(x_t), *([1] * (len(sigma.shape) - 1)))
            alpha_bar = alpha_bar.repeat(len(x_t), *([1] * (len(alpha_bar.shape) - 1)))

            if guide_scale is None or guide_scale == 1.0:
                out = model(x=x_t, t=t, **model_kwargs)
            else:
                if use_dynamic_cfg:
                    guidance_scale = 1 + guide_scale * (
                        (1 - math.cos(math.pi * (
                            (steps - timestamp.item()) / steps)**5.0)) / 2)
                else:
                    guidance_scale = guide_scale
                y_out = model(x=x_t, t=t, **model_kwargs[0])
                u_out = model(x=x_t, t=t, **model_kwargs[1])
                out = u_out + guidance_scale * (y_out - u_out)
            if guide_rescale is not None and guide_rescale > 0.0:
                ratio = (
                    y_out.flatten(1).std(dim=1) /
                    (out.flatten(1).std(dim=1) + 1e-12)).view((-1, ) + (1, ) *
                                                              (y_out.ndim - 1))
                out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0

            if self.prediction_type == 'x0':
                x0 = out
            elif self.prediction_type == 'eps':
                x0 = (x_t - sigma * out) / alpha_bar
            elif self.prediction_type == 'v':
                x0 = alpha_bar * x_t - sigma * out
            else:
                raise NotImplementedError(
                    f'prediction_type {self.prediction_type} not implemented')
            return x0

        sampler_ins = self.get_sampler(sampler)

        # this is ignored for schnell
        sampler_output = sampler_ins.preprare_sampler(
            noise,
            x = x,
            steps=steps,
            reverse_scale= reverse_scale,
            prediction_type=self.prediction_type,
            scheduler_ins=self.sampler_scheduler,
            callback_fn=callback_fn)

        for _ in trange(sampler_output.steps, disable=not show_progress):
            trange.desc = sampler_output.msg
            sampler_output = sampler_ins.step(sampler_output)
            if return_intermediate == 'x_0':
                intermediates.append(sampler_output.x_0)
            elif return_intermediate == 'x_t':
                intermediates.append(sampler_output.x_t)
            if intermediate_callback is not None:
                intermediate_callback(intermediates[-1])
        return (sampler_output.x_0, intermediates
                ) if return_intermediate is not None else sampler_output.x_0

    def loss(self,
             x_0,
             model,
             model_kwargs={},
             reduction='mean',
             noise=None,
             **kwargs):
        # use noise scheduler to add noise
        if noise is None:
            noise = torch.randn_like(x_0)
        schedule_output = self.noise_scheduler.add_noise(x_0, noise, **kwargs)
        x_t, t, sigma, alpha_bar = schedule_output.x_t, schedule_output.t, schedule_output.sigma, schedule_output.alpha_bar
        out = model(x=x_t, t=t, **model_kwargs)

        # mse loss
        target = {
            'eps': noise,
            'x0': x_0,
            'v': alpha_bar * noise - sigma * x_0
        }[self.prediction_type]

        loss = (out - target).pow(2)
        if reduction == 'mean':
            loss = loss.flatten(1).mean(dim=1)
        return loss

    def get_sampler(self, sampler):
        if isinstance(sampler, str):
            if sampler not in DIFFUSION_SAMPLERS.class_map:
                if self.logger is not None:
                    self.logger.info(
                        f'{sampler} not in the defined samplers list {DIFFUSION_SAMPLERS.class_map.keys()}'
                    )
                else:
                    print(
                        f'{sampler} not in the defined samplers list {DIFFUSION_SAMPLERS.class_map.keys()}'
                    )
                return None
            sampler_cfg = Config(cfg_dict={'NAME': sampler}, load=False)
            sampler_ins = DIFFUSION_SAMPLERS.build(sampler_cfg,
                                                   logger=self.logger)
        elif isinstance(sampler, (Config, dict, OrderedDict)):
            if isinstance(sampler, (dict, OrderedDict)):
                sampler = Config(
                    cfg_dict={k.upper(): v
                              for k, v in dict(sampler).items()},
                    load=False)
            sampler_ins = DIFFUSION_SAMPLERS.build(sampler, logger=self.logger)
        else:
            raise NotImplementedError
        return sampler_ins

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' + ' ' + super().__repr__()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DIFFUSIONS',
                            __class__.__name__,
                            BaseDiffusion.para_dict,
                            set_name=True)


@DIFFUSIONS.register_class()
class DiffusionFluxRF(BaseDiffusion):
    para_dict = {
        'PREDICTION_TYPE': {
            'value': 'raw',
            'description':
            'The type of prediction to use for the loss function.'
        }
    }
    para_dict.update(BaseDiffusion.para_dict)

    def __init__(self, cfg, logger=None):
        super(DiffusionFluxRF, self).__init__(cfg, logger=logger)
        self.prediction_type = self.cfg.get('PREDICTION_TYPE', 'raw')

    def loss(self,
             x_0,
             model,
             model_kwargs={},
             reduction='mean',
             noise=None,
             **kwargs):
        if noise is None:
            noise = torch.randn_like(x_0)
        schedule_output = self.noise_scheduler.add_noise(x_0, noise, **kwargs)
        x_t, t, sigma = schedule_output.x_t, schedule_output.t, schedule_output.sigma
        out = model(x=x_t, t=sigma, **model_kwargs)
        # raw
        if self.prediction_type == 'raw':
            target = noise - x_0
            out = out
        elif self.prediction_type == 'sigma_scaled':
            target = x_0
            out = out * (-sigma) + x_t
        else:
            raise NotImplementedError

        loss = (target - out)**2
        if reduction == 'mean':
            loss = loss.flatten(1).mean(dim=1)
        return loss

    @torch.no_grad()
    def sample(self,
               noise,
               model,
               model_kwargs={},
               steps=20,
               sampler=None,
               show_progress=False,
               return_intermediate=None,
               intermediate_callback=None,
               reverse_scale=-1.,
               x=None,
               **kwargs):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert return_intermediate in (None, 'x0', 'xt')
        assert isinstance(sampler, (str, dict, Config))
        intermediates = []

        def callback_fn(x_t, t, sigma=None, alpha_bar=None):
            sigma = torch.full((x_t.shape[0], ),
                               sigma,
                               dtype=x_t.dtype,
                               device=x_t.device)
            x_0 = model(x=x_t, t=sigma, **model_kwargs)
            return x_0

        sampler_ins = self.get_sampler(sampler)

        # this is ignored for schnell
        sampler_output = sampler_ins.preprare_sampler(
            noise,
            x=x,
            steps=steps,
            reverse_scale=reverse_scale,
            prediction_type=self.prediction_type,
            scheduler_ins=self.sampler_scheduler,
            callback_fn=callback_fn)

        for _ in trange(sampler_output.steps, disable=not show_progress):
            trange.desc = sampler_output.msg
            sampler_output = sampler_ins.step(sampler_output)
            if return_intermediate == 'x_0':
                intermediates.append(sampler_output.x_0)
            elif return_intermediate == 'x_t':
                intermediates.append(sampler_output.x_t)
            if intermediate_callback is not None:
                intermediate_callback(intermediates[-1])
        return (sampler_output.x_0, intermediates
                ) if return_intermediate is not None else sampler_output.x_t

    @staticmethod
    def get_config_template():
        return dict_to_yaml('DIFFUSIONS',
                            __class__.__name__,
                            DiffusionFluxRF.para_dict,
                            set_name=True)
