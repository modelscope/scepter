# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import numbers
import random
from collections import OrderedDict

import torch

from scepter.modules.model.network.diffusion.diffusion import GaussianDiffusion
from scepter.modules.model.network.diffusion.schedules import noise_schedule
from scepter.modules.model.network.ldm import LatentDiffusion
from scepter.modules.model.network.train_module import TrainModule
from scepter.modules.model.registry import (BACKBONES, EMBEDDERS, LOSSES,
                                            MODELS, TOKENIZERS)
from scepter.modules.model.utils.basic_utils import count_params, default
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@MODELS.register_class()
class LatentDiffusionPixart(LatentDiffusion):
    para_dict = LatentDiffusion.para_dict
    para_dict['DECODER_BIAS'] = {'value': 0, 'description': ''}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.decoder_bias = cfg.get('DECODER_BIAS', 0.5)

    def construct_network(self):
        self.model = BACKBONES.build(self.model_config, logger=self.logger)
        self.logger.info('all parameters:{}'.format(count_params(self.model)))
        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model).eval()
            for param in self.model_ema.parameters():
                param.requires_grad = False
        if self.loss_config:
            self.loss = LOSSES.build(self.loss_config, logger=self.logger)
        if self.tokenizer_config is not None:
            self.tokenizer = TOKENIZERS.build(self.tokenizer_config,
                                              logger=self.logger)

        if self.first_stage_config:
            self.first_stage_model = MODELS.build(self.first_stage_config,
                                                  logger=self.logger)
            self.first_stage_model = self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
        else:
            self.first_stage_model = None
        if self.tokenizer_config is not None:
            self.cond_stage_config.KWARGS = {
                'vocab_size': self.tokenizer.vocab_size
            }
        if self.cond_stage_config == '__is_unconditional__':
            print(
                f'Training {self.__class__.__name__} as an unconditional model.'
            )
            self.cond_stage_model = None
        else:
            model = EMBEDDERS.build(self.cond_stage_config, logger=self.logger)
            self.cond_stage_model = model.eval().requires_grad_(False)
            self.cond_stage_model.train = disabled_train

    def forward_train(self,
                      image=None,
                      noise=None,
                      prompt=None,
                      label=None,
                      **kwargs):
        n, c, h, w = image.shape
        x_start = self.encode_first_stage(image, **kwargs)
        t = torch.randint(0, self.num_timesteps, (n, ),
                          device=x_start.device).long()
        ar = torch.tensor([[h / w]], device=we.device_id).repeat(n, 1)
        hw = torch.tensor([[h, w]], dtype=torch.float,
                          device=we.device_id).repeat(n, 1)
        context = {}
        cont_mask = None
        if prompt and self.cond_stage_model:
            with torch.autocast(device_type='cuda',
                                enabled=True,
                                dtype=torch.bfloat16):
                cont, cont_mask = getattr(self.cond_stage_model,
                                          'encode')(prompt, return_mask=True)
            context['crossattn'] = cont.float()
        else:
            assert label is not None
            context['label'] = label

        if 'hint' in kwargs and kwargs['hint'] is not None:
            hint = kwargs.pop('hint')
            if isinstance(context, dict):
                context['hint'] = hint
            else:
                context = {'crossattn': context, 'hint': hint}
        else:
            hint = None
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
                                       'mask': cont_mask,
                                       'data_info': {
                                           'img_hw': hw,
                                           'aspect_ratio': ar
                                       }
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
                     label=None,
                     sampler='ddim',
                     sample_steps=20,
                     seed=2023,
                     guide_scale=4.5,
                     guide_rescale=0.5,
                     discretization='trailing',
                     **kwargs):
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        num_samples = label.shape[0] if label is not None else len(prompt)
        context = {}
        null_context = {}
        cont_mask = None
        if prompt and self.cond_stage_model:
            with torch.autocast(device_type='cuda',
                                enabled=True,
                                dtype=torch.bfloat16):
                cont, cont_mask = getattr(self.cond_stage_model,
                                          'encode')(prompt, return_mask=True)
            context['crossattn'] = cont.float()
            null_context['crossattn'] = self.model.y_embedder.y_embedding[
                None].repeat(len(prompt), 1, 1)
        else:
            assert label is not None
            context['label'] = label
            null_context['label'] = torch.tensor(
                [self.model.num_classes]).repeat(num_samples).to(we.device_id)

        if 'hint' in kwargs and kwargs['hint'] is not None:
            hint = kwargs.pop('hint')
            if isinstance(context, dict):
                context['hint'] = hint
            if isinstance(null_context, dict):
                null_context['hint'] = hint
        else:
            hint = None
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
        noise = self.noise_sample(num_samples, height // self.size_factor,
                                  width // self.size_factor, g)
        # UNet use input n_prompt
        samples = self.diffusion.sample(
            solver=sampler,
            noise=noise,
            model=self.model,
            model_kwargs=[{
                'cond': context,
                'mask': cont_mask,
                'data_info': {
                    'img_hw':
                    torch.tensor([image_size],
                                 dtype=torch.float,
                                 device=we.device_id).repeat(num_samples, 1),
                    'aspect_ratio':
                    torch.tensor([[1.]], device=we.device_id).repeat(
                        num_samples, 1)
                }
            }, {
                'cond': null_context,
                'mask': cont_mask,
                'data_info': {
                    'img_hw':
                    torch.tensor([image_size],
                                 dtype=torch.float,
                                 device=we.device_id).repeat(num_samples, 1),
                    'aspect_ratio':
                    torch.tensor([[1.]], device=we.device_id).repeat(
                        num_samples, 1)
                }
            }] if guide_scale is not None and guide_scale > 0 else {
                'cond': context,
                'mask': cont_mask,
                'data_info': {
                    'img_hw':
                    torch.tensor([image_size],
                                 dtype=torch.float,
                                 device=we.device_id).repeat(num_samples, 1),
                    'aspect_ratio':
                    torch.tensor([[1.]], device=we.device_id).repeat(
                        num_samples, 1)
                }
            },
            cat_uc=False,
            steps=sample_steps,
            guide_scale=guide_scale,
            guide_rescale=guide_rescale,
            discretization=discretization,
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
        x_samples = torch.clamp(
            (x_samples + 1.0) / 2.0 + self.decoder_bias / 255,
            min=0.0,
            max=1.0)
        outputs = list()
        prompt = label.detach().cpu().numpy().tolist(
        ) if prompt is None else prompt
        for i, (p, img) in enumerate(zip(prompt, x_samples)):
            one_tup = {'prompt': str(p), 'n_prompt': '', 'image': img}
            if hint is not None:
                one_tup.update({'hint': hint[i]})
            outputs.append(one_tup)

        return outputs

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            LatentDiffusionPixart.para_dict,
                            set_name=True)
