# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math
import numbers
import random
import torch
from scepter.modules.model.network.ldm import LatentDiffusion
from scepter.modules.model.registry import MODELS, BACKBONES, LOSSES, TOKENIZERS, EMBEDDERS, DIFFUSIONS
from scepter.modules.model.utils.basic_utils import disabled_train
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.model.utils.basic_utils import count_params


@MODELS.register_class()
class LatentDiffusionFlux(LatentDiffusion):
    para_dict = LatentDiffusion.para_dict

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.guide_scale = cfg.get('GUIDE_SCALE', 3.5)

    def init_params(self):
        self.parameterization = self.cfg.get('PARAMETERIZATION', 'rf')
        assert self.parameterization in [
            'eps', 'x0', 'v', 'rf'
        ], 'currently only supporting "eps" and "x0" and "v" and "rf"'

        diffusion_cfg = self.cfg.get("DIFFUSION", None)
        assert diffusion_cfg is not None
        if self.cfg.have("WORK_DIR"):
            diffusion_cfg.WORK_DIR = self.cfg.WORK_DIR
        self.diffusion = DIFFUSIONS.build(diffusion_cfg, logger=self.logger)

        self.pretrained_model = self.cfg.get('PRETRAINED_MODEL', None)
        self.ignore_keys = self.cfg.get('IGNORE_KEYS', [])

        self.model_config = self.cfg.DIFFUSION_MODEL
        self.first_stage_config = self.cfg.FIRST_STAGE_MODEL
        self.cond_stage_config = self.cfg.COND_STAGE_MODEL
        self.tokenizer_config = self.cfg.get('TOKENIZER', None)
        self.loss_config = self.cfg.get('LOSS', None)

        self.scale_factor = self.cfg.get('SCALE_FACTOR', 0.18215)
        self.size_factor = self.cfg.get('SIZE_FACTOR', 16)
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

    def construct_network(self):
        # embedding_context = torch.device("meta") if self.model_config.get("PRETRAINED_MODEL", None) else nullcontext()
        # with embedding_context:
        self.model = BACKBONES.build(self.model_config, logger=self.logger).to(torch.bfloat16)
        self.logger.info('all parameters:{}'.format(count_params(self.model)))
        if self.use_ema:
            if self.model_ema_config:
                self.model_ema = BACKBONES.build(self.model_ema_config,
                                                 logger=self.logger)
            else:
                self.model_ema = copy.deepcopy(self.model)
            self.model_ema = self.model_ema.eval()
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

    def noise_sample(self, num_samples, h, w, seed, dtype = torch.bfloat16):
        noise = torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(h / 16),
            2 * math.ceil(w / 16),
            device=we.device_id,
            dtype=dtype,
            generator=torch.Generator(device=we.device_id).manual_seed(seed),
        )
        return noise

    def forward_train(self, image=None, noise=None, prompt=None, **kwargs):
        x_start = self.encode_first_stage(image, **kwargs)
        if prompt and self.cond_stage_model:
            ctx = getattr(self.cond_stage_model, 'encode')(prompt)
        else:
            assert False
        if 'index' in kwargs:
            kwargs.pop('index')
        guide_scale = self.guide_scale
        if guide_scale is not None:
            guide_scale = torch.full((x_start.shape[0],), guide_scale, device=x_start.device, dtype=x_start.dtype)
        else:
            guide_scale = None
        loss = self.diffusion.loss(x_0=x_start,
                                   model=self.model,
                                   model_kwargs={"cond": ctx, "guidance": guide_scale},
                                   noise=noise,
                                   **kwargs)
        loss = loss.mean()
        ret = {'loss': loss, 'probe_data': {'prompt': prompt}}
        return ret


    @torch.no_grad()
    def forward_test(self,
                     image=None,
                     prompt=None,
                     sampler='flow_eluer',
                     sample_steps=20,
                     seed=2023,
                     guide_scale=4.5,
                     guide_rescale=0.0,
                     show_process=False,
                     **kwargs):
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        if isinstance(prompt, str):
            prompt = [prompt]
        assert isinstance(prompt, list)
        num_samples = len(prompt)
        if prompt and self.cond_stage_model:
            ctx = getattr(self.cond_stage_model, 'encode')(prompt)
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
        noise = self.noise_sample(
                                  num_samples,
                                  height,
                                  width,
                                  seed
                                  )
        guide_scale = guide_scale or self.guide_scale
        if guide_scale is not None:
            guide_scale = torch.full((noise.shape[0],), guide_scale, device=noise.device, dtype=noise.dtype)
        else:
            guide_scale = None
        # UNet use input n_prompt
        samples = self.diffusion.sample(
            noise=noise,
            sampler=sampler,
            model=self.model,
            model_kwargs= {"cond": ctx, "guidance": guide_scale},
            steps=sample_steps,
            show_progress=True,
            guide_scale = guide_scale,
            return_intermediate=None,
            **kwargs).float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                            LatentDiffusionFlux.para_dict,
                            set_name=True)

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        z = self.first_stage_model.encode(x)
        if isinstance(z, (tuple, list)):
            z = z[0]
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)
