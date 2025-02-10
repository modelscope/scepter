# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import random
import torch
from typing import Tuple

from scepter.modules.model.network.ldm import LatentDiffusion
from scepter.modules.model.registry import MODELS
from scepter.modules.model.utils.basic_utils import default
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.model.backbone.cogvideox.utils import get_3d_rotary_pos_embed, get_resize_crop_region_for_grid


@MODELS.register_class()
class LatentDiffusionCogVideoX(LatentDiffusion):
    para_dict = LatentDiffusion.para_dict

    def init_params(self):
        super().init_params()
        self.latent_channels = self.model_config.get('LATENT_CHANNELS', self.model_config.IN_CHANNELS)
        self.scale_factor_spatial = self.cfg.get('SCALE_FACTOR_SPATIAL', 8)
        self.scale_factor_temporal = self.cfg.get('SCALE_FACTOR_TEMPORAL', 4)
        self.scaling_factor_image = self.cfg.get('SCALING_FACTOR_IMAGE', 0.7)
        self.use_rotary_positional_embeddings = self.model_config.get('USE_ROTARY_POSITIONAL_EMBEDDINGS', False)
        self.attention_head_dim = self.model_config.get('ATTENTION_HEAD_DIM', 64)
        self.patch_size = self.model_config.get('PATCH_SIZE', 2)
        self.patch_size_t = self.model_config.get('PATCH_SIZE_T', None)
        self.ofs_embed_dim = self.model_config.get('OFS_EMBED_DIM', None)
        self.sample_height = self.first_stage_config.get('SAMPLE_HEIGHT', 60)
        self.sample_width = self.first_stage_config.get('SAMPLE_WIDTH', 90)
        self.noised_image_dropout = self.cfg.get('NOISED_IMAGE_DROPOUT', 0.05)
        self.invert_scale_latents = self.cfg.get('INVERT_SCALE_LATENTS', False)

    def construct_network(self):
        super().construct_network()
        self.model = self.model.to(getattr(torch, self.model_config.DTYPE))
        self.first_stage_model = self.first_stage_model.to(getattr(torch, self.first_stage_config.DTYPE))

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        if isinstance(x, list):
            x = torch.stack(x, dim=0)  # [B, C, F, H, W]
        image_latents = self.first_stage_model.encode(x).sample()
        if not self.invert_scale_latents:
            latents = self.scaling_factor_image * image_latents
        else:
            latents = 1 / self.scaling_factor_image * image_latents
        return latents

    @torch.no_grad()
    def decode_first_stage(self, latents):
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.scaling_factor_image * latents
        frames = self.first_stage_model.decode(latents)
        return frames

    def get_image_latent(self, image, video, noise):
        latent = torch.zeros_like(noise)
        if isinstance(image, list):
            image = torch.stack(image, dim=0)  # [B, C, F, H, W]
        if len(image.shape) == 4:  # [B, C, H, W]
            image = image.unsqueeze(2)  # [B, C, F, H, W]
        image_latent = self.encode_first_stage(image)  # [B, C, F, H, W]
        image_latent = image_latent.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        latent[:, :1, :, :, :] = image_latent
        return latent, image

    def noise_sample(self, batch_size, num_frames, height, width, generator, dtype=torch.bfloat16):
        shape = (batch_size,
                (num_frames - 1) // self.scale_factor_temporal + 1,
                self.latent_channels,
                height // self.scale_factor_spatial,
                width // self.scale_factor_spatial
                 )
        noise = torch.randn(shape, generator=generator, dtype=dtype, device='cpu').to(we.device_id)
        return noise

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.scale_factor_spatial * self.patch_size)
        grid_width = width // (self.scale_factor_spatial * self.patch_size)

        p = self.patch_size
        p_t = self.patch_size_t

        base_size_width = self.sample_width // p
        base_size_height = self.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
            )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin


    def forward_train(self, video=None, video_latent=None, image=None, noise=None, prompt=None, image_size=None, **kwargs):
        # video: [B, C, F, H, W]
        if image_size is None: image_size = [480, 720]
        if video_latent is not None:
            x_start = torch.stack(video_latent)
        else:
            x_start = self.encode_first_stage(video, **kwargs)
        x_start = x_start.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        t = torch.randint(low=0, high=self.num_timesteps, size=(len(video),), device=we.device_id)

        if prompt and self.cond_stage_model:
            with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                cont = getattr(self.cond_stage_model, 'encode')(prompt, return_mask=False, use_mask=False)

        if noise is None:
            noise = torch.randn_like(x_start)

        if image is not None:
            if random.random() < self.noised_image_dropout:
                image_latent = torch.zeros_like(noise)
            else:
                image_latent, _ = self.get_image_latent(image, video, noise)
        else:
            image_latent = None

        height, width = image_size[0] if isinstance(image_size, list) and all(isinstance(elem, list) for elem in image_size) else image_size
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height=height, width=width, num_frames=noise.size(1), device=we.device_id)
            if self.use_rotary_positional_embeddings
            else None
        )
        ofs_emb = None if self.ofs_embed_dim is None else image_latent.new_full((1,), fill_value=2.0)

        loss = self.diffusion.loss(x_0=x_start,
                                   t=t,
                                   model=self.model,
                                   model_kwargs={"cond": cont,
                                                 'image_latent': image_latent,
                                                 'image_rotary_emb': image_rotary_emb,
                                                 'ofs': ofs_emb},
                                   noise=noise,
                                   **kwargs)
        loss = loss.mean()
        ret = {'loss': loss, 'probe_data': {'prompt': prompt}}
        return ret

    @torch.no_grad()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def forward_test(self,
                     video=None,
                     image=None,
                     prompt=None,
                     n_prompt=None,
                     sampler='ddim',
                     sample_steps=50,
                     seed=42,
                     guide_scale=6.0,
                     guide_rescale=0.0,
                     num_frames=49,
                     image_size=None,
                     show_process=False,
                     **kwargs):
        if image_size is None:
            image_size = [480, 720]
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator().manual_seed(seed)
        # generator = torch.Generator(we.device_id).manual_seed(seed)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        num_samples = len(prompt)
        n_prompt = default(n_prompt, [self.default_n_prompt] * len(prompt))

        if prompt and self.cond_stage_model:
            with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                cont = getattr(self.cond_stage_model, 'encode')(prompt, return_mask=False, use_mask=False)
                null_cont = getattr(self.cond_stage_model, 'encode')(n_prompt, return_mask=False, use_mask=False)

        height, width = image_size[0] if isinstance(image_size, list) and all(isinstance(elem, list) for elem in image_size) else image_size
        latent_frames = (num_frames - 1) // self.scale_factor_temporal + 1
        additional_frames = 0
        if self.patch_size_t is not None and latent_frames % self.patch_size_t != 0:
            additional_frames = self.patch_size_t - latent_frames % self.patch_size_t
            num_frames += additional_frames * self.scale_factor_temporal
        noise = self.noise_sample(num_samples, num_frames, height, width, generator)
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, noise.size(1), we.device_id)
            if self.use_rotary_positional_embeddings
            else None
        )

        image_latent, image = self.get_image_latent(image, video, noise) if image is not None else (None, None)
        ofs_emb = None if self.ofs_embed_dim is None else image_latent.new_full((1,), fill_value=2.0)

        samples = self.diffusion.sample(noise=noise,
                                        sampler=sampler,
                                        model=self.model,
                                        model_kwargs=[{
                                            'cond': cont,
                                            'image_latent': image_latent,
                                            'image_rotary_emb': image_rotary_emb,
                                            'ofs': ofs_emb
                                        }, {
                                            'cond': null_cont,
                                            'image_latent': image_latent,
                                            'image_rotary_emb': image_rotary_emb,
                                            'ofs': ofs_emb
                                        }],
                                        steps=sample_steps,
                                        show_progress=True,
                                        guide_scale=guide_scale,
                                        guide_rescale=guide_rescale,
                                        return_intermediate=None,
                                        **kwargs).float()

        samples = samples[:, additional_frames:]
        x_frames = self.decode_first_stage(samples).float()

        outputs = []
        for batch_idx in range(num_samples):
            rec_video = torch.clamp(x_frames[batch_idx] / 2 + 0.5, min=0.0, max=1.0)
            one_tup = {
                'reconstruct_video': rec_video.squeeze(0).float(),
                'instruction': prompt[batch_idx]
            }
            if image is not None:
                ori_image = torch.clamp(image[batch_idx] / 2 + 0.5, min=0.0, max=1.0)
                one_tup['edit_image'] = ori_image
            if video is not None:
                ori_video = torch.clamp(video[batch_idx] / 2 + 0.5, min=0.0, max=1.0)
                one_tup['target_video'] = ori_video.squeeze(0)
            outputs.append(one_tup)
        return outputs


    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            LatentDiffusionCogVideoX.para_dict,
                            set_name=True)