# -*- coding: utf-8 -*-

# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.registry import BACKBONES
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

from .layers import CogVideoXBlock, CogVideoXPatchEmbed, TimestepEmbedding, Timesteps, AdaLayerNorm


@BACKBONES.register_class()
class CogVideoXTransformer3DModel(BaseModel):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    def __init__(
        self,
        cfg,
        logger=None
    ):
        super().__init__(cfg, logger=logger)
        num_attention_heads = cfg.get("NUM_ATTENTION_HEADS", 30)
        attention_head_dim = cfg.get("ATTENTION_HEAD_DIM", 64)
        in_channels = cfg.get("IN_CHANNELS", 16)
        out_channels = cfg.get("OUT_CHANNELS", 16)
        flip_sin_to_cos = cfg.get("FLIP_SIN_TO_COS", True)
        freq_shift = cfg.get("FREQ_SHIFT", 0)
        time_embed_dim = cfg.get("TIME_EMBED_DIM", 512)
        ofs_embed_dim = cfg.get("OFS_EMBED_DIM", None)  # 1.5
        text_embed_dim = cfg.get("TEXT_EMBED_DIM", 4096)
        num_layers = cfg.get("NUM_LAYERS", 30)
        dropout = cfg.get("DROPOUT", 0.0)
        attention_bias = cfg.get("ATTENTION_BIAS", True)
        sample_width = cfg.get("SAMPLE_WIDTH", 90)
        sample_height = cfg.get("SAMPLE_HEIGHT", 60)
        sample_frames = cfg.get("SAMPLE_FRAMES", 49)
        patch_size = cfg.get("PATCH_SIZE", 2)
        patch_size_t = cfg.get("PATCH_SIZE_T", None)
        patch_bias = cfg.get("PATCH_BIAS", True)
        temporal_compression_ratio = cfg.get("TEMPORAL_COMPRESSION_RATIO", 4)
        max_text_seq_length = cfg.get("MAX_TEXT_SEQ_LENGTH", 226)
        activation_fn = cfg.get("ACTIVATION_FN", "gelu-approximate")
        timestep_activation_fn = cfg.get("TIMESTEP_ACTIVATION_FN", "silu")
        norm_elementwise_affine = cfg.get("NORM_ELEMENTWISE_AFFINE", True)
        norm_eps = cfg.get("NORM_EPS", 1e-5)
        spatial_interpolation_scale = cfg.get("SPATIAL_INTERPOLATION_SCALE", 1.875)
        temporal_interpolation_scale = cfg.get("TEMPORAL_INTERPOLATION_SCALE", 1.0)
        use_rotary_positional_embeddings = cfg.get("USE_ROTARY_POSITIONAL_EMBEDDINGS", False)
        use_learned_positional_embeddings = cfg.get("USE_LEARNED_POSITIONAL_EMBEDDINGS", False)
        self.gradient_checkpointing = cfg.get("GRADIENT_CHECKPOINTING", False)
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor = None,
        t: Union[int, float, torch.LongTensor] = None,
        cond: torch.Tensor = None,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        if 'image_latent' in kwargs and kwargs['image_latent'] is not None:
            hidden_states = torch.cat([x, kwargs['image_latent']], dim=2)
        else:
            hidden_states = x
        timestep = t
        encoder_hidden_states = cond

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=encoder_hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.patch_size
        p_t = self.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        return output

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model is not None:
            pretrained_model_list = [pretrained_model] if isinstance(pretrained_model, str) else pretrained_model
            ckpt_all = OrderedDict()
            for pretrained_model in pretrained_model_list:
                with FS.get_from(pretrained_model,
                                 wait_finish=True) as local_model:
                    if local_model.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        ckpt = load_safetensors(local_model)
                    else:
                        ckpt = torch.load(local_model, map_location='cpu', weights_only=True)
                    ckpt_all.update(ckpt)
            missing, unexpected = self.load_state_dict(ckpt_all, strict=False)
            if we.rank == 0:
                self.logger.info(
                    f'Restored from {pretrained_model_list} with {len(missing)} missing and {len(unexpected)} unexpected keys'
                )
                if len(missing) > 0:
                    self.logger.info(f'Missing Keys:\n {missing}')
                if len(unexpected) > 0:
                    self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            CogVideoXTransformer3DModel.para_dict,
                            set_name=True)


if __name__ == "__main__":
    import argparse
    from scepter.modules.utils.file_system import FS
    from scepter.modules.utils.config import Config
    from scepter.modules.utils.logger import get_logger

    parser = argparse.ArgumentParser()
    cfg = Config(parser_ins=parser)
    for file_sys in cfg.FILE_SYSTEM:
        FS.init_fs_client(file_sys)
    model = BACKBONES.build(cfg.DIFFUSION_MODEL, logger=get_logger()).eval().requires_grad_(False).to('cuda').to(torch.bfloat16)

    hidden_states = torch.load(FS.get_from(cfg.HIDDEN_STATES), weights_only=True)
    encoder_hidden_states = torch.load(FS.get_from(cfg.ENCODER_HIDDEN_STATES), weights_only=True)
    timestep = torch.load(FS.get_from(cfg.TIMESTEP), weights_only=True)
    timestep_cond = None
    image_rotary_emb = None
    attention_kwargs = None
    output = model(hidden_states, encoder_hidden_states, timestep, timestep_cond, image_rotary_emb, attention_kwargs)
    print(output, torch.sum(output))
