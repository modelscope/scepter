# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# This file contains code that is adapted from
# diffusers: https://github.com/huggingface/diffusers
# ComfyUI: https://github.com/comfyanonymous/ComfyUI

import logging
import math
import re
from collections import OrderedDict
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.registry import BACKBONES
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

BROKEN_XFORMERS = False
try:
    x_vers = xformers.__version__
    # XFormers bug confirmed on all versions from 0.0.21 to 0.0.26 (q with bs bigger than 65535 gives CUDA error)
    BROKEN_XFORMERS = x_vers.startswith(
        '0.0.2') and not x_vers.startswith('0.0.20')
except:
    pass


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None):
    b, _, dim_head = q.shape
    dim_head //= heads

    disabled_xformers = False

    if BROKEN_XFORMERS:
        if b * heads > 65535:
            disabled_xformers = True

    if not disabled_xformers:
        if torch.jit.is_tracing() or torch.jit.is_scripting():
            disabled_xformers = True

    if disabled_xformers:
        return attention_pytorch(q, k, v, heads, mask)

    q, k, v = map(
        lambda t: t.reshape(b, -1, heads, dim_head),
        (q, k, v),
    )

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty([q.shape[0], q.shape[1], q.shape[1] + pad],
                               dtype=q.dtype,
                               device=q.device)
        mask_out[:, :, :mask.shape[-1]] = mask
        mask = mask_out[:, :, :mask.shape[-1]]

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    out = (out.reshape(b, -1, heads * dim_head))
    return out


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    out = torch.nn.functional.scaled_dot_product_attention(q,
                                                           k,
                                                           v,
                                                           attn_mask=mask,
                                                           dropout_p=0.0,
                                                           is_causal=False)
    out = (out.transpose(1, 2).reshape(b, -1, heads * dim_head))
    return out


def default(x, y):
    if x is not None:
        return x
    return y


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.,
        use_conv=False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = partial(
            operations.Conv2d,
            kernel_size=1) if use_conv else operations.Linear

        self.fc1 = linear_layer(in_features,
                                hidden_features,
                                bias=bias,
                                dtype=dtype,
                                device=device)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features,
                                out_features,
                                bias=bias,
                                dtype=dtype,
                                device=device)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = operations.Conv2d(in_chans,
                                      embed_dim,
                                      kernel_size=patch_size,
                                      stride=patch_size,
                                      bias=bias,
                                      dtype=dtype,
                                      device=device)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # if self.img_size is not None:
        #     if self.strict_img_size:
        #         _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
        #         _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
        #     elif not self.dynamic_img_pad:
        #         _assert(
        #             H % self.patch_size[0] == 0,
        #             f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
        #         )
        #         _assert(
        #             W % self.patch_size[1] == 0,
        #             f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        #         )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] -
                     H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] -
                     W % self.patch_size[1]) % self.patch_size[1]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h),
                                        mode='reflect')
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    scaling_factor=None,
    offset=None,
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim,
                                            pos,
                                            device=None,
                                            dtype=torch.float32):
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_torch(embed_dim,
                                  w,
                                  h,
                                  val_center=7.5,
                                  val_magnitude=7.5,
                                  device=None,
                                  dtype=torch.float32):
    small = min(h, w)
    val_h = (h / small) * val_magnitude
    val_w = (w / small) * val_magnitude
    grid_h, grid_w = torch.meshgrid(torch.linspace(-val_h + val_center,
                                                   val_h + val_center,
                                                   h,
                                                   device=device,
                                                   dtype=dtype),
                                    torch.linspace(-val_w + val_center,
                                                   val_w + val_center,
                                                   w,
                                                   device=device,
                                                   dtype=dtype),
                                    indexing='ij')
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2,
                                                    grid_h,
                                                    device=device,
                                                    dtype=dtype)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2,
                                                    grid_w,
                                                    device=device,
                                                    dtype=dtype)
    emb = torch.cat([emb_w, emb_h], dim=1)  # (H*W, D)
    return emb


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self,
                 hidden_size,
                 frequency_embedding_size=256,
                 dtype=None,
                 device=None,
                 operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size,
                              hidden_size,
                              bias=True,
                              dtype=dtype,
                              device=device),
            nn.SiLU(),
            operations.Linear(hidden_size,
                              hidden_size,
                              bias=True,
                              dtype=dtype,
                              device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class VectorEmbedder(nn.Module):
    """
    Embeds a flat vector of dimension input_dim
    """
    def __init__(self,
                 input_dim: int,
                 hidden_size: int,
                 dtype=None,
                 device=None,
                 operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(input_dim,
                              hidden_size,
                              bias=True,
                              dtype=dtype,
                              device=device),
            nn.SiLU(),
            operations.Linear(hidden_size,
                              hidden_size,
                              bias=True,
                              dtype=dtype,
                              device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.mlp(x)
        return emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1,
                      head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


def optimized_attention(qkv, num_heads):
    if XFORMERS_IS_AVAILBLE:
        optimized_attention_ops = attention_xformers
    else:
        optimized_attention_ops = attention_pytorch
    return optimized_attention_ops(qkv[0], qkv[1], qkv[2], num_heads)


class SelfAttention(nn.Module):
    ATTENTION_MODES = ('xformers', 'torch', 'torch-hb', 'math', 'debug')

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
        attn_mode: str = 'xformers',
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        rmsnorm: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = operations.Linear(dim,
                                     dim * 3,
                                     bias=qkv_bias,
                                     dtype=dtype,
                                     device=device)
        if not pre_only:
            self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
            self.proj_drop = nn.Dropout(proj_drop)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only

        if qk_norm == 'rms':
            self.ln_q = RMSNorm(self.head_dim,
                                elementwise_affine=True,
                                eps=1.0e-6,
                                dtype=dtype,
                                device=device)
            self.ln_k = RMSNorm(self.head_dim,
                                elementwise_affine=True,
                                eps=1.0e-6,
                                dtype=dtype,
                                device=device)
        elif qk_norm == 'ln':
            self.ln_q = operations.LayerNorm(self.head_dim,
                                             elementwise_affine=True,
                                             eps=1.0e-6,
                                             dtype=dtype,
                                             device=device)
            self.ln_k = operations.LayerNorm(self.head_dim,
                                             elementwise_affine=True,
                                             eps=1.0e-6,
                                             dtype=dtype,
                                             device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.pre_attention(x)
        x = optimized_attention(qkv, num_heads=self.num_heads)
        x = self.post_attention(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 elementwise_affine: bool = False,
                 eps: float = 1e-6,
                 device=None,
                 dtype=None):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(
                torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DismantledBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    ATTENTION_MODES = ('xformers', 'torch', 'torch-hb', 'math', 'debug')

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = 'xformers',
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        operations=None,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = operations.LayerNorm(hidden_size,
                                              elementwise_affine=False,
                                              eps=1e-6,
                                              dtype=dtype,
                                              device=device)
        else:
            self.norm1 = RMSNorm(hidden_size,
                                 elementwise_affine=False,
                                 eps=1e-6)
        self.attn = SelfAttention(dim=hidden_size,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  attn_mode=attn_mode,
                                  pre_only=pre_only,
                                  qk_norm=qk_norm,
                                  rmsnorm=rmsnorm,
                                  dtype=dtype,
                                  device=device,
                                  operations=operations)
        if not pre_only:
            if not rmsnorm:
                self.norm2 = operations.LayerNorm(hidden_size,
                                                  elementwise_affine=False,
                                                  eps=1e-6,
                                                  dtype=dtype,
                                                  device=device)
            else:
                self.norm2 = RMSNorm(hidden_size,
                                     elementwise_affine=False,
                                     eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(in_features=hidden_size,
                               hidden_features=mlp_hidden_dim,
                               act_layer=lambda: nn.GELU(approximate='tanh'),
                               drop=0,
                               dtype=dtype,
                               device=device,
                               operations=operations)
            else:
                self.mlp = SwiGLUFeedForward(
                    dim=hidden_size,
                    hidden_dim=mlp_hidden_dim,
                    multiple_of=256,
                )
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size,
                              n_mods * hidden_size,
                              bias=True,
                              dtype=dtype,
                              device=device))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if not self.pre_only:
            if not self.scale_mod_only:
                (
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                (
                    scale_msa,
                    gate_msa,
                    scale_mlp,
                    gate_mlp,
                ) = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(
                modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (
                x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )
        else:
            if not self.scale_mod_only:
                (
                    shift_msa,
                    scale_msa,
                ) = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(
                modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp,
                       gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        qkv, intermediates = self.pre_attention(x, c)
        attn = optimized_attention(
            qkv,
            num_heads=self.attn.num_heads,
        )
        return self.post_attention(attn, *intermediates)


def block_mixing(*args, use_checkpoint=True, **kwargs):
    if use_checkpoint:
        return torch.utils.checkpoint.checkpoint(_block_mixing,
                                                 *args,
                                                 use_reentrant=False,
                                                 **kwargs)
    else:
        return _block_mixing(*args, **kwargs)


def _block_mixing(context, x, context_block, x_block, c):
    context_qkv, context_intermediates = context_block.pre_attention(
        context, c)

    x_qkv, x_intermediates = x_block.pre_attention(x, c)

    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    qkv = tuple(o)

    attn = optimized_attention(
        qkv,
        num_heads=x_block.attn.num_heads,
    )
    context_attn, x_attn = (
        attn[:, :context_qkv[0].shape[1]],
        attn[:, context_qkv[0].shape[1]:],
    )

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn,
                                               *context_intermediates)

    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        pre_only = kwargs.pop('pre_only')
        qk_norm = kwargs.pop('qk_norm', None)
        self.context_block = DismantledBlock(*args,
                                             pre_only=pre_only,
                                             qk_norm=qk_norm,
                                             **kwargs)
        self.x_block = DismantledBlock(*args,
                                       pre_only=False,
                                       qk_norm=qk_norm,
                                       **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args,
                            context_block=self.context_block,
                            x_block=self.x_block,
                            **kwargs)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        total_out_channels: Optional[int] = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size,
                                               elementwise_affine=False,
                                               eps=1e-6,
                                               dtype=dtype,
                                               device=device)
        self.linear = (operations.Linear(hidden_size,
                                         patch_size * patch_size *
                                         out_channels,
                                         bias=True,
                                         dtype=dtype,
                                         device=device) if
                       (total_out_channels is None) else operations.Linear(
                           hidden_size,
                           total_out_channels,
                           bias=True,
                           dtype=dtype,
                           device=device))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size,
                              2 * hidden_size,
                              bias=True,
                              dtype=dtype,
                              device=device))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SelfAttentionContext(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dtype=None,
                 device=None,
                 operations=None):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim

        self.heads = heads
        self.dim_head = dim_head

        self.qkv = operations.Linear(dim,
                                     dim * 3,
                                     bias=True,
                                     dtype=dtype,
                                     device=device)

        self.proj = operations.Linear(inner_dim,
                                      dim,
                                      dtype=dtype,
                                      device=device)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.dim_head)
        x = optimized_attention((q.reshape(q.shape[0], q.shape[1], -1), k, v),
                                self.heads)
        return self.proj(x)


class ContextProcessorBlock(nn.Module):
    def __init__(self, context_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(context_size,
                                          elementwise_affine=False,
                                          eps=1e-6,
                                          dtype=dtype,
                                          device=device)
        self.attn = SelfAttentionContext(context_size,
                                         dtype=dtype,
                                         device=device,
                                         operations=operations)
        self.norm2 = operations.LayerNorm(context_size,
                                          elementwise_affine=False,
                                          eps=1e-6,
                                          dtype=dtype,
                                          device=device)
        self.mlp = Mlp(in_features=context_size,
                       hidden_features=(context_size * 4),
                       act_layer=lambda: nn.GELU(approximate='tanh'),
                       drop=0,
                       dtype=dtype,
                       device=device,
                       operations=operations)

    def forward(self, x):
        x += self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


class ContextProcessor(nn.Module):
    def __init__(self,
                 context_size,
                 num_layers,
                 dtype=None,
                 device=None,
                 operations=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            ContextProcessorBlock(context_size,
                                  dtype=dtype,
                                  device=device,
                                  operations=operations)
            for i in range(num_layers)
        ])
        self.norm = operations.LayerNorm(context_size,
                                         elementwise_affine=False,
                                         eps=1e-6,
                                         dtype=dtype,
                                         device=device)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
        return self.norm(x)


@BACKBONES.register_class()
class MMDiT(BaseModel):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.ignore_keys = cfg.get('IGNORE_KEYS', None)
        self.input_size = cfg.get('INPUT_SIZE', 32)
        self.patch_size = cfg.get('PATCH_SIZE', 2)
        self.in_channels = cfg.get('IN_CHANNELS', 4)
        self.depth = cfg.get('DEPTH', 28)
        self.mlp_ratio = cfg.get('MLP_RATIO', 4.0)
        self.learn_sigma = cfg.get('LEARN_SIGMA', False)
        self.adm_in_channels = cfg.get('ADM_IN_CHANNELS', None)
        self.context_embedder_config = cfg.get('CONTEXT_EMBEDDER_CONFIG', None)
        self.compile_core = cfg.get('COMPILE_CORE', False)
        self.use_checkpoint = cfg.get('USE_CHECKPOINT', False)
        self.register_length = cfg.get('REGISTER_LENGTH', 0)
        self.attn_mode = cfg.get('ATTN_MODE', 'torch')
        self.rmsnorm = cfg.get('RMSNORM', False)
        self.scale_mod_only = cfg.get('SCALE_MOD_ONLY', False)
        self.swiglu = cfg.get('SWIGLU', False)
        self.out_channels = cfg.get('OUT_CHANNELS', None)
        self.pos_embed_scaling_factor = cfg.get('POS_EMBED_SCALING_FACTOR',
                                                None)
        self.pos_embed_offset = cfg.get('POS_EMBED_OFFSET', None)
        self.pos_embed_max_size = cfg.get('POS_EMBED_MAX_SIZE', None)
        self.num_patches = cfg.get('NUM_PATCHES', None)
        self.qk_norm = cfg.get('QK_NORM', None)
        self.qkv_bias = cfg.get('QKV_BIAS', True)
        self.context_processor_layers = cfg.get('CONTEXT_PROCESSOR_LAYERS',
                                                None)
        self.context_size = cfg.get('CONTEXT_SIZE', 4096)
        self.dtype = cfg.get('DTYPE', None)
        self.device = cfg.get('DEVICE', None)
        self.operations = cfg.get('OPERATIONS', nn)

        default_out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.out_channels = default(self.out_channels, default_out_channels)

        # hidden_size = default(hidden_size, 64 * depth)
        # num_heads = default(num_heads, hidden_size // 64)

        # apply magic --> this defines a head_size of 64
        self.hidden_size = 64 * self.depth
        num_heads = self.depth

        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=self.pos_embed_max_size is None,
            dtype=self.dtype,
            device=self.device,
            operations=self.operations)
        self.t_embedder = TimestepEmbedder(self.hidden_size,
                                           dtype=self.dtype,
                                           device=self.device,
                                           operations=self.operations)

        self.y_embedder = None
        if self.adm_in_channels is not None:
            assert isinstance(self.adm_in_channels, int)
            self.y_embedder = VectorEmbedder(self.adm_in_channels,
                                             self.hidden_size,
                                             dtype=self.dtype,
                                             device=self.device,
                                             operations=self.operations)

        if self.context_processor_layers is not None:
            self.context_processor = ContextProcessor(
                self.context_size,
                self.context_processor_layers,
                dtype=self.dtype,
                device=self.device,
                operations=self.operations)
        else:
            self.context_processor = None

        self.context_embedder = nn.Identity()
        if self.context_embedder_config is not None:
            self.context_embedder_config = Config.get_dict(
                self.context_embedder_config)
            if self.context_embedder_config['target'] == 'torch.nn.Linear':
                self.context_embedder = self.operations.Linear(
                    **self.context_embedder_config['params'],
                    dtype=self.dtype,
                    device=self.device)

        self.register_length = self.register_length
        if self.register_length > 0:
            self.register = nn.Parameter(
                torch.randn(1,
                            self.register_length,
                            self.hidden_size,
                            dtype=self.dtype,
                            device=self.device))

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # just use a buffer already
        if self.num_patches is not None:
            self.register_buffer(
                'pos_embed',
                torch.empty(1,
                            self.num_patches,
                            self.hidden_size,
                            dtype=self.dtype,
                            device=self.device),
            )
        else:
            self.pos_embed = None

        self.use_checkpoint = self.use_checkpoint
        self.joint_blocks = nn.ModuleList([
            JointBlock(self.hidden_size,
                       num_heads,
                       mlp_ratio=self.mlp_ratio,
                       qkv_bias=self.qkv_bias,
                       attn_mode=self.attn_mode,
                       pre_only=i == self.depth - 1,
                       rmsnorm=self.rmsnorm,
                       scale_mod_only=self.scale_mod_only,
                       swiglu=self.swiglu,
                       qk_norm=self.qk_norm,
                       dtype=self.dtype,
                       device=self.device,
                       operations=self.operations) for i in range(self.depth)
        ])

        self.final_layer = FinalLayer(self.hidden_size,
                                      self.patch_size,
                                      self.out_channels,
                                      dtype=self.dtype,
                                      device=self.device,
                                      operations=self.operations)

        if self.compile_core:
            assert False
            self.forward_core_with_concat = torch.compile(
                self.forward_core_with_concat)

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                if local_path.endswith('safetensors'):
                    from safetensors.torch import load_file as load_safetensors
                    model = load_safetensors(local_path)
                else:
                    model = torch.load(local_path, map_location='cpu')
                if 'state_dict' in model:
                    model = model['state_dict']
                new_ckpt = OrderedDict()
                ignore_ckpt = OrderedDict()
                for k, v in model.items():
                    if self.ignore_keys is not None:
                        if (isinstance(self.ignore_keys, str) and re.match(self.ignore_keys, k)) or \
                            (isinstance(self.ignore_keys, list) and k in self.ignore_keys):
                            ignore_ckpt[k] = v
                            continue
                    k = k.replace('model.diffusion_model.', '')
                    new_ckpt[k] = v
                missing, unexpected = self.load_state_dict(new_ckpt,
                                                           strict=False)
                print(
                    f'Restored from {pretrained_model} with {len(missing)} missing and {len(unexpected)} unexpected keys'
                )
                if len(missing) > 0:
                    print(f'Missing Keys:\n {missing}')
                if len(unexpected) > 0:
                    print(f'\nUnexpected Keys:\n {unexpected}')

    def cropped_pos_embed(self, hw, device=None):
        p = self.x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = (h + 1) // p
        w = (w + 1) // p
        if self.pos_embed is None:
            return get_2d_sincos_pos_embed_torch(self.hidden_size,
                                                 w,
                                                 h,
                                                 device=device)
        assert self.pos_embed_max_size is not None
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            '1 (h w) c -> 1 h w c',
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top:top + h, left:left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed,
                                      '1 h w c -> 1 (h w) c')
        # print(spatial_pos_embed, top, left, h, w)
        # # t = get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, 7.875, 7.875, device=device) #matches exactly for 1024 res
        # t = get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, 7.5, 7.5, device=device) #scales better
        # # print(t)
        # return t
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1]**0.5)
        else:
            h, w = hw
            h = (h + 1) // p
            w = (w + 1) // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(
        self,
        x: torch.Tensor,
        c_mod: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.register_length > 0:
            context = torch.cat(
                (
                    repeat(self.register, '1 ... -> b ...', b=x.shape[0]),
                    default(context,
                            torch.Tensor([]).type_as(x)),
                ),
                1,
            )

        # context is B, L', D
        # x is B, L, D
        for block in self.joint_blocks:
            context, x = block(
                context,
                x,
                c=c_mod,
                use_checkpoint=self.use_checkpoint,
            )

        x = self.final_layer(x,
                             c_mod)  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward(self,
                x,
                t=None,
                cond=dict(),
                mask=None,
                data_info=None,
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if isinstance(cond, dict):
            if 'label' in cond and cond['label'] is not None:
                label = cond['label']
            if 'concat' in cond:
                concat = cond['concat']
                x = torch.cat([x, concat], dim=1)
            context = cond.get('crossattn', None)
            y = cond.get('y', None)
        else:
            context = cond
            y = None

        if self.context_processor is not None:
            context = self.context_processor(context)

        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(
            hw, device=x.device).to(dtype=x.dtype, device=x.device)
        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        if context is not None:
            context = self.context_embedder(context)

        x = self.forward_core_with_concat(x, c, context)

        x = self.unpatchify(x, hw=hw)  # (N, out_channels, H, W)
        return x[:, :, :hw[-2], :hw[-1]]


if __name__ == '__main__':
    config_dict = {
        'NMAE':
        'MMDiT',
        'IN_CHANNELS':
        16,
        'PATCH_SIZE':
        2,
        'OUT_CHANNELS':
        16,
        'DEPTH':
        24,
        'INPUT_SIZE':
        None,
        'ADM_IN_CHANNELS':
        2048,
        'CONTEXT_EMBEDDER_CONFIG': {
            'target': 'torch.nn.Linear',
            'params': {
                'in_features': 4096,
                'out_features': 1536
            }
        },
        'NUM_PATCHES':
        36864,
        'POS_EMBED_MAX_SIZE':
        192,
        'POS_EMBED_SCALING_FACTOR':
        None,
        'DTYPE':
        torch.float16,
        'IGNORE_KEYS':
        '^first_stage_model',
        'PRETRAINED_MODEL':
        '/mnt/data/huggingface_repo/stabilityai/stable-diffusion-3-medium/sd3_medium.safetensors'
    }
    config = Config(load=False, cfg_dict=config_dict)
    model = MMDiT(config)
    model.load_pretrained_model(config_dict['PRETRAINED_MODEL'])
    # print('=====')
