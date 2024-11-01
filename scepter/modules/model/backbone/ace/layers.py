# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import warnings

import torch
import torch.nn as nn

from scepter.modules.model.backbone.transformer.attention import RMSNorm
from scepter.modules.model.backbone.transformer.layers import (DropPath, Mlp,
                                                               modulate)
from scepter.modules.model.backbone.transformer.pos_embed import \
    rope_apply_multires as rope_apply

try:
    from flash_attn import (flash_attn_varlen_func)
    FLASHATTN_IS_AVAILABLE = True
except ImportError as e:
    FLASHATTN_IS_AVAILABLE = False
    flash_attn_varlen_func = None
    warnings.warn(f'{e}')


class ACEBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4.0,
                 drop_path=0.,
                 window_size=0,
                 backend=None,
                 use_condition=True,
                 qk_norm=False,
                 **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_condition = use_condition
        self.norm1 = nn.LayerNorm(hidden_size,
                                  elementwise_affine=False,
                                  eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size,
                                       num_heads=num_heads,
                                       qkv_bias=True,
                                       backend=backend,
                                       qk_norm=qk_norm,
                                       **block_kwargs)
        if self.use_condition:
            self.cross_attn = MultiHeadAttention(hidden_size,
                                                 context_dim=hidden_size,
                                                 num_heads=num_heads,
                                                 qkv_bias=True,
                                                 backend=backend,
                                                 qk_norm=qk_norm,
                                                 **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size,
                                  elementwise_affine=False,
                                  eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu,
                       drop=0)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, **kwargs):
        B = x.size(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(1), scale_msa.squeeze(1), gate_msa.squeeze(1),
            shift_mlp.squeeze(1), scale_mlp.squeeze(1), gate_mlp.squeeze(1))
        x = x + self.drop_path(gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa, unsqueeze=False), **
            kwargs))
        if self.use_condition:
            x = x + self.cross_attn(x, context=y, **kwargs)

        x = x + self.drop_path(gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp, unsqueeze=False)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 attn_drop=0.0,
                 qkv_bias=False,
                 dropout=0.0,
                 backend=None,
                 qk_norm=False,
                 eps=1e-6,
                 **block_kwargs):
        super().__init__()
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        context_dim = context_dim or dim
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)
        # layers
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.attention_op = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.backend = backend
        assert self.backend in ('flash_attn', 'xformer_attn', 'pytorch_attn',
                                None)
        if FLASHATTN_IS_AVAILABLE and self.backend in ('flash_attn', None):
            self.backend = 'flash_attn'
            self.softmax_scale = block_kwargs.get('softmax_scale', None)
            self.causal = block_kwargs.get('causal', False)
            self.window_size = block_kwargs.get('window_size', (-1, -1))
            self.deterministic = block_kwargs.get('deterministic', False)
        else:
            raise NotImplementedError

    def flash_attn(self, x, context=None, **kwargs):
        '''
         The implementation will be very slow when mask is not None,
         because we need rearange the x/context features according to mask.
        Args:
            x:
            context:
            mask:
            **kwargs:
        Returns: x
        '''
        dtype = kwargs.get('dtype', torch.float16)

        def half(x):
            return x if x.dtype in [torch.float16, torch.bfloat16
                                    ] else x.to(dtype)

        x_shapes = kwargs['x_shapes']
        freqs = kwargs['freqs']
        self_x_len = kwargs['self_x_len']
        cross_x_len = kwargs['cross_x_len']
        txt_lens = kwargs['txt_lens']
        n, d = self.num_heads, self.head_dim

        if context is None:
            # self-attn
            q = self.norm_q(self.q(x)).view(-1, n, d)
            k = self.norm_q(self.k(x)).view(-1, n, d)
            v = self.v(x).view(-1, n, d)
            q = rope_apply(q, self_x_len, x_shapes, freqs, pad=False)
            k = rope_apply(k, self_x_len, x_shapes, freqs, pad=False)
            q_lens = k_lens = self_x_len
        else:
            # cross-attn
            q = self.norm_q(self.q(x)).view(-1, n, d)
            k = self.norm_q(self.k(context)).view(-1, n, d)
            v = self.v(context).view(-1, n, d)
            q_lens = cross_x_len
            k_lens = txt_lens

        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]),
                                  q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]),
                                  k_lens]).cumsum(0, dtype=torch.int32)
        max_seqlen_q = q_lens.max()
        max_seqlen_k = k_lens.max()

        out_dtype = q.dtype
        q, k, v = half(q), half(k), half(v)
        x = flash_attn_varlen_func(q,
                                   k,
                                   v,
                                   cu_seqlens_q=cu_seqlens_q,
                                   cu_seqlens_k=cu_seqlens_k,
                                   max_seqlen_q=max_seqlen_q,
                                   max_seqlen_k=max_seqlen_k,
                                   dropout_p=self.attn_drop.p,
                                   softmax_scale=self.softmax_scale,
                                   causal=self.causal,
                                   window_size=self.window_size,
                                   deterministic=self.deterministic)

        x = x.type(out_dtype)
        x = x.reshape(-1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x

    def forward(self, x, context=None, **kwargs):
        x = getattr(self, self.backend)(x, context=context, **kwargs)
        return x
