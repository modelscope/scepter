# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# All rights reserved.
# This file contains code that is adapted from
# timm: https://github.com/huggingface/pytorch-image-models
# pixart: https://github.com/PixArt-alpha/PixArt-alpha
import math
import time
import warnings

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from scepter.modules.model.backbone.transformer.pos_embed import apply_2d_rope

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except Exception as e:
    XFORMERS_IS_AVAILABLE = False
    warnings.warn(f'{e}')
try:
    from flash_attn import (flash_attn_varlen_func)
    FLASHATTN_IS_AVAILABLE = True
except ImportError:
    FLASHATTN_IS_AVAILABLE = False
    flash_attn_varlen_func = None


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


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
        elif XFORMERS_IS_AVAILABLE and self.backend in ('xformer_attn', None):
            self.backend = 'xformer_attn'
        else:
            self.backend = 'pytorch_attn'

    def xformer_attn(self, x, context=None, mask=None, **kwargs):
        context = x if context is None else context
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.q(x).view(b, -1, n, d)
        k = self.k(context).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        attn_bias = None
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            # To use an `attn_bias` with a sequence length that is not a multiple of 8,
            # you need to ensure memory is aligned by slicing a bigger tensor.
            # Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]`
            # instead of `torch.zeros([1, 1, 5, 5])
            q_size = math.ceil(q.size(1) / 8) * 8
            k_size = math.ceil(k.size(1) / 8) * 8
            attn_bias = x.new_zeros(b, n, q_size,
                                    k_size)[:, :, :q.size(1), :k.size(1)]
            attn_bias = attn_bias.masked_fill_(mask == 0,
                                               torch.finfo(x.dtype).min).to(
                                                   q.dtype)
        x = xformers.ops.memory_efficient_attention(q,
                                                    k,
                                                    v,
                                                    p=self.attn_drop.p,
                                                    attn_bias=attn_bias)
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x

    def flash_attn(self, x, context=None, mask=None, **kwargs):
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
        context = x if context is None else context
        dtype = kwargs.get('dtype', torch.float16)
        q_lens = kwargs.get('q_lens', None)

        # if mask is not None or q_lens is not None:
        #     warnings.warn("Detected mask or q_lens is not None, "
        #                   "which will be very slow because of the x/context features' rearrangement,"
        #                   "please use FlashMultiHeadAttention instead.")
        def half(x):
            return x if x.dtype in [torch.float16, torch.bfloat16
                                    ] else x.to(dtype)

        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.q(x).view(b, -1, n, d)  # [B, Lq, Nq, C1].
        k = self.k(context).view(b, -1, n, d)  # [B, Lk, Nk, C1]
        v = self.v(context).view(
            b, -1, n, d)  # [B, Lk, Nk, C2] Nq must be divisible by Nk.

        assert q.device.type == 'cuda' and q.size(-1) <= 256
        lq, lk, out_dtype = int(q.size(1)), int(k.size(1)), q.dtype
        # preprocess query
        if q_lens is None:
            q_lens = torch.tensor([lq] * b,
                                  dtype=torch.int32).to(q.device,
                                                        non_blocking=True)
            # q_lens = (q.flatten(2, ).bool() + 1).sum(dim=-1).bool().sum(dim=-1)
            q = half(q.flatten(0, 1))
        else:
            q = half(torch.cat([q_v[:q_l] for q_v, q_l in zip(q, q_lens)]))

        # preprocess key, value
        if mask is None:
            k_lens = torch.tensor([lk] * b,
                                  dtype=torch.int32).to(k.device,
                                                        non_blocking=True)
            # k_lens = (k.flatten(2, ).bool() + 1).sum(dim=-1).bool().sum(dim=-1)
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
        else:
            assert mask.ndim in [1, 2, 3]
            k_lens = mask if mask.ndim == 1 else mask.flatten(start_dim=1).sum(
                dim=-1)
            k = half(torch.cat([k_v[:k_l] for k_v, k_l in zip(k, k_lens)]))
            v = half(torch.cat([v_v[:v_l] for v_v, v_l in zip(v, k_lens)]))

        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]),
                                    q_lens]).cumsum(0, dtype=torch.int32),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]),
                                    k_lens]).cumsum(0, dtype=torch.int32),
            max_seqlen_q=int(torch.max(q_lens).cpu().numpy()),
            max_seqlen_k=int(torch.max(k_lens).cpu().numpy()),
            dropout_p=self.attn_drop.p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=self.window_size,  # -1 means infinite context window
            deterministic=self.deterministic).unflatten(0, (b, lq))
        x = x.type(out_dtype)
        x = x.flatten(2)
        # output
        x = self.o(x)
        x = self.dropout(x)
        return x

    def pytorch_attn(self, x, context=None, mask=None, **kwargs):
        """x:       [B, L, C].
           context: [B, L', C'] or None.
        """
        context = x if context is None else context
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, d)
        k = self.k(context).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias = attn_bias.masked_fill_(mask == 0,
                                               torch.finfo(x.dtype).min).to(
                                                   q.dtype)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum('binc,bjnc->bnij', q * self.scale,
                            k * self.scale) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v.float())
        # output
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x

    def forward(self, x, context=None, mask=None, **kwargs):
        """x: [B, L, C].
           context: [B, L', C'] or None.
        """
        x = getattr(self, self.backend)(x,
                                        context=context,
                                        mask=mask,
                                        **kwargs)
        return x


def flash_preprocess(x, context=None, q_mask=None, mask=None):
    context = x if context is None else context
    b, x_l, x_hidden_size = x.shape
    x = x.flatten(0, 1)
    if q_mask is None:
        q_lens = torch.tensor([x_l] * b,
                              dtype=torch.int32).to(x.device,
                                                    non_blocking=True)
    else:
        assert q_mask.ndim in [1, 2, 3]
        q_lens = q_mask if q_mask.ndim == 1 else q_mask.flatten(
            start_dim=1).sum(dim=-1)

    mask_b, mask_l, mask_hidden_size = context.shape

    if mask is None:
        mask_lens = torch.tensor([mask_l] * mask_b,
                                 dtype=torch.int32).to(context.device,
                                                       non_blocking=True)
    else:
        assert mask.ndim in [1, 2, 3]
        mask_lens = mask if mask.ndim == 1 else mask.flatten(start_dim=1).sum(
            dim=-1)

    return_data = {
        'x':
        x,
        'context':
        torch.cat([u[:v] for u, v in zip(context, mask_lens)]),
        'cu_seqlens_q':
        torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0,
                                                          dtype=torch.int32),
        'max_seqlen_q':
        int(torch.max(q_lens).cpu().numpy()),
        'cu_seqlens_k':
        torch.cat([mask_lens.new_zeros([1]),
                   mask_lens]).cumsum(0, dtype=torch.int32),
        'max_seqlen_k':
        int(torch.max(mask_lens).cpu().numpy())
    }
    return return_data


class FlashMultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 attn_drop=0.0,
                 qkv_bias=False,
                 dropout=0.0,
                 softmax_scale=None,
                 causal=False,
                 window_size=(-1, -1),
                 deterministic=False,
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
        self.dropout = nn.Dropout(dropout)
        self.attention_op = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(self,
                x,
                context=None,
                cu_seqlens_q=None,
                max_seqlen_q=None,
                cu_seqlens_k=None,
                max_seqlen_k=None,
                dtype=torch.float16,
                **kwargs):
        '''
            The implementation used the rearanaged x/context according to q_lens or k_lens.
            Args:
                x: [batch_size * max_seq_len or sum(q_lens) , heads, hidden_size].
                context:  [batch_size * max_seq_len or sum(q_lens) , heads, hidden_size].
                cu_seqlens_q: cumsum of seq_q to index the postion of query in the batch.
                max_seqlen_q: max length of query.
                cu_seqlens_k: cumsum of seq_k to index the postion of key/value in the batch.
                max_seqlen_k: max length of key/value.
                dtype: the dtype for attention.
                **kwargs:
            Returns: x
            '''
        context = x if context is None else context

        def half(x):
            return x if x.dtype in [torch.float16, torch.bfloat16
                                    ] else x.to(dtype)

        n, d, out_dtype = self.num_heads, self.head_dim, x.dtype
        q = self.q(x).view(-1, n, d)  # [B * Lq, Nq, C1].
        k = self.k(context).view(-1, n, d)  # [B * Lk, Nk, C1]
        v = self.v(context).view(
            -1, n, d)  # [B * Lk, Nk, C2] Nq must be divisible by Nk.
        q, k, v = half(q), half(k), half(v)
        assert q.device.type == 'cuda' and d <= 256
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.attn_drop.p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            window_size=self.window_size,  # -1 means infinite context window
            deterministic=self.deterministic).unflatten(0, (x.shape[0], ))
        x = x.flatten(1).type(out_dtype)
        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


def multi_head_varlen_attention(q_img,
                                k_img,
                                v_img,
                                q_txt,
                                k_txt,
                                v_txt,
                                n,
                                d,
                                img_lens,
                                txt_lens,
                                dropout_p=0.0,
                                flash_dtype=torch.bfloat16):
    '''
        q/k/v: b, s, n*d
        q_lens/k_lens: b,
    '''
    from flash_attn import flash_attn_varlen_func
    q_lens = k_lens = img_lens + txt_lens

    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]),
                              q_lens]).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]),
                              k_lens]).cumsum(0, dtype=torch.int32)
    max_seqlen_q = q_lens.max()
    max_seqlen_k = k_lens.max()

    # concat img & txt for joint attention
    q = torch.cat([
        torch.cat([i[:i_len], t[:t_len]], dim=0)
        for i, i_len, t, t_len in zip(q_img, img_lens, q_txt, txt_lens)
    ],
                  dim=0).view(-1, n, d)

    k = torch.cat([
        torch.cat([i[:i_len], t[:t_len]], dim=0)
        for i, i_len, t, t_len in zip(k_img, img_lens, k_txt, txt_lens)
    ],
                  dim=0).view(-1, n, d)

    v = torch.cat([
        torch.cat([i[:i_len], t[:t_len]], dim=0)
        for i, i_len, t, t_len in zip(v_img, img_lens, v_txt, txt_lens)
    ],
                  dim=0).view(-1, n, d)

    # attention
    dtype = q.dtype
    if dtype != flash_dtype:
        q = q.type(flash_dtype)
        k = k.type(flash_dtype)
        v = v.type(flash_dtype)

    with amp.autocast():
        x = flash_attn_varlen_func(q=q,
                                   k=k,
                                   v=v,
                                   cu_seqlens_q=cu_seqlens_q,
                                   cu_seqlens_k=cu_seqlens_k,
                                   max_seqlen_q=max_seqlen_q,
                                   max_seqlen_k=max_seqlen_k,
                                   dropout_p=dropout_p).type(dtype)

    return x, cu_seqlens_q


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class FullAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=None,
                 head_dim=None,
                 dropout=0.0,
                 qkv_bias=False,
                 qk_norm=False,
                 eps=1e-6,
                 flash_dtype=torch.bfloat16):

        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        assert flash_dtype in (None, torch.float16, torch.bfloat16)
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)
        self.flash_dtype = flash_dtype
        # layers
        self.qkv_W = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if qk_norm:
            from apex.normalization import FusedRMSNorm
            self.q_img_norm = FusedRMSNorm(head_dim, eps=eps)
            self.k_img_norm = FusedRMSNorm(head_dim, eps=eps)
            self.q_txt_norm = FusedRMSNorm(head_dim, eps=eps)
            self.k_txt_norm = FusedRMSNorm(head_dim, eps=eps)
        else:
            self.q_img_norm = nn.Identity()
            self.k_img_norm = nn.Identity()
            self.q_txt_norm = nn.Identity()
            self.k_txt_norm = nn.Identity()

    def forward(self,
                img,
                txt,
                img_lens=None,
                txt_lens=None,
                padded_pos_index=None):
        '''
            img: B, L, C
            txt: B, L', C
        '''
        b, img_len, c = img.shape
        txt_len, n, d = txt.shape[1], self.num_heads, self.head_dim

        # compute query, key, value
        img_txt = torch.cat([img, txt], dim=1)
        img_tokens, txt_tokens = torch.split(self.qkv_W(img_txt),
                                             [img_len, txt_len],
                                             dim=1)

        q_img, k_img, v_img = img_tokens.chunk(3, dim=-1)
        q_txt, k_txt, v_txt = txt_tokens.chunk(3, dim=-1)

        # multi-head qk norm
        q_img, k_img = q_img.view(b, -1, n, d), k_img.view(b, -1, n, d)
        q_txt, k_txt = q_txt.view(b, -1, n, d), k_txt.view(b, -1, n, d)
        q_img, q_txt = self.q_img_norm(q_img).view(
            b, -1, n * d), self.q_txt_norm(q_txt).view(b, -1, n * d)
        k_img, k_txt = self.k_img_norm(k_img).view(
            b, -1, n * d), self.k_txt_norm(k_txt).view(b, -1, n * d)

        ### add position
        q_img, k_img = apply_2d_rope(q_img, k_img, padded_pos_index, n, d)

        # support varying length
        if img_lens is None:
            img_lens = torch.tensor([img.size(1)] * b,
                                    dtype=torch.int32,
                                    device=img.device)
        if txt_lens is None:
            txt_lens = torch.tensor([txt.size(1)] * b,
                                    dtype=torch.int32,
                                    device=txt.device)

        # attention
        x, cu_seqlens_q = multi_head_varlen_attention(
            q_img,
            k_img,
            v_img,
            q_txt,
            k_txt,
            v_txt,
            n,
            d,
            img_lens,
            txt_lens,
            dropout_p=self.dropout.p if self.training else 0.0,
            flash_dtype=self.flash_dtype)

        # output proj.
        x = x.reshape(-1, n * d)
        x = self.out_proj(x)
        x = self.dropout(x)

        # split img & txt and padding to max_len
        img = pad_sequence(tuple([
            x[s:s + img_len] for s, e, img_len in zip(
                cu_seqlens_q[:-1], cu_seqlens_q[1:], img_lens)
        ]),
                           batch_first=True)
        txt = pad_sequence(tuple([
            x[s + img_len:e] for s, e, img_len in zip(
                cu_seqlens_q[:-1], cu_seqlens_q[1:], img_lens)
        ]),
                           batch_first=True)

        return img, txt


class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.W1 = nn.Linear(in_features, hidden_features, bias=False)
        self.W2 = nn.Linear(in_features, hidden_features, bias=False)
        self.W3 = nn.Linear(hidden_features, in_features, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.W3(self.silu(self.W1(x)) * self.W2(x))


if __name__ == '__main__':
    # Align results for different attention implementation
    torch.manual_seed(2023)
    hidden_dim = 4096
    q_weight = torch.randn((hidden_dim, hidden_dim))
    q_bias = torch.zeros((hidden_dim))
    k_weight = torch.randn((hidden_dim, hidden_dim))
    k_bias = torch.zeros((hidden_dim))
    v_weight = torch.randn((hidden_dim, hidden_dim))
    v_bias = torch.zeros((hidden_dim))
    o_weight = torch.randn((hidden_dim, hidden_dim))
    o_bias = torch.randn((hidden_dim))
    pytorch_attn = MultiHeadAttention(hidden_dim,
                                      context_dim=hidden_dim,
                                      num_heads=32,
                                      head_dim=None,
                                      attn_drop=0.0,
                                      dropout=0.0,
                                      backend='pytorch_attn')

    pytorch_attn.load_state_dict({
        'q.weight': q_weight,
        'k.weight': k_weight,
        'v.weight': v_weight,
        'o.weight': o_weight,
        'o.bias': o_bias
    })
    pytorch_attn.to(0)

    xformer_attn = MultiHeadAttention(hidden_dim,
                                      context_dim=hidden_dim,
                                      num_heads=32,
                                      head_dim=None,
                                      attn_drop=0.0,
                                      dropout=0.0,
                                      backend='xformer_attn')

    xformer_attn.load_state_dict({
        'q.weight': q_weight,
        'k.weight': k_weight,
        'v.weight': v_weight,
        'o.weight': o_weight,
        'o.bias': o_bias
    })
    xformer_attn.to(0)

    flash_attn = MultiHeadAttention(hidden_dim,
                                    context_dim=hidden_dim,
                                    num_heads=32,
                                    head_dim=None,
                                    attn_drop=0.0,
                                    dropout=0.0,
                                    backend='flash_attn',
                                    dtype=torch.float16)
    flash_attn.load_state_dict({
        'q.weight': q_weight,
        'k.weight': k_weight,
        'v.weight': v_weight,
        'o.weight': o_weight,
        'o.bias': o_bias
    })
    flash_attn.to(0)

    improved_flash_attn = FlashMultiHeadAttention(hidden_dim,
                                                  context_dim=hidden_dim,
                                                  num_heads=32,
                                                  head_dim=None,
                                                  attn_drop=0.0,
                                                  dropout=0.0,
                                                  backend='flash_attn',
                                                  dtype=torch.float16)

    improved_flash_attn.load_state_dict({
        'q.weight': q_weight,
        'k.weight': k_weight,
        'v.weight': v_weight,
        'o.weight': o_weight,
        'o.bias': o_bias
    })
    improved_flash_attn.to(0)

    batch_size = 1
    query_length = 1024
    key_length = 1024
    # mask = None
    run_num = 10
    torch.cuda.empty_cache()
    x = torch.randn((batch_size, query_length, hidden_dim)).to(0)
    context = torch.randn((batch_size, key_length, hidden_dim)).to(0)
    # mask = torch.cat([torch.ones((batch_size, 80)), torch.zeros((batch_size, key_length - 80))], dim=1).long().to(0)
    # mask = torch.randint(1, key_length, [batch_size]).to(0)
    mask = None
    st = time.time()
    for i in tqdm(range(run_num)):
        pytorch_res = pytorch_attn(x.clone(), context.clone(),
                                   mask.clone() if mask is not None else mask)
        if i == run_num - 1:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_mem, total_mem = free_mem / (1024**3), total_mem / (1024**3)
            mem_msg = f'GPU {0}: free mem {free_mem:.3f}G, total mem {total_mem:.3f}G \n'
            pytorch_res_data = pytorch_res.clone().detach().cpu()
            print('pytorch attn ', mem_msg,
                  f'Cost time per time {(time.time() - st) / run_num}s')
    #
    torch.cuda.empty_cache()
    st = time.time()
    for i in tqdm(range(run_num)):
        xformer_res = xformer_attn(x.clone(), context.clone(),
                                   mask.clone() if mask is not None else mask)
        if i == run_num - 1:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_mem, total_mem = free_mem / (1024**3), total_mem / (1024**3)
            mem_msg = f'GPU {0}: free mem {free_mem:.3f}G, total mem {total_mem:.3f}G \n'
            xformer_res_data = xformer_res.clone().detach().cpu()
            print('xformer attn ', mem_msg,
                  f'Cost time per time {(time.time() - st) / run_num}s')
    #
    torch.cuda.empty_cache()
    # mask = None
    st = time.time()
    for i in tqdm(range(run_num)):
        flash_res = flash_attn(x.clone(), context.clone(),
                               mask.clone() if mask is not None else mask)
        if i == run_num - 1:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_mem, total_mem = free_mem / (1024**3), total_mem / (1024**3)
            mem_msg = f'GPU {0}: free mem {free_mem:.3f}G, total mem {total_mem:.3f}G \n'
            flash_res_data = flash_res.clone().detach().cpu()
            print('flash attn ', mem_msg,
                  f'Cost time per time {(time.time() - st) / run_num}s')

    # recommend this style for multi blocks to save the preprocess time.

    flash_input = flash_preprocess(
        x.clone(),
        context.clone(),
        mask=mask.clone() if mask is not None else mask)
    st = time.time()
    for i in tqdm(range(run_num)):
        improved_flash_res_v1 = improved_flash_attn(**flash_input).reshape(
            (batch_size, -1, hidden_dim))
        if i == 0:
            improved_flash_res_v1_data = improved_flash_res_v1.clone().detach(
            ).cpu()
        if i == run_num - 1:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_mem, total_mem = free_mem / (1024**3), total_mem / (1024**3)
            mem_msg = f'GPU {0}: free mem {free_mem:.3f}G, total mem {total_mem:.3f}G \n'
            print('improved flash attn ', mem_msg,
                  f'Cost time per time {(time.time() - st) / run_num}s')
    #
    print(pytorch_res_data, xformer_res_data, flash_res_data,
          improved_flash_res_v1_data)
    print(pytorch_res_data.shape, xformer_res_data.shape, flash_res_data.shape,
          improved_flash_res_v1_data.shape)
    print(
        torch.sum(pytorch_res_data) / (batch_size * query_length * hidden_dim),
        torch.sum(xformer_res_data) / (batch_size * query_length * hidden_dim),
        torch.sum(flash_res_data) / (batch_size * query_length * hidden_dim),
        torch.sum(improved_flash_res_v1_data) /
        (batch_size * query_length * hidden_dim))
