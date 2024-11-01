# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# All rights reserved.
# This file contains code that is adapted from
# timm: https://github.com/huggingface/pytorch-image-models
# pixart: https://github.com/PixArt-alpha/PixArt-alpha
from itertools import repeat as iter_repeat
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.cuda import amp
from torch.nn.utils.rnn import pad_sequence


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(iter_repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def get_2d_sincos_pos_embed(embed_dim,
                            grid_size,
                            cls_token=False,
                            extra_tokens=0,
                            lewei_scale=1.0,
                            base_h_size=16.,
                            base_w_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (
        grid_size[0] / base_h_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (
        grid_size[1] / base_w_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

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

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def apply_2d_rope(xq,
                  xk,
                  padded_pos_index,
                  num_head,
                  head_dim,
                  rotary_base=10000):
    '''
        x query/key: [b, seq, num_head*head_dim]
        padded_pos_index: [b, seq, 2]
    '''
    b = xq.shape[0]
    assert head_dim % 4 == 0, 'the 2d_rope dims should be divided by 4'
    rope_dim = head_dim // 2  # 2d_rope_dim, 1d_rope_dim = head_dim
    # 1. theta_d = b ** (-2d/D)
    theta = 1.0 / (rotary_base**(
        torch.arange(0, rope_dim, 2)[:(rope_dim // 2)].float() / rope_dim))
    # 2. [h * Theta || w * Theta]
    theta = theta.to(xq.device).expand(b, 1, rope_dim // 2)
    freqs_h = torch.bmm(padded_pos_index[:, :, :1],
                        theta).float()  # h * \theta
    freqs_w = torch.bmm(padded_pos_index[:, :, 1:],
                        theta).float()  # w * \theta
    freqs = torch.cat([freqs_h, freqs_w], dim=2).repeat(1, 1,
                                                        num_head)  # multi-head
    # 3. as_complex for complex multiply
    # if freqs = [x, y] then freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(
        torch.ones_like(freqs),
        freqs)  # torch.polar(abs, angle)=> abs⋅cos(angle)+abs⋅sin(angle)⋅j
    # xq.shape = [b, seq_len, dim]
    # xq_.shape = [b, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.view_as_complex(
        xq_)  # [b, seq_len, dim // 2, 2]=>xq.shape = [b, seq_len, dim]
    xk_ = torch.view_as_complex(xk_)
    # 4. complex multiply and as real
    # xq_out.shape = [b, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(
        2)  # point_wise mul, then flatten eg[[1,2],[3,4],[5,6]]->[1,2,3,4,5,6]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.float()


def frame_pad(x, seq_len, shapes):
    max_h, max_w = np.max(shapes, 0)
    frames = []
    cur_len = 0
    for h, w in shapes:
        frame_len = h * w
        frames.append(
            F.pad(
                x[cur_len:cur_len + frame_len].view(h, w, -1),
                (0, 0, 0, max_w - w, 0, max_h - h))  # .view(max_h * max_w, -1)
        )
        cur_len += frame_len
        if cur_len >= seq_len:
            break
    return torch.stack(frames)


def frame_unpad(x, shapes):
    max_h, max_w = np.max(shapes, 0)
    x = rearrange(x, '(b h w) n c -> b h w n c', h=max_h, w=max_w)
    frames = []
    for i, (h, w) in enumerate(shapes):
        if i >= len(x):
            break
        frames.append(rearrange(x[i, :h, :w], 'h w n c -> (h w) n c'))
    return torch.concat(frames)


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """
    Precompute the frequency tensor for complex exponentials.
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2).type_as(x)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output)


@amp.autocast(enabled=False)
def rope_apply_multires_pad(x, x_lens, x_shapes, freqs, pad=True):
    """
    x:          [B, L, N, C].
    x_lens:     [B].
    x_shapes:   [B, F, 2].
    freqs:      [M, C // 2].
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (seq_len,
            shapes) in enumerate(zip(x_lens.tolist(), x_shapes.tolist())):
        x_i = frame_pad(x[i], seq_len, shapes)  # f, h, w, c
        f, h, w = x_i.shape[:3]
        pad_seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x_i.to(torch.float64).reshape(pad_seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(pad_seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2).type_as(x)
        x_i = frame_unpad(x_i, shapes)
        if pad:
            x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output) if pad else torch.concat(output)


@amp.autocast(enabled=False)
def rope_apply_multires(x, x_lens, x_shapes, freqs, pad=True):
    """
    x:          [B*L, N, C].
    x_lens:     [B].
    x_shapes:   [B, F, 2].
    freqs:      [M, C // 2].
    """
    n, c = x.size(1), x.size(2) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # loop over samples
    output = []
    st = 0
    for i, (seq_len,
            shapes) in enumerate(zip(x_lens.tolist(), x_shapes.tolist())):
        x_i = frame_pad(x[st:st + seq_len], seq_len, shapes)  # f, h, w, c
        f, h, w = x_i.shape[:3]
        pad_seq_len = f * h * w
        # precompute multipliers
        x_i = torch.view_as_complex(
            x_i.to(torch.float64).reshape(pad_seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(pad_seq_len, 1, -1)
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2).type_as(x)
        x_i = frame_unpad(x_i, shapes)
        # append to collection
        output.append(x_i)
        st += seq_len
    return pad_sequence(output) if pad else torch.concat(output)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64,
                         device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum('...n,d->...nd', pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out),
         torch.sin(out),
         torch.cos(out)],
        dim=-1)
    out = rearrange(out, 'b n d (i j) -> b n d i j', i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor,
               freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(
        *xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [
                rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(n_axes)
            ],
            dim=-3,
        )

        return emb.unsqueeze(1)
