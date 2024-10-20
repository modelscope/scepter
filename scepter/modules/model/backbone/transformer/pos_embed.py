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
