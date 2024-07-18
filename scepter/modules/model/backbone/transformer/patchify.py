# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# All rights reserved.
# This file contains code that is adapted from
# timm: https://github.com/huggingface/pytorch-image-models
# pixart: https://github.com/PixArt-alpha/PixArt-alpha
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def unpatchify(x, h, w, c, p_h, p_w):
    '''
    Args:
        x: input tensor for unpatchified with shape as  (N, T, patch_size**2 * C).
        h: tokens' number align height
        w: tokens' number align width
        c: output channels
        p_h: patch size for h
        p_w: patch size for w
    Returns: unpatchified imgs with shape as (N, H, W, C)
    '''
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(shape=(x.shape[0], c, h * p_h, w * p_w))
