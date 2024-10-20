# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# All rights reserved.
# This file contains code that is adapted from
# timm: https://github.com/huggingface/pytorch-image-models
# pixart: https://github.com/PixArt-alpha/PixArt-alpha

# This source code is also licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from collections import OrderedDict
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from typing import Iterable

import torch
import torch.nn as nn
# References:
# GLIDE: https://github.com/openai/glide-text2im
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from scepter.modules.model.backbone.transformer.attention import \
    MultiHeadAttention
from scepter.modules.model.backbone.transformer.layers import (
    CaptionEmbedder, DropPath, LabelEmbedder, Mlp, SizeEmbedder,
    TimestepEmbedder, modulate)
from scepter.modules.model.backbone.transformer.patchify import (PatchEmbed,
                                                                 unpatchify)
from scepter.modules.model.backbone.transformer.pos_embed import \
    get_2d_sincos_pos_embed
from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.registry import BACKBONES
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.file_system import FS


def auto_grad_checkpoint(module, *args, use_grad_checkpoint=False, **kwargs):
    if use_grad_checkpoint:
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module,
                                     gc_step,
                                     *args,
                                     use_reentrant=False,
                                     **kwargs)
    return module(*args, **kwargs)


class FinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size * patch_size * out_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # modulate x
        x = modulate(self.norm_final(x), shift, scale, unsqueeze=True)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size * patch_size * out_channels,
                                bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2,
                                                                         dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DitFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size * patch_size * out_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, unsqueeze=True)
        x = self.linear(x)
        return x


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4.0,
                 drop_path=0.,
                 window_size=0,
                 use_rel_pos=False,
                 backend=None,
                 use_condition=True,
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
                                       **block_kwargs)
        if self.use_condition:
            self.cross_attn = MultiHeadAttention(hidden_size,
                                                 context_dim=hidden_size,
                                                 num_heads=num_heads,
                                                 qkv_bias=True,
                                                 backend=backend,
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

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa, unsqueeze=False)))
        if self.use_condition:
            x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp, unsqueeze=False)))
        return x


@BACKBONES.register_class()
class PixArt(BaseModel):
    """
    Diffusion model with a Transformer backbone.
    """
    para_dict = BaseModel.para_dict
    para_dict.update({
        'PATCH_SIZE': {
            'value': 2,
            'description': ''
        },
        'IN_CHANNELS': {
            'value': 4,
            'description': ''
        },
        'HIDDEN_SIZE': {
            'value': 1152,
            'description': ''
        },
        'DEPTH': {
            'value': 28,
            'description': ''
        },
        'NUM_HEADS': {
            'value': 16,
            'description': ''
        },
        'MLP_RATIO': {
            'value': 4.0,
            'description': ''
        },
        'CLASS_DROPOUT_PROB': {
            'value': 0.1,
            'description': ''
        },
        'PRED_SIGMA': {
            'value': True,
            'description': ''
        },
        'DROP_PATH': {
            'value': 0.,
            'description': ''
        },
        'WINDOW_DIZE': {
            'value': 0,
            'description': ''
        },
        'WINDOW_BLOCK_INDEXES': {
            'value': None,
            'description': ''
        },
        'USE_REL_POS': {
            'value': False,
            'description': ''
        },
        'CAPTION_CHANNELS': {
            'value': 4096,
            'description': ''
        },
        'USE_AR_SIZE': {
            'value': True,
            'description': ''
        },
        'DIT_FINAL_LAYER': {
            'value': False,
            'description': ''
        },
        'LEWEI_SCALE': {
            'value': 1.0,
            'description': ''
        },
        'MODEL_MAX_LENGTH': {
            'value': 120,
            'description': ''
        },
        'NUM_CLASSES': {
            'value':
            None,
            'description':
            'The class num for class guided setting, also can be set as continuous.'
        },
        'ATTENTION_BACKEND': {
            'value': None,
            'description': ''
        }
    })

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger=logger)
        self.window_block_indexes = cfg.get('WINDOW_BLOCK_INDEXES', None)
        if self.window_block_indexes is None:
            self.window_block_indexes = []
        self.pred_sigma = cfg.get('PRED_SIGMA', True)
        self.in_channels = cfg.get('IN_CHANNELS', 4)
        self.out_channels = self.in_channels * 2 if self.pred_sigma else self.in_channels
        self.patch_size = cfg.get('PATCH_SIZE', 2)
        self.num_heads = cfg.get('NUM_HEADS', 16)
        self.hidden_size = cfg.get('HIDDEN_SIZE', 1152)
        self.lewei_scale = cfg.get('LEWEI_SCALE', 1.0),
        self.caption_channels = cfg.get('CAPTION_CHANNELS', 4096)
        self.class_dropout_prob = cfg.get('CLASS_DROPOUT_PROB', 0.1)
        self.model_max_length = cfg.get('MODEL_MAX_LENGTH', 120)
        self.drop_path = cfg.get('DROP_PATH', 0.)
        self.depth = cfg.get('DEPTH', 28)
        self.mlp_ratio = cfg.get('MLP_RATIO', 4.0)
        self.num_classes = cfg.get('NUM_CLASSES', None)
        self.use_grad_checkpoint = cfg.get('USE_GRAD_CHECKPOINT', False)
        self.use_ar_size = cfg.get('USE_AR_SIZE', True)
        self.use_dit_final_layer = cfg.get('DIT_FINAL_LAYER', False)
        self.attention_backend = cfg.get('ATTENTION_BACKEND', None)
        self.ignore_keys = cfg.get('IGNORE_KEYS', [])

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_embedder = LabelEmbedder(
                    self.num_classes,
                    self.hidden_size,
                    dropout_prob=self.class_dropout_prob)
            elif self.num_classes == 'continuous':
                print('setting up linear c_adm embedding layer')
                self.label_embedder = nn.Linear(1, self.hidden_size)
            else:
                raise ValueError()

        self.x_embedder = PatchEmbed(self.patch_size,
                                     self.in_channels,
                                     self.hidden_size,
                                     bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)

        # self.base_size = self.input_size // self.patch_size
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))
        if self.num_classes is None:
            self.y_embedder = CaptionEmbedder(
                in_channels=self.caption_channels,
                hidden_size=self.hidden_size,
                uncond_prob=self.class_dropout_prob,
                act_layer=approx_gelu,
                token_num=self.model_max_length)
        if self.use_ar_size:
            self.csize_embedder = SizeEmbedder(self.hidden_size //
                                               3)  # c_size embed
            self.ar_embedder = SizeEmbedder(self.hidden_size //
                                            3)  # aspect ratio embed

        drop_path = [
            x.item() for x in torch.linspace(0, self.drop_path, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock(self.hidden_size,
                        self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        drop_path=drop_path[i],
                        window_size=self.window_size
                        if i in self.window_block_indexes else 0,
                        use_rel_pos=self.use_rel_pos
                        if i in self.window_block_indexes else False,
                        backend=self.attention_backend,
                        use_condition=self.num_classes is None)
            for i in range(self.depth)
        ])
        if self.use_dit_final_layer:
            self.final_layer = DitFinalLayer(self.hidden_size, self.patch_size,
                                             self.out_channels)
        else:
            self.final_layer = T2IFinalLayer(self.hidden_size, self.patch_size,
                                             self.out_channels)

        self.initialize_weights()

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                model = torch.load(local_path, map_location='cpu')
                if 'state_dict' in model:
                    model = model['state_dict']
                new_ckpt = OrderedDict()
                for k, v in model.items():
                    k = k.replace('.cross_attn.q_linear.', '.cross_attn.q.')
                    k = k.replace('.cross_attn.proj.',
                                  '.cross_attn.o.').replace(
                                      '.attn.proj.', '.attn.o.')
                    if '.cross_attn.kv_linear.' in k:
                        k_p, v_p = torch.split(v, v.shape[0] // 2)
                        new_ckpt[k.replace('.cross_attn.kv_linear.',
                                           '.cross_attn.k.')] = k_p
                        new_ckpt[k.replace('.cross_attn.kv_linear.',
                                           '.cross_attn.v.')] = v_p
                    elif '.attn.qkv.' in k:
                        q_p, k_p, v_p = torch.split(v, v.shape[0] // 3)
                        new_ckpt[k.replace('.attn.qkv.', '.attn.q.')] = q_p
                        new_ckpt[k.replace('.attn.qkv.', '.attn.k.')] = k_p
                        new_ckpt[k.replace('.attn.qkv.', '.attn.v.')] = v_p
                    else:
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

    def forward(self,
                x,
                t=None,
                cond=dict(),
                mask=None,
                data_info=None,
                **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        label = None
        if isinstance(cond, dict):
            if 'label' in cond and cond['label'] is not None:
                label = cond['label']
            if 'concat' in cond:
                concat = cond['concat']
                x = torch.cat([x, concat], dim=1)
            context = cond.get('crossattn', None)
        else:
            context = cond

        y = context
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.hidden_size, (h, w),
                                    lewei_scale=self.lewei_scale,
                                    base_h_size=h,
                                    base_w_size=w)).unsqueeze(0).float().to(
                                        x.device)
        x = x + pos_embed

        t = self.t_embedder(t)  # (N, D)
        if self.num_classes is not None and label is not None:
            t = t + self.label_embedder(label, self.training)
        if self.use_ar_size and data_info is not None:
            bs = x.shape[0]
            c_size, ar = data_info['img_hw'], data_info['aspect_ratio']
            csize = self.csize_embedder(c_size, bs)  # (N, D)
            ar = self.ar_embedder(ar, bs)  # (N, D)
            t = t + torch.cat([csize, ar], dim=1)
        t0 = self.t_block(t)
        if self.num_classes is not None:
            y = None
        else:
            y = self.y_embedder(y, self.training)
        for block in self.blocks:
            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                mask,
                use_grad_checkpoint=self.use_grad_checkpoint)
            # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = unpatchify(x, h, w, self.out_channels, self.patch_size,
                       self.patch_size)  # (N, out_channels, H, W)
        if self.pred_sigma:
            return x.chunk(2, dim=1)[0]
        else:
            return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        if self.use_ar_size:
            nn.init.normal_(self.csize_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.csize_embedder.mlp[2].weight, std=0.02)
            nn.init.normal_(self.ar_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ar_embedder.mlp[2].weight, std=0.02)
        if self.num_classes is not None:
            nn.init.normal_(self.label_embedder.embedding_table.weight,
                            std=0.02)
        # Initialize caption embedding MLP:
        if hasattr(self, 'y_embedder'):
            nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        # Zero-out adaLN modulation layers in PixArt blocks:
        if self.num_classes is None:
            for block in self.blocks:
                nn.init.constant_(block.cross_attn.o.weight, 0)
                nn.init.constant_(block.cross_attn.o.bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            PixArt.para_dict,
                            set_name=True)
