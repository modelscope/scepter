# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint_sequential

from scepter.modules.model.backbone.transformer.layers import (Mlp,
                                                               T2IFinalLayer,
                                                               TimestepEmbedder
                                                               )
from scepter.modules.model.backbone.transformer.patchify import PatchEmbed
from scepter.modules.model.backbone.transformer.pos_embed import rope_params
from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.registry import BACKBONES
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.file_system import FS

from .layers import ACEBlock


@BACKBONES.register_class()
class ACE(BaseModel):

    para_dict = {
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
        'PRED_SIGMA': {
            'value': True,
            'description': ''
        },
        'DROP_PATH': {
            'value': 0.,
            'description': ''
        },
        'WINDOW_SIZE': {
            'value': 0,
            'description': ''
        },
        'WINDOW_BLOCK_INDEXES': {
            'value': None,
            'description': ''
        },
        'Y_CHANNELS': {
            'value': 4096,
            'description': ''
        },
        'ATTENTION_BACKEND': {
            'value': None,
            'description': ''
        },
        'QK_NORM': {
            'value': True,
            'description': 'Whether to use RMSNorm for query and key.',
        },
    }
    para_dict.update(BaseModel.para_dict)

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
        self.y_channels = cfg.get('Y_CHANNELS', 4096)
        self.drop_path = cfg.get('DROP_PATH', 0.)
        self.depth = cfg.get('DEPTH', 28)
        self.mlp_ratio = cfg.get('MLP_RATIO', 4.0)
        self.use_grad_checkpoint = cfg.get('USE_GRAD_CHECKPOINT', False)
        self.attention_backend = cfg.get('ATTENTION_BACKEND', None)
        self.max_seq_len = cfg.get('MAX_SEQ_LEN', 1024)
        self.qk_norm = cfg.get('QK_NORM', False)
        self.ignore_keys = cfg.get('IGNORE_KEYS', [])
        assert (self.hidden_size % self.num_heads
                ) == 0 and (self.hidden_size // self.num_heads) % 2 == 0
        d = self.hidden_size // self.num_heads
        self.freqs = torch.cat(
            [
                rope_params(self.max_seq_len, d - 4 * (d // 6)),  # T (~1/3)
                rope_params(self.max_seq_len, 2 * (d // 6)),  # H (~1/3)
                rope_params(self.max_seq_len, 2 * (d // 6))  # W (~1/3)
            ],
            dim=1)

        # init embedder
        self.x_embedder = PatchEmbed(self.patch_size,
                                     self.in_channels + 1,
                                     self.hidden_size,
                                     bias=True,
                                     flatten=False)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = Mlp(in_features=self.y_channels,
                              hidden_features=self.hidden_size,
                              out_features=self.hidden_size,
                              act_layer=lambda: nn.GELU(approximate='tanh'),
                              drop=0)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))
        # init blocks
        drop_path = [
            x.item() for x in torch.linspace(0, self.drop_path, self.depth)
        ]
        self.blocks = nn.ModuleList([
            ACEBlock(self.hidden_size,
                     self.num_heads,
                     mlp_ratio=self.mlp_ratio,
                     drop_path=drop_path[i],
                     window_size=self.window_size
                     if i in self.window_block_indexes else 0,
                     backend=self.attention_backend,
                     use_condition=True,
                     qk_norm=self.qk_norm) for i in range(self.depth)
        ])
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
                    if self.ignore_keys is not None:
                        if (isinstance(self.ignore_keys, str) and re.match(self.ignore_keys, k)) or \
                                (isinstance(self.ignore_keys, list) and k in self.ignore_keys):
                            continue
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
                    elif 'y_embedder.y_proj.' in k:
                        new_ckpt[k.replace('y_embedder.y_proj.',
                                           'y_embedder.')] = v
                    elif k in ('x_embedder.proj.weight'):
                        model_p = self.state_dict()[k]
                        if v.shape != model_p.shape:
                            model_p.zero_()
                            model_p[:, :4, :, :].copy_(v)
                            new_ckpt[k] = torch.nn.parameter.Parameter(model_p)
                        else:
                            new_ckpt[k] = v
                    elif k in ('x_embedder.proj.bias'):
                        new_ckpt[k] = v
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
                text_position_embeddings=None,
                gc_seg=-1,
                **kwargs):
        if self.freqs.device != x.device:
            self.freqs = self.freqs.to(x.device)
        if isinstance(cond, dict):
            context = cond.get('crossattn', None)
        else:
            context = cond
        if text_position_embeddings is not None:
            # default use the text_position_embeddings in state_dict
            # if state_dict doesn't including this key, use the arg: text_position_embeddings
            proj_position_embeddings = self.y_embedder(
                text_position_embeddings)
        else:
            proj_position_embeddings = None

        ctx_batch, txt_lens = [], []
        if mask is not None and isinstance(mask, list):
            for ctx, ctx_mask in zip(context, mask):
                for frame_id, one_ctx in enumerate(zip(ctx, ctx_mask)):
                    u, m = one_ctx
                    t_len = m.flatten().sum()  # l
                    u = u[:t_len]
                    u = self.y_embedder(u)
                    if frame_id == 0:
                        u = u + proj_position_embeddings[
                            len(ctx) -
                            1] if proj_position_embeddings is not None else u
                    else:
                        u = u + proj_position_embeddings[
                            frame_id -
                            1] if proj_position_embeddings is not None else u
                    ctx_batch.append(u)
                    txt_lens.append(t_len)
        else:
            raise TypeError
        y = torch.cat(ctx_batch, dim=0)
        txt_lens = torch.LongTensor(txt_lens).to(x.device, non_blocking=True)

        batch_frames = []
        for u, shape, m in zip(x, cond['x_shapes'], cond['x_mask']):
            u = u[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            m = torch.ones_like(u[[0], :, :]) if m is None else m.squeeze(0)
            batch_frames.append([torch.cat([u, m], dim=0).unsqueeze(0)])
        if 'edit' in cond:
            for i, (edit, edit_mask) in enumerate(
                    zip(cond['edit'], cond['edit_mask'])):
                if edit is None:
                    continue
                for u, m in zip(edit, edit_mask):
                    u = u.squeeze(0)
                    m = torch.ones_like(
                        u[[0], :, :]) if m is None else m.squeeze(0)
                    batch_frames[i].append(
                        torch.cat([u, m], dim=0).unsqueeze(0))

        patch_batch, shape_batch, self_x_len, cross_x_len = [], [], [], []
        for frames in batch_frames:
            patches, patch_shapes = [], []
            self_x_len.append(0)
            for frame_id, u in enumerate(frames):
                u = self.x_embedder(u)
                h, w = u.size(2), u.size(3)
                u = rearrange(u, '1 c h w -> (h w) c')
                if frame_id == 0:
                    u = u + proj_position_embeddings[
                        len(frames) -
                        1] if proj_position_embeddings is not None else u
                else:
                    u = u + proj_position_embeddings[
                        frame_id -
                        1] if proj_position_embeddings is not None else u
                patches.append(u)
                patch_shapes.append([h, w])
                cross_x_len.append(h * w)  # b*s, 1
                self_x_len[-1] += h * w  # b, 1
            # u = torch.cat(patches, dim=0)
            patch_batch.extend(patches)
            shape_batch.append(
                torch.LongTensor(patch_shapes).to(x.device, non_blocking=True))
        # repeat t to align with x
        t = torch.cat([t[i].repeat(l) for i, l in enumerate(self_x_len)])
        self_x_len, cross_x_len = (torch.LongTensor(self_x_len).to(
            x.device, non_blocking=True), torch.LongTensor(cross_x_len).to(
                x.device, non_blocking=True))
        # x = pad_sequence(tuple(patch_batch), batch_first=True)  # b, s*max(cl), c
        x = torch.cat(patch_batch, dim=0)
        x_shapes = pad_sequence(tuple(shape_batch),
                                batch_first=True)  # b, max(len(frames)), 2
        t = self.t_embedder(t)  # (N, D)
        t0 = self.t_block(t)
        # y = self.y_embedder(context)

        kwargs = dict(y=y,
                      t=t0,
                      x_shapes=x_shapes,
                      self_x_len=self_x_len,
                      cross_x_len=cross_x_len,
                      freqs=self.freqs,
                      txt_lens=txt_lens)
        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.blocks],
                segments=gc_seg if gc_seg > 0 else len(self.blocks),
                input=x,
                use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, **kwargs)
        x = self.final_layer(x, t)  # b*s*n, d
        outs, cur_length = [], 0
        p = self.patch_size
        for seq_length, shape in zip(self_x_len, shape_batch):
            x_i = x[cur_length:cur_length + seq_length]
            h, w = shape[0].tolist()
            u = x_i[:h * w].view(h, w, p, p, -1)
            u = rearrange(u, 'h w p q c -> (h p w q) c'
                          )  # dump into sequence for following tensor ops
            cur_length = cur_length + seq_length
            outs.append(u)
        x = pad_sequence(tuple(outs), batch_first=True).permute(0, 2, 1)
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
        # Initialize caption embedding MLP:
        if hasattr(self, 'y_embedder'):
            nn.init.normal_(self.y_embedder.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.fc2.weight, std=0.02)
        # Zero-out adaLN modulation layers
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
                            ACE.para_dict,
                            set_name=True)
