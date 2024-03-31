# -*- coding: utf-8 -*-
"""Concise re-implementation of ``https://github.com/openai/CLIP'' and
    ``https://github.com/mlfoundations/open_clip''.
"""
import math
from functools import partial
from importlib import find_loader

import torch
import torch.nn as nn
import torch.nn.functional as F

from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.embedder.xlm_roberta import \
    XLMRoberta  # used in XLMRobertaCLIP (multilingual)
from scepter.modules.model.registry import EMBEDDERS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS


def map_dtype(m, dtype=torch.float16):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        _ = m.to(dtype)
    elif isinstance(m, LayerNorm):
        _ = m.float()
    elif hasattr(m, 'head') and isinstance(m.head, nn.Parameter):
        p = getattr(m, 'head')
        p.data = p.data.to(dtype)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 causal=False,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 flash_dtype=torch.float16):
        assert dim % num_heads == 0
        assert flash_dtype in (None, torch.float16, torch.bfloat16)
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.scale = math.pow(self.head_dim, -0.25)
        self.flash_dtype = flash_dtype

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """x:   [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        qkv = self.to_qkv(x).view(b, s, 3, n, d)

        # compute attention
        if x.device.type != 'cpu' and find_loader('flash_attn') and \
                self.flash_dtype is not None:
            # flash implementation
            from flash_attn.flash_attn_interface import (
                flash_attn_unpadded_qkvpacked_func, )
            dtype = qkv.dtype
            if dtype != self.flash_dtype:
                qkv = qkv.type(self.flash_dtype)
            cu_seqlens = torch.arange(0,
                                      b * s + 1,
                                      s,
                                      dtype=torch.int32,
                                      device=x.device)
            x = flash_attn_unpadded_qkvpacked_func(
                qkv=qkv.reshape(-1, 3, n, d),
                cu_seqlens=cu_seqlens,
                max_seqlen=s,
                dropout_p=self.attn_dropout if self.training else 0.0,
                causal=self.causal,
                return_attn_probs=False).reshape(b, s, n, d).type(dtype)
        else:
            # torch implementation
            q, k, v = qkv.unbind(2)
            attn = torch.einsum('binc,bjnc->bnij', q * self.scale,
                                k * self.scale)
            if self.causal:
                attn = attn.masked_fill(
                    torch.tril(attn.new_ones(1, 1, s,
                                             s).float()).type_as(attn) == 0,
                    float('-inf'))
            attn = F.softmax(attn.float(), dim=-1).type_as(attn)
            x = torch.einsum('bnij,bjnc->binc', attn, v)

        # output
        x = x.reshape(b, s, c)
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class AttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 causal=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 flash_dtype=torch.float16):
        assert activation in ['quick_gelu', 'gelu']
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.causal = causal
        self.flash_dtype = flash_dtype

        # layers
        self.norm1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout,
                                  proj_dropout, flash_dtype)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 flash_dtype=torch.float16):
        assert image_size % patch_size == 0
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.flash_dtype = flash_dtype

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(3,
                                         dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         bias=False)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            gain * torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim)
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, False, activation,
                           attn_dropout, proj_dropout, flash_dtype)
            for _ in range(num_layers)
        ])
        self.post_norm = LayerNorm(dim)

        # head
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x):
        b, dtype = x.size(0), self.head.dtype
        x = x.type(dtype)

        # patch-embedding
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.repeat(b, 1, 1).type(dtype), x],
                      dim=1)
        x = self.dropout(x + self.pos_embedding.type(dtype))
        x = self.pre_norm(x)

        # transformer
        x = self.transformer(x)

        # head
        x = self.post_norm(x)
        x = torch.mm(x[:, 0, :], self.head)
        return x

    def fp16(self, dtype=torch.float16):
        return self.apply(partial(map_dtype, dtype=dtype))


class TextTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 text_len,
                 dim=512,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=8,
                 num_layers=12,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 flash_dtype=torch.float16):
        super().__init__()
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.flash_dtype = flash_dtype

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(0.01 * torch.randn(1, text_len, dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, True, activation,
                           attn_dropout, proj_dropout, flash_dtype)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(dim)

        # head
        gain = 1.0 / math.sqrt(dim)
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x):
        eot, dtype = x.argmax(dim=-1), self.head.dtype

        # embeddings
        x = self.dropout(
            self.token_embedding(x).type(dtype) +
            self.pos_embedding.type(dtype))

        # transformer
        x = self.transformer(x)

        # head
        x = self.norm(x)
        x = torch.mm(x[torch.arange(x.size(0)), eot], self.head)
        return x

    def fp16(self, dtype=torch.float16):
        return self.apply(partial(map_dtype, dtype=dtype))


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim=512,
                 image_size=224,
                 patch_size=16,
                 vision_dim=768,
                 vision_mlp_ratio=4,
                 vision_heads=12,
                 vision_layers=12,
                 vocab_size=49408,
                 text_len=77,
                 text_dim=512,
                 text_mlp_ratio=4,
                 text_heads=8,
                 text_layers=12,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 flash_dtype=torch.float16,
                 use_module=['visual', 'textual']):
        assert flash_dtype in (None, torch.float16, torch.bfloat16)
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.text_dim = text_dim
        self.text_mlp_ratio = text_mlp_ratio
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.flash_dtype = flash_dtype
        self.use_module = use_module
        # models
        if 'visual' in use_module:
            self.visual = VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                dim=vision_dim,
                mlp_ratio=vision_mlp_ratio,
                out_dim=embed_dim,
                num_heads=vision_heads,
                num_layers=vision_layers,
                activation=activation,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                embedding_dropout=embedding_dropout,
                flash_dtype=flash_dtype)
            self.scale = math.sqrt(self.visual.out_dim)
        else:
            self.visual = nn.Identity()
        if 'textual' in use_module:
            self.textual = TextTransformer(vocab_size=vocab_size,
                                           text_len=text_len,
                                           dim=text_dim,
                                           mlp_ratio=text_mlp_ratio,
                                           out_dim=embed_dim,
                                           num_heads=text_heads,
                                           num_layers=text_layers,
                                           activation=activation,
                                           attn_dropout=attn_dropout,
                                           proj_dropout=proj_dropout,
                                           embedding_dropout=embedding_dropout,
                                           flash_dtype=flash_dtype)
        else:
            self.textual = nn.Identity()
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

        # initialize weights
        self.init_weights()

    def forward(self, imgs, txt_tokens):
        """imgs:        [B, 3, H, W] of torch.float32.
                        mean:   [0.48145466, 0.4578275, 0.40821073]
                        std:    [0.26862954, 0.26130258, 0.27577711]
           txt_tokens:  [B, L] of torch.long.
                        Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_tokens)
        return xi, xt

    def encode_image(self, x, skip_layers=0):
        # clip inference
        b, dtype = x.size(0), self.visual.head.dtype
        x = x.type(dtype)

        # # patch-embedding
        x = self.visual.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat(
            [self.visual.cls_embedding.repeat(b, 1, 1).type(dtype), x], dim=1)
        x = self.visual.dropout(x + self.visual.pos_embedding.type(dtype))
        x = self.visual.pre_norm(x)

        # # transformer
        # assert skip_layers < 12
        if skip_layers == 0:
            x = self.visual.transformer(x)
        else:
            for m in self.visual.transformer[:-skip_layers]:
                x = m(x)

        # # head
        x = self.visual.post_norm(x)
        x = torch.mm(x[:, 0, :], self.visual.head)
        x = self.scale * F.normalize(x, p=2, dim=1)
        return x

    def encode_text(self):
        pass

    def init_weights(self):
        # embeddings
        if 'textual' in self.use_module:
            nn.init.normal_(self.textual.token_embedding.weight, std=0.02)
        if 'visual' in self.use_module:
            nn.init.normal_(self.visual.patch_embedding.weight, std=0.1)

        # attentions
        for modality in self.use_module:
            dim = self.vision_dim if modality == 'visual' else self.text_dim
            transformer = getattr(self, modality).transformer
            proj_gain = (1.0 / math.sqrt(dim)) * (
                1.0 / math.sqrt(2 * len(transformer)))
            attn_gain = 1.0 / math.sqrt(dim)
            mlp_gain = 1.0 / math.sqrt(2.0 * dim)
            for block in transformer:
                nn.init.normal_(block.attn.to_qkv.weight, std=attn_gain)
                nn.init.normal_(block.attn.proj.weight, std=proj_gain)
                nn.init.normal_(block.mlp[0].weight, std=mlp_gain)
                nn.init.normal_(block.mlp[2].weight, std=proj_gain)

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay':
            0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups

    def fp16(self, dtype=torch.float16):
        return self.apply(partial(map_dtype, dtype=dtype))

    def load_from_open_clip(self, checkpoint_or_path, **kwargs):
        """Load and remap state-dict from open-clip.
        """
        # load state-dict
        device = next(self.parameters()).device
        state = checkpoint_or_path
        if isinstance(state, str):
            state = torch.load(state, map_location=device)

        # reorder
        prefix = [
            'logit_scale', 'visual.', 'position', 'text_proj', 'token',
            'transformer.', 'ln_final.'
        ]
        state = type(state)([(k, v) for u in prefix for k, v in state.items()
                             if k.startswith(u)])

        # convert to target keys
        target = self.state_dict()
        target = {
            k: v.view(target[k].shape)
            for k, v in zip(target.keys(), state.values())
        }
        return self.load_state_dict(target, **kwargs)


class XLMRobertaWithHead(XLMRoberta):
    def __init__(self, **kwargs):
        self.out_dim = kwargs.pop('out_dim')
        super().__init__(**kwargs)

        # head
        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.Sequential(nn.Linear(self.dim, mid_dim, bias=False),
                                  nn.GELU(),
                                  nn.Linear(mid_dim, self.out_dim, bias=False))

    def forward(self, tokens):
        # xlm-roberta
        x = super().forward(tokens)

        # average pooling
        mask = tokens.ne(self.pad_token).unsqueeze(-1).to(x)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        # head
        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Module):
    def __init__(self,
                 embed_dim=1024,
                 image_size=224,
                 patch_size=14,
                 vision_dim=1280,
                 vision_mlp_ratio=4,
                 vision_heads=16,
                 vision_layers=32,
                 activation='gelu',
                 vocab_size=250002,
                 max_text_len=514,
                 type_size=1,
                 pad_token=1,
                 text_dim=1024,
                 text_heads=16,
                 text_layers=24,
                 text_eps=1e-5,
                 text_dropout=0.1,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 flash_dtype=torch.float16,
                 use_module=['visual', 'textual']):
        assert flash_dtype in (None, torch.float16, torch.bfloat16)
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_token = pad_token
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_eps = text_eps
        self.flash_dtype = flash_dtype

        # models
        if 'visual' in use_module:
            self.visual = VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                dim=vision_dim,
                mlp_ratio=vision_mlp_ratio,
                out_dim=embed_dim,
                num_heads=vision_heads,
                num_layers=vision_layers,
                activation=activation,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                embedding_dropout=embedding_dropout,
                flash_dtype=flash_dtype)
        else:
            self.visual = nn.Identity()
        if 'textual' in use_module:
            self.textual = XLMRobertaWithHead(vocab_size=vocab_size,
                                              max_seq_len=max_text_len,
                                              type_size=type_size,
                                              pad_token=pad_token,
                                              dim=text_dim,
                                              out_dim=embed_dim,
                                              num_heads=text_heads,
                                              num_layers=text_layers,
                                              dropout=text_dropout,
                                              eps=text_eps)
        else:
            self.textual = nn.Identity()
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

    def forward(self, imgs, txt_tokens):
        """imgs:        [B, 3, H, W] of torch.float32.
                        mean:   [0.48145466, 0.4578275, 0.40821073]
                        std:    [0.26862954, 0.26130258, 0.27577711]
           txt_tokens:  [B, L] of torch.long.
                        Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_tokens)
        return xi, xt

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay':
            0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups

    def fp16(self, dtype=torch.float16):
        return self.apply(partial(map_dtype, dtype=dtype))

    def load_from_open_clip(self, checkpoint_or_path, **kwargs):
        """Load and remap state-dict from open-clip.
        """
        # load state-dict
        device = next(self.parameters()).device
        state = checkpoint_or_path
        if isinstance(state, str):
            state = torch.load(state, map_location=device)
        if 'state_dict' in state:
            state = state['state_dict']

        # reorder
        keys = [
            'logit_scale', 'visual.', 'word_embeddings',
            'token_type_embeddings', 'position_embeddings',
            'embeddings.LayerNorm', 'encoder.layer.', 'text.proj'
        ]
        state = type(state)([(k, v) for u in keys for k, v in state.items()
                             if u in k])

        # target state-dict
        target = self.state_dict()
        target = {
            k: v.view(target[k].shape)
            for k, v in zip(target.keys(), state.values())
        }
        return self.load_state_dict(target, **kwargs)


def _clip(pretrained=False, pretrained_path=None, model_cls=CLIP, **kwargs):
    model = model_cls(**kwargs)
    if pretrained and pretrained_path:
        pretrain_model = torch.load(pretrained_path, map_location='cpu')
        key_str = ' '.join(list(pretrain_model.keys()))
        have_load = False
        if 'use_module' in kwargs and len(kwargs['use_module']) < 2:
            for module_name in kwargs['use_module']:
                if hasattr(model, module_name) and module_name not in key_str:
                    missing, unexpected = getattr(
                        model, module_name).load_state_dict(pretrain_model,
                                                            strict=False)
                    if we.rank == 0:
                        print(f'Restored from {pretrained_path} with'
                              '{len(missing)} missing and {len(unexpected)}'
                              'unexpected keys')
                        if len(missing) > 0:
                            print(f'Missing Keys:\n {missing}')
                        if len(unexpected) > 0:
                            print(f'\nUnexpected Keys:\n {unexpected}')
                    have_load = True
        if not have_load:
            missing, unexpected = model.load_state_dict(pretrain_model,
                                                        strict=False)
            if we.rank == 0:
                print(
                    f'Restored from {pretrained_path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
                )
                if len(missing) > 0:
                    print(f'Missing Keys:\n {missing}')
                if len(unexpected) > 0:
                    print(f'\nUnexpected Keys:\n {unexpected}')
    return model


def clip_vit_b_32(**kwargs):
    cfg = dict(embed_dim=512,
               image_size=224,
               patch_size=32,
               vision_dim=768,
               vision_heads=12,
               vision_layers=12,
               vocab_size=49408,
               text_len=77,
               text_dim=512,
               text_heads=8,
               text_layers=12,
               activation='quick_gelu')
    cfg.update(**kwargs)
    return cfg, CLIP


def clip_vit_b_16(**kwargs):
    cfg = dict(embed_dim=512,
               image_size=224,
               patch_size=16,
               vision_dim=768,
               vision_heads=12,
               vision_layers=12,
               vocab_size=49408,
               text_len=77,
               text_dim=512,
               text_heads=8,
               text_layers=12,
               activation='quick_gelu')
    cfg.update(**kwargs)
    return cfg, CLIP


def clip_vit_l_14(**kwargs):
    cfg = dict(embed_dim=768,
               image_size=224,
               patch_size=14,
               vision_dim=1024,
               vision_heads=16,
               vision_layers=24,
               vocab_size=49408,
               text_len=77,
               text_dim=768,
               text_heads=12,
               text_layers=12,
               activation='quick_gelu')
    cfg.update(**kwargs)
    return cfg, CLIP


def clip_vit_l_14_336px(**kwargs):
    cfg = dict(embed_dim=768,
               image_size=336,
               patch_size=14,
               vision_dim=1024,
               vision_heads=16,
               vision_layers=24,
               vocab_size=49408,
               text_len=77,
               text_dim=768,
               text_heads=12,
               text_layers=12,
               activation='quick_gelu')
    cfg.update(**kwargs)
    return cfg, CLIP


def clip_vit_h_14(**kwargs):
    cfg = dict(embed_dim=1024,
               image_size=224,
               patch_size=14,
               vision_dim=1280,
               vision_heads=16,
               vision_layers=32,
               vocab_size=49408,
               text_len=77,
               text_dim=1024,
               text_heads=16,
               text_layers=24,
               activation='gelu')
    cfg.update(**kwargs)
    return cfg, CLIP


def clip_vit_g_14(**kwargs):
    cfg = dict(embed_dim=1024,
               image_size=224,
               patch_size=14,
               vision_dim=1408,
               vision_mlp_ratio=4.3637,
               vision_heads=16,
               vision_layers=40,
               vocab_size=49408,
               text_len=77,
               text_dim=1024,
               text_heads=16,
               text_layers=24,
               activation='gelu')
    cfg.update(**kwargs)
    return cfg, CLIP


def clip_vit_bigG_14(**kwargs):
    cfg = dict(embed_dim=1280,
               image_size=224,
               patch_size=14,
               vision_dim=1664,
               vision_mlp_ratio=4.9231,
               vision_heads=16,
               vision_layers=48,
               vocab_size=49408,
               text_len=77,
               text_dim=1280,
               text_heads=20,
               text_layers=32,
               activation='gelu')
    cfg.update(**kwargs)
    return cfg, CLIP


def clip_xlm_roberta_vit_h_14(**kwargs):
    cfg = dict(embed_dim=1024,
               image_size=224,
               patch_size=14,
               vision_dim=1280,
               vision_mlp_ratio=4,
               vision_heads=16,
               vision_layers=32,
               activation='gelu',
               vocab_size=250002,
               max_text_len=514,
               type_size=1,
               pad_token=1,
               text_dim=1024,
               text_heads=16,
               text_layers=24,
               text_eps=1e-5,
               text_dropout=0.1,
               attn_dropout=0.0,
               proj_dropout=0.0,
               embedding_dropout=0.0,
               flash_dtype=torch.float16)
    cfg.update(**kwargs)
    return cfg, XLMRobertaCLIP


clip_functions = {
    'clip_vit_b_32': clip_vit_b_32,
    'clip_vit_b_16': clip_vit_b_16,
    'clip_vit_l_14': clip_vit_l_14,
    'clip_vit_l_14_336px': clip_vit_l_14_336px,
    'clip_vit_h_14': clip_vit_h_14,
    'clip_vit_g_14': clip_vit_g_14,
    'clip_vit_bigG_14': clip_vit_bigG_14,
    'clip_xlm_roberta_vit_h_14': clip_xlm_roberta_vit_h_14
}


@EMBEDDERS.register_class()
class ClipEncoder(BaseModel):
    para_dict = {
        'CLIP_FUNC': {
            'value': 'clip_vit_b_32',
            'description':
            f'Select clip model from {list(clip_functions.keys())}'
        },
        'PRETRAINED': {
            'value': False,
            'description': 'Wether load from pretrained model or not.'
        },
        'USE_GRAD': {
            'value': False,
            'description': ''
        },
        'PRETRAINED_PATH': {
            'value': None,
            'description': 'Pretrained model load from.'
        },
        'USE_MODULE': {
            'value': ['visual', 'textual'],
            'description':
            "Use module from visual or textual, default is ['visual', 'textual']."
        },
        'CLIP_SKIP': {
            'value': 2,
            'description': "Textuxl branch skip blocks' num. Default is 2."
        },
        'TOKEN_LENGTH': {
            'value': 77,
            'description': 'The input token length for text. Default is 77.'
        },
        'KWARGS': {}
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        clip_func = cfg.CLIP_FUNC
        pretrained = cfg.get('PRETRAINED', False)
        pretrained_path = cfg.get('PRETRAINED_PATH', None)
        use_module = cfg.get('USE_MODULE', ['visual', 'textual'])
        self.clip_skip = cfg.get('CLIP_SKIP', 2)
        self.use_grad = cfg.get('USE_GRAD', False)
        self.token_length = cfg.get('TOKEN_LENGTH', 77)
        kwargs = {k.lower(): v for k, v in cfg.get('KWARGS', {}).items()}
        if pretrained and pretrained_path:
            local_path = FS.get_from(pretrained_path, wait_finish=True)
        else:
            local_path = None
        assert clip_func in clip_functions
        if clip_func in clip_functions:
            conf, model_cls = clip_functions[clip_func](**kwargs)
            conf['use_module'] = use_module
            self.clip_model = _clip(pretrained, local_path, model_cls, **conf)
            for module in use_module:
                if hasattr(self.clip_model, module):
                    setattr(self, module, getattr(self.clip_model, module))

    def encode_image(self, image):
        if not self.use_grad:
            with torch.no_grad():
                m = self.clip_model.visual
                return m(image)
        else:
            m = self.clip_model.visual
            return m(image)

    def encode_text(self,
                    tokens,
                    tokenizer=None,
                    append_sentence_embedding=True):
        def fn():
            m = self.clip_model.textual
            b, s = tokens.shape
            mask = tokens.ne(m.pad_token).long()
            # embeddings
            x = m.token_embedding(tokens) + \
                m.type_embedding(torch.zeros_like(tokens)) + \
                m.pos_embedding(m.pad_token + torch.cumsum(mask, dim=1) * mask)
            x = m.norm(x)
            x = m.dropout(x)

            # blocks
            for block in m.blocks[:-1]:
                x = block(x, mask.view(b, 1, 1, s))
            words = x

            sentence = m.blocks[-1](x, mask.view(b, 1, 1, s))
            mask = tokens.ne(m.pad_token).unsqueeze(-1).to(sentence)
            sentence = (sentence * mask).sum(dim=1) / mask.sum(dim=1)
            sentence = m.head(sentence)

            return {'crossattn': words, 'y': sentence}

        if not self.use_grad:
            with torch.no_grad():
                return fn()
        else:
            return fn()

    def dynamic_encode_text(self,
                            all_tokens,
                            tokenizer=None,
                            append_sentence_embedding=True):
        '''
                m: clip model
                t: tokenzer
                tokens: tensor(1, N)
            '''
        if tokenizer is None:
            tokenizer = self.tokenizer

        def fn():
            m = self.clip_model.textual
            ret_data = {'crossattn': [], 'y': []}
            for tokens_id in range(all_tokens.shape[0]):
                tokens = all_tokens[tokens_id]
                text_len = self.token_length
                device = tokens.device
                dtype = m.type_embedding.weight.dtype
                # special tokens
                sos_emb, eos_emb, pad_emb = m.token_embedding(
                    torch.LongTensor([
                        tokenizer.sos_token, tokenizer.eos_token,
                        tokenizer.pad_token
                    ]).to(device)).type(dtype).chunk(3)

                # get raw input tokens
                tokens = list(tokens.cpu().numpy())
                while tokens[-1] == tokenizer.pad_token:
                    tokens = tokens[:-1]
                tokens = tokens[1:-1]
                embeds = m.token_embedding(
                    torch.LongTensor(tokens).to(device)).type(dtype)

                # split into chunks to support any-length text
                chunk_embeds, chunk_tokens = [], []
                max_words = text_len - 2
                if len(tokens) == 0:
                    chunk = torch.cat([sos_emb, eos_emb])
                    chunk = torch.cat(
                        [chunk,
                         pad_emb.repeat(text_len - len(chunk), 1)])
                    chunk_embeds.append(chunk)

                    chunk = torch.LongTensor([tokenizer.sos_token] +
                                             [tokenizer.eos_token])
                    chunk = torch.cat([
                        chunk,
                        torch.LongTensor([tokenizer.pad_token] *
                                         (text_len - len(chunk)))
                    ])
                    chunk_tokens.append(chunk)
                else:
                    while len(tokens) > 0:
                        # find splitting position
                        if len(tokens) <= max_words:
                            pos = len(tokens)
                        else:
                            pos = [
                                i for i, u in enumerate(tokens[:max_words])
                                if u == tokenizer.comma_token
                            ]
                            pos = max_words if len(pos) == 0 else pos[-1] + 1

                        # collect chunk
                        chunk = torch.cat([sos_emb, embeds[:pos], eos_emb])
                        chunk = torch.cat(
                            [chunk,
                             pad_emb.repeat(text_len - len(chunk), 1)])
                        chunk_embeds.append(chunk)

                        chunk = torch.LongTensor([tokenizer.sos_token] +
                                                 tokens[:pos] +
                                                 [tokenizer.eos_token])
                        chunk = torch.cat([
                            chunk,
                            torch.LongTensor([tokenizer.pad_token] *
                                             (text_len - len(chunk)))
                        ])
                        chunk_tokens.append(chunk)

                        # update
                        tokens = tokens[pos:]
                        embeds = embeds[pos:]
                # loop over chunks
                words = []
                sentences = []
                for i, (chunk_token, chunk_embed) in enumerate(
                        zip(chunk_tokens, chunk_embeds)):
                    chunk_token = chunk_token.unsqueeze(0).to(device)
                    chunk_embed = chunk_embed.unsqueeze(0).to(device)
                    # embeddings
                    mask = chunk_token.ne(tokenizer.pad_token).long()
                    x = chunk_embed.type(dtype) + m.type_embedding(
                        torch.zeros_like(chunk_token)) + m.pos_embedding(
                            m.pad_token + torch.cumsum(mask, dim=1) * mask)

                    x = m.norm(x)
                    x = m.dropout(x)
                    blocks = m.blocks[:-(self.clip_skip - 1)]
                    for block in blocks:
                        x = block(x, mask.view(1, 1, 1, -1))
                    print('word', torch.sum(x))
                    words.append(x.clone())

                    # if append_sentence_embedding:
                    # last layers
                    blocks = m.blocks[-(self.clip_skip - 1):]
                    for block in blocks:
                        x = block(x, mask.view(1, 1, 1, -1))

                    # get global embedding
                    x = (x * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1)
                    x = m.head(x)
                    print('sentence', torch.sum(x))
                    # output
                    sentences.append(x.unsqueeze(0))
            sentence = torch.cat(sentences, dim=0).mean(dim=0)
            words = torch.cat(words, dim=1)
            ret_data['crossattn'] = words
            ret_data['y'] = sentence
            # ret_data['y'].append(sentence)
            # ret_data.append(torch.cat([sentence] + words, dim=1))
            # ret_data['crossattn'].append(torch.cat(words, dim=1))
            # ret_data['crossattn'] = torch.cat(ret_data['crossattn'], dim=0)
            # ret_data['y'] = torch.cat(ret_data['y'], dim=0)
            return ret_data

        if not self.use_grad:
            with torch.no_grad():
                return fn()
        else:
            return fn()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            ClipEncoder.para_dict,
                            set_name=True)
