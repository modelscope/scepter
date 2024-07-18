# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import Dict

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.utils.dlpack
from einops import rearrange
from scepter.modules.model.backbone.unet.unet_utils import Timestep
from scepter.modules.model.embedder.base_embedder import BaseEmbedder
from scepter.modules.model.embedder.resampler import Resampler
from scepter.modules.model.registry import EMBEDDERS
from scepter.modules.model.tokenizer.tokenizer_component import (
    basic_clean, canonicalize, heavy_clean, whitespace_clean)
from scepter.modules.model.utils.basic_utils import expand_dims_like
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from torch.utils.checkpoint import checkpoint

try:
    from transformers import (CLIPTextModel, CLIPTokenizer,
                              CLIPVisionModelWithProjection, AutoTokenizer,
                              T5EncoderModel, CLIPTextModelWithProjection)
except Exception as e:
    warnings.warn(
        f'Import transformers error, please deal with this problem: {e}')


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
                enabled=enabled,
                dtype=torch.get_autocast_gpu_dtype(),
                cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


@EMBEDDERS.register_class()
class FrozenCLIPEmbedder(BaseEmbedder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    para_dict = {
        'PRETRAINED_MODEL': {
            'value': None,
            'description': 'You should set pretrained_model: modelcard.'
        },
        'TOKENIZER_PATH': {
            'value':
            None,
            'description':
            'If you want to use tokenizer in embedder, you should set this field.'
        },
        'MAX_LENGTH': {
            'value': 77,
            'description': ''
        },
        'FREEZE': {
            'value': True,
            'description': ''
        },
        'USE_GRAD': {
            'value': False,
            'description': 'Compute grad or not.'
        },
        'LAYER': {
            'value': 'last',
            'description': ''
        },
        'LAYER_IDX': {
            'value': None,
            'description': ''
        },
        'USE_FINAL_LAYER_NORM': {
            'value': False,
            'description': ''
        },
    }
    LAYERS = ['last', 'pooled', 'hidden']

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        tokenizer_path = cfg.get('TOKENIZER_PATH', None)
        if tokenizer_path is not None:
            with FS.get_dir_to_local_dir(tokenizer_path,
                                         wait_finish=True) as local_path:
                self.tokenizer = CLIPTokenizer.from_pretrained(local_path)

        pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        if pretrained_model is None:
            raise 'You should set pretrained_model: modelcard.'
        with FS.get_dir_to_local_dir(cfg.PRETRAINED_MODEL,
                                     wait_finish=True) as local_path:
            self.transformer = CLIPTextModel.from_pretrained(local_path)

        self.use_grad = cfg.get('USE_GRAD', False)
        self.freeze_flag = cfg.get('FREEZE', True)
        if self.freeze_flag:
            self.freeze()

        self.max_length = cfg.get('MAX_LENGTH', 77)
        self.layer = cfg.get('LAYER', 'last')
        self.layer_idx = cfg.get('LAYER_IDX', None)
        self.use_final_layer_norm = cfg.get('USE_FINAL_LAYER_NORM', False)
        assert self.layer in self.LAYERS
        if self.layer == 'hidden':
            assert self.layer_idx is not None
            assert 0 <= abs(self.layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _forward(self, text):
        batch_encoding = self.tokenizer(text,
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_length=True,
                                        return_overflowing_tokens=False,
                                        padding='max_length',
                                        return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(we.device_id)
        outputs = self.transformer(input_ids=tokens,
                                   output_hidden_states=self.layer == 'hidden')
        if self.layer == 'last':
            z = outputs.last_hidden_state
        elif self.layer == 'pooled':
            z = outputs.pooler_output[:, None, :]
            if self.use_final_layer_norm:
                z = self.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.hidden_states[self.layer_idx]
            if self.use_final_layer_norm:
                z = self.transformer.text_model.final_layer_norm(z)
        return z

    @autocast
    def forward(self, text):
        if not self.use_grad:
            with torch.no_grad():
                output = self._forward(text)
        else:
            output = self._forward(text)
        return output

    def encode(self, text):
        return self(text)

    # @torch.no_grad()
    def _encode_text(self,
                     tokens,
                     tokenizer=None,
                     append_sentence_embedding=False):
        outputs = self.transformer(input_ids=tokens,
                                   output_hidden_states=self.layer == 'hidden')
        if self.layer == 'last':
            z = outputs.last_hidden_state
        elif self.layer == 'pooled':
            z = outputs.pooler_output[:, None, :]
            if self.use_final_layer_norm:
                z = self.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.hidden_states[self.layer_idx]
            if self.use_final_layer_norm:
                z = self.transformer.text_model.final_layer_norm(z)
        return z

    def encode_text(self,
                    tokens,
                    tokenizer=None,
                    append_sentence_embedding=False):
        if not self.use_grad:
            with torch.no_grad():
                output = self._encode_text(tokens, tokenizer,
                                           append_sentence_embedding)
        else:
            output = self._encode_text(tokens, tokenizer,
                                       append_sentence_embedding)
        return output

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            FrozenCLIPEmbedder.para_dict,
                            set_name=True)


@EMBEDDERS.register_class()
class FrozenOpenCLIPEmbedder(BaseEmbedder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    para_dict = {
        'ARCH': {
            'value': 'ViT-H-14',
            'description': ''
        },
        'PRETRAINED_MODEL': {
            'value': '',
            'description': ''
        },
        'MAX_LENGTH': {
            'value': 77,
            'description': ''
        },
        'FREEZE': {
            'value': True,
            'description': ''
        },
        'USE_GRAD': {
            'value': False,
            'description': 'Compute grad or not.'
        },
        'LAYER': {
            'value': 'last',
            'description': ''
        },
    }
    LAYERS = ['last', 'penultimate']

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

        arch = cfg.get('ARCH', 'ViT-H-14')
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=None)
        del model.visual
        if cfg.PRETRAINED_MODEL is not None:
            with FS.get_from(cfg.PRETRAINED_MODEL,
                             wait_finish=True) as local_path:
                model.load_state_dict(torch.load(local_path), strict=False)
        self.model = model

        self.use_grad = cfg.get('USE_GRAD', False)
        self.freeze_flag = cfg.get('FREEZE', True)
        if self.freeze_flag:
            self.freeze()

        self.max_length = cfg.get('MAX_LENGTH', 77)
        self.layer = cfg.get('LAYER', 'penultimate')
        assert self.layer in self.LAYERS
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            tokens = open_clip.tokenize(text)
            z = self.encode_with_transformer(tokens.to(we.device_id))
            return z

    def encode_with_transformer(self, text):
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            x = self.model.token_embedding(
                text)  # [batch_size, n_ctx, d_model]
            x = x + self.model.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_transformer_forward(x,
                                              attn_mask=self.model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_final(x)
            return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            for i, r in enumerate(self.model.transformer.resblocks):
                if i == len(self.model.transformer.resblocks) - self.layer_idx:
                    break
                if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting(
                ):
                    x = checkpoint(r, x, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x

    def encode_text(self,
                    tokens,
                    tokenizer=None,
                    append_sentence_embedding=False):
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            z = self.encode_with_transformer(tokens.to(we.device_id))
            return z

    def encode(self, text):
        return self(text)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            FrozenOpenCLIPEmbedder.para_dict,
                            set_name=True)


@EMBEDDERS.register_class()
class FrozenOpenCLIPEmbedder2(BaseEmbedder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    """
        Uses the OpenCLIP transformer encoder for text
        """
    para_dict = {
        'ARCH': {
            'value': 'ViT-H-14',
            'description': ''
        },
        'PRETRAINED_MODEL': {
            'value': 'laion2b_s32b_b79k',
            'description': ''
        },
        'MAX_LENGTH': {
            'value': 77,
            'description': ''
        },
        'FREEZE': {
            'value': True,
            'description': ''
        },
        'USE_GRAD': {
            'value': False,
            'description': 'Compute grad or not.'
        },
        'ALWAYS_RETURN_POOLED': {
            'value':
            False,
            'description':
            'Whether always return pooled results or not ,default False.'
        },
        'LEGACY': {
            'value':
            True,
            'description':
            'Whether use legacy returnd feature or not ,default True.'
        },
        'LAYER': {
            'value': 'last',
            'description': ''
        },
    }

    LAYERS = ['pooled', 'last', 'penultimate']

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        arch = cfg.get('ARCH', 'ViT-H-14')
        if cfg.get('PRETRAINED_MODEL', None) is None:
            model, _, _ = open_clip.create_model_and_transforms(
                arch, device=torch.device('cpu'), pretrained=None)
            del model.visual
        else:
            with FS.get_from(cfg.PRETRAINED_MODEL,
                             wait_finish=True) as local_path:
                model, _, _ = open_clip.create_model_and_transforms(
                    arch, device=torch.device('cpu'), pretrained=local_path)
                del model.visual
        self.model = model

        self.max_length = cfg.get('MAX_LENGTH', 77)
        self.layer = cfg.get('LAYER', 'last')
        self.return_pooled = cfg.get('ALWAYS_RETURN_POOLED', False)
        self.use_grad = cfg.get('USE_GRAD', False)
        self.freeze_flag = cfg.get('FREEZE', True)
        if self.freeze_flag:
            self.freeze()
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = cfg.get('LEGACY', True)

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            tokens = open_clip.tokenize(text)
            z = self.encode_with_transformer(tokens.to(we.device_id))
            if not self.return_pooled and self.legacy:
                return z
            if self.return_pooled:
                assert not self.legacy
                return z[self.layer], z['pooled']
            return z[self.layer]

    def encode_with_transformer(self, text):
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            x = self.model.token_embedding(
                text)  # [batch_size, n_ctx, d_model]
            x = x + self.model.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_transformer_forward(x,
                                              attn_mask=self.model.attn_mask)
            if self.legacy:
                x = x[self.layer]
                x = self.model.ln_final(x)
                return x
            else:
                # x is a dict and will stay a dict
                o = x['last']
                o = self.model.ln_final(o)
                pooled = self.pool(o, text)
                x['pooled'] = pooled
                return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (x[torch.arange(x.shape[0]),
               text.argmax(dim=-1)] @ self.model.text_projection)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            for i, r in enumerate(self.model.transformer.resblocks):
                if i == len(self.model.transformer.resblocks) - 1:
                    outputs['penultimate'] = x.permute(1, 0, 2)  # LND -> NLD
                if (self.model.transformer.grad_checkpointing
                        and not torch.jit.is_scripting()):
                    x = checkpoint(r, x, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            outputs['last'] = x.permute(1, 0, 2)  # LND -> NLD
            return outputs

    def encode(self, text):
        return self(text)

    def encode_text(self,
                    tokens,
                    tokenizer=None,
                    append_sentence_embedding=False):
        z = self.encode_with_transformer(tokens.to(we.device_id))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z['pooled']
        return z[self.layer]

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            FrozenOpenCLIPEmbedder2.para_dict,
                            set_name=True)


@EMBEDDERS.register_class()
class ConcatTimestepEmbedderND(BaseEmbedder):
    """embeds each dimension independently and concatenates them"""
    para_dict = {
        'OUT_DIM': {
            'value': 256,
            'description': 'Output dim'
        },
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg=cfg, logger=logger)
        outdim = cfg.get('OUT_DIM', 256)
        self.timestep = Timestep(outdim, legacy=True)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, 'b d -> (b d)')
        emb = self.timestep(x)
        emb = rearrange(emb,
                        '(b d) d2 -> b (d d2)',
                        b=b,
                        d=dims,
                        d2=self.outdim)
        return emb

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            ConcatTimestepEmbedderND.para_dict,
                            set_name=True)


@EMBEDDERS.register_class()
class IPAdapterPlusEmbedder(BaseEmbedder):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

        with FS.get_dir_to_local_dir(cfg.CLIP_DIR,
                                     wait_finish=True) as local_path:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                local_path)

        self.image_proj_model = Resampler(
            dim=self.cfg.get('IN_DIM', 768),
            depth=self.cfg.get('DEPTH', 4),
            dim_head=64,
            heads=self.cfg.get('HEADS', 12),
            num_queries=self.cfg.get('NUM_TOKENS', 16),
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.cfg.get('CROSSATTN_DIM', 768),
            ff_mult=4,
        )

        with FS.get_from(cfg.PRETRAINED_MODEL, wait_finish=True) as local_path:
            ckpt = torch.load(local_path, map_location='cpu')
            self.image_proj_model.load_state_dict(ckpt['image_proj'],
                                                  strict=True)

        self.patch_projector = nn.Linear(self.image_encoder.config.hidden_size,
                                         self.cfg.get('CROSSATTN_DIM', 768))

    def encode(self, ref_ip, ref_detail):
        encoder_output = self.image_encoder(ref_ip, output_hidden_states=True)
        image_prompt_embeds = self.image_proj_model(
            encoder_output.hidden_states[-2])
        encoder_output_2 = self.image_encoder(ref_detail,
                                              output_hidden_states=True)
        image_patch_embeds = self.patch_projector(
            encoder_output_2.last_hidden_state)
        out = {
            'img_crossattn': image_prompt_embeds,
            'ref_crossattn': image_patch_embeds,
        }
        return out

    def forward(self, ref_ip, ref_detail):
        return self.encode(ref_ip, ref_detail)


class RefCrossEmbedder(BaseEmbedder):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

        with FS.get_dir_to_local_dir(cfg.CLIP_DIR,
                                     wait_finish=True) as local_path:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                local_path)

        self.patch_projector = nn.Linear(self.image_encoder.config.hidden_size,
                                         self.cfg.get('CROSSATTN_DIM', 768))

    def encode(self, img):
        encoder_output = self.image_encoder(img, output_hidden_states=True)
        image_patch_embeds = self.patch_projector(
            encoder_output.last_hidden_state)
        out = {
            'ref_crossattn': image_patch_embeds,
        }
        return out

    def forward(self, img):
        return self.encode(img)


@EMBEDDERS.register_class()
class TransparentEmbedder(BaseEmbedder):
    def forward(self, *args):
        out = dict()
        for key, val in zip(self.input_keys, args):
            out[key] = val
        return out


@EMBEDDERS.register_class()
class NoiseConcatEmbedder(BaseEmbedder):
    def forward(self, *args):
        return {'concat': torch.cat(args, dim=1)}


@EMBEDDERS.register_class()
class GeneralConditioner(BaseEmbedder):
    OUTPUT_DIM2KEYS = {2: 'y', 3: 'crossattn', 4: 'concat', 5: 'concat'}
    KEY2CATDIM = {'y': 1, 'crossattn': 2, 'concat': 1}
    para_dict = {
        'EMBEDDERS': [],
        'USE_GRAD': {
            'value': False,
            'description': 'Compute grad or not.'
        },
    }
    para_dict.update(para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        emb_models = cfg.get('EMBEDDERS', [])
        use_grad = cfg.get('USE_GRAD', False)
        self.embedders = nn.ModuleList([])
        for n, embconfig in enumerate(emb_models):
            embconfig.USE_GRAD = use_grad if not embconfig.have(
                'USE_GRAD'
            ) or embconfig.USE_GRAD is None else embconfig.USE_GRAD
            embedder = EMBEDDERS.build(embconfig, logger=logger)
            embedder.ucg_rate = embconfig.get('UCG_RATE', 0.0)
            embedder.input_keys = embconfig.get('INPUT_KEYS', [])
            embedder.legacy_ucg_val = embconfig.get('LEGACY_UCG_VALUE', None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            self.embedders.append(embedder)

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model is not None:
            with FS.get_from(pretrained_model,
                             wait_finish=True) as local_model:
                self.init_from_ckpt(local_model)

    def init_from_ckpt(self, path, ignore_keys=list()):
        if path.endswith('safetensors'):
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(path)
        else:
            sd = torch.load(path, map_location='cpu')
        new_sd = OrderedDict()
        for k, v in sd.items():
            ignored = False
            for ik in ignore_keys:
                if ik in k:
                    if we.rank == 0:
                        self.logger.info(
                            'Ignore key {} from state_dict.'.format(k))
                    ignored = True
                    break
            if not ignored:
                new_sd[k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if we.rank == 0:
            self.logger.info(
                f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def possibly_get_ucg_val(self, embedder, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(self, batch: Dict, force_zero_embeddings=None) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if hasattr(
                embedder, 'use_grad') and embedder.use_grad else torch.no_grad
            with embedding_context():
                if hasattr(embedder, 'input_key') and (embedder.input_key
                                                       is not None):
                    if embedder.input_key not in batch:
                        continue
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, 'input_keys'):
                    if any([k not in batch for k in embedder.input_keys]):
                        continue
                    emb_out = embedder(
                        *[batch[k] for k in embedder.input_keys])

            if isinstance(emb_out, dict):
                for key, val in emb_out.items():
                    if key in output:
                        assert key in self.KEY2CATDIM
                        output[key] = torch.cat([output[key], val],
                                                dim=self.KEY2CATDIM[key])
                    else:
                        output[key] = val
            else:
                assert isinstance(
                    emb_out, (torch.Tensor, list, tuple)
                ), f'encoder outputs must be tensors or a sequence, but got {type(emb_out)}'

                if not isinstance(emb_out, (list, tuple)):
                    emb_out = [emb_out]

                for emb in emb_out:
                    out_key = self.OUTPUT_DIM2KEYS[emb.dim()]

                    if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                        emb = (expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate) *
                                torch.ones(emb.shape[0], device=emb.device)),
                            emb,
                        ) * emb)

                    if (hasattr(embedder, 'input_keys')):
                        if np.sum(
                                np.array([
                                    key in force_zero_embeddings
                                    for key in embedder.input_keys
                                ])) > 0:
                            emb = torch.zeros_like(emb)

                    if out_key in output:
                        output[out_key] = torch.cat((output[out_key], emb),
                                                    self.KEY2CATDIM[out_key])
                    else:
                        output[out_key] = emb

        return output

    def get_unconditional_conditioning(self,
                                       batch_c,
                                       batch_uc=None,
                                       force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc,
                  force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc

    def encode(self,
               batch_dict,
               is_unconditional=False,
               force_uc_zero_embeddings=None):
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        if is_unconditional:
            c = self(batch_dict, force_uc_zero_embeddings)
        else:
            c = self(batch_dict)
        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            GeneralConditioner.para_dict,
                            set_name=True)


@EMBEDDERS.register_class()
class T5EmbedderHF(BaseEmbedder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    """
        Uses the OpenCLIP transformer encoder for text
        """
    para_dict = {
        'PRETRAINED_MODEL': {
            'value':
            'google/umt5-small',
            'description':
            'Pretrained Model for umt5, modelcard path or local path.'
        },
        'TOKENIZER_PATH': {
            'value': 'google/umt5-small',
            'description':
            'Tokenizer Path for umt5, modelcard path or local path.'
        },
        'FREEZE': {
            'value': True,
            'description': ''
        },
        'USE_GRAD': {
            'value': False,
            'description': 'Compute grad or not.'
        },
        'CLEAN': {
            'value':
            'whitespace',
            'description':
            'Set the clean strtegy for tokenizer, used when TOKENIZER_PATH is not None.'
        },
        'LAYER': {
            'value': 'last',
            'description': ''
        },
        'LEGACY': {
            'value':
            True,
            'description':
            'Whether use legacy returnd feature or not ,default True.'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        pretrained_path = cfg.get('PRETRAINED_MODEL', None)
        t5_dtype = cfg.get('T5_DTYPE', None)
        assert pretrained_path
        with FS.get_dir_to_local_dir(pretrained_path,
                                     wait_finish=True) as local_path:
            if t5_dtype is not None:
                self.model = T5EncoderModel.from_pretrained(
                    local_path, torch_dtype=getattr(torch, t5_dtype))
            else:
                self.model = T5EncoderModel.from_pretrained(local_path)
        tokenizer_path = cfg.get('TOKENIZER_PATH', None)
        self.length = cfg.get('LENGTH', 77)
        if tokenizer_path:
            self.tokenize_kargs = {'return_tensors': 'pt'}
            with FS.get_dir_to_local_dir(tokenizer_path,
                                         wait_finish=True) as local_path:
                self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            if self.length is not None:
                self.tokenize_kargs.update({
                    'padding': 'max_length',
                    'truncation': True,
                    'max_length': self.length
                })
            self.eos_token = self.tokenizer(
                self.tokenizer.eos_token)['input_ids'][0]
        else:
            self.tokenizer = None
            self.tokenize_kargs = {}

        self.use_grad = cfg.get('USE_GRAD', False)
        self.clean = cfg.get('CLEAN', 'whitespace')

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    # encode && encode_text
    def forward(self, tokens, return_mask=False):
        # tokenization
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            x = self.model(tokens.input_ids.to(we.device_id),
                           tokens.attention_mask.to(we.device_id))
            x = x.last_hidden_state
            # if not self.return_pooled:
            #     return x.detach()
            # else:
            #     return x.detach(), self.pool(x, tokens.input_ids)
            if return_mask:
                return x.detach() + 0.0, tokens.attention_mask.to(we.device_id)
            else:
                return x.detach() + 0.0

    def pool(self, x, tokens):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        return x[torch.arange(x.shape[0]),
                 torch.argmax((tokens.input_ids == 1).float(), dim=-1)]

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        elif self.clean == 'heavy':
            text = heavy_clean(heavy_clean(text))
        return text

    def encode_text(self,
                    tokens,
                    tokenizer=None,
                    append_sentence_embedding=False,
                    return_mask=False):
        return self(tokens, return_mask=return_mask)

    def encode(self, text, return_mask=False):
        if isinstance(text, str):
            text = [text]
        if self.clean:
            text = [self._clean(u) for u in text]
        assert self.tokenizer is not None
        tokens = self.tokenizer(text, **self.tokenize_kargs)
        return self(tokens, return_mask=return_mask)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            T5EmbedderHF.para_dict,
                            set_name=True)


@EMBEDDERS.register_class()
class FrozenCLIPEmbedder2(FrozenCLIPEmbedder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    para_dict = {
        'RETURN_POOLED': {
            'value': False,
            'description':
            'Whether return pooled results or not, default False.'
        }
    }
    para_dict.update(FrozenCLIPEmbedder.para_dict)
    LAYERS = ['hidden', 'last', 'penultimate']

    def __init__(self, cfg, logger=None):
        super(FrozenCLIPEmbedder, self).__init__(cfg, logger=logger)
        self.return_pooled = cfg.get('RETURN_POOLED', False)
        tokenizer_path = cfg.get('TOKENIZER_PATH', None)
        if tokenizer_path is not None:
            with FS.get_dir_to_local_dir(tokenizer_path,
                                         wait_finish=True) as local_path:
                self.tokenizer = CLIPTokenizer.from_pretrained(local_path)

        pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        if pretrained_model is None:
            raise 'You should set pretrained_model: modelcard.'
        with FS.get_dir_to_local_dir(cfg.PRETRAINED_MODEL,
                                     wait_finish=True) as local_path:
            self.transformer = CLIPTextModelWithProjection.from_pretrained(
                local_path)

        self.use_grad = cfg.get('USE_GRAD', False)
        self.freeze_flag = cfg.get('FREEZE', True)
        if self.freeze_flag:
            self.freeze()

        self.max_length = cfg.get('MAX_LENGTH', 77)
        self.layer = cfg.get('LAYER', 'last')
        self.layer_idx = cfg.get('LAYER_IDX', None)
        self.use_final_layer_norm = cfg.get('USE_FINAL_LAYER_NORM', False)
        assert self.layer in self.LAYERS
        if self.layer == 'hidden':
            assert self.layer_idx is not None
            assert 0 <= abs(self.layer_idx) <= 12

    def _forward(self, text):
        batch_encoding = self.tokenizer(text,
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_length=True,
                                        return_overflowing_tokens=False,
                                        padding='max_length',
                                        return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(we.device_id)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=True)
        if self.layer == 'last':
            context = outputs.last_hidden_state
        elif self.layer == 'penultimate':
            context = outputs.hidden_states[-2]
        else:
            context = outputs.hidden_states[self.layer_idx]

        if self.return_pooled:
            pooled = outputs[0]
            return context, pooled
        return context


@EMBEDDERS.register_class()
class SD3TextEmbedder(BaseEmbedder):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

        clip_l_config = cfg.get('CLIP_L', None)
        clip_g_config = cfg.get('CLIP_G', None)
        t5_xxl_config = cfg.get('T5_XXL', None)

        self.clip_l = EMBEDDERS.build(clip_l_config) if clip_l_config else None
        self.clip_g = EMBEDDERS.build(clip_g_config) if clip_g_config else None
        self.t5_xxl = EMBEDDERS.build(t5_xxl_config) if t5_xxl_config else None

        self.p_zero = cfg.get('P_ZERO', 0.464)

    def encode(self, text):
        return self(text)

    def forward(self, text):
        l_ctx, g_ctx, t5_ctx = None, None, None
        n = len(text)
        l_pooled = torch.zeros((n, 768), device=we.device_id)
        g_pooled = torch.zeros((n, 1280), device=we.device_id)
        if self.clip_l:
            with torch.autocast(device_type='cuda',
                                enabled=True,
                                dtype=torch.float16):
                l_ctx, l_pooled = self.clip_l.encode(text)
        if self.clip_g:
            with torch.autocast(device_type='cuda',
                                enabled=True,
                                dtype=torch.float16):
                g_ctx, g_pooled = self.clip_g.encode(text)
        if self.t5_xxl:
            with torch.autocast(device_type='cuda',
                                enabled=True,
                                dtype=torch.float16):
                t5_ctx = self.t5_xxl.encode(text)

        pooled = torch.cat((l_pooled, g_pooled), dim=-1)

        if l_ctx is not None and g_ctx is not None:
            lg_ctx = torch.cat([l_ctx, g_ctx], dim=-1)
            lg_ctx = torch.nn.functional.pad(lg_ctx,
                                             (0, 4096 - lg_ctx.shape[-1]))
        elif l_ctx is not None:
            lg_ctx = torch.nn.functional.pad(l_ctx,
                                             (0, 4096 - l_ctx.shape[-1]))
        elif g_ctx is not None:
            lg_ctx = torch.nn.functional.pad(g_ctx, (768, 0))
            lg_ctx = torch.nn.functional.pad(lg_ctx,
                                             (0, 4096 - lg_ctx.shape[-1]))
        else:
            lg_ctx = None

        if t5_ctx is not None and lg_ctx is not None:
            ctx = torch.cat([lg_ctx, t5_ctx], dim=-2)
        elif t5_ctx is not None:
            ctx = t5_ctx
        elif lg_ctx is not None:
            ctx = lg_ctx
        else:
            ctx = torch.zeros((n, 77, 4096), device=we.device_id)

        return ctx, pooled


if __name__ == '__main__':
    import argparse
    from scepter.modules.utils.config import Config
    from scepter.modules.utils.logger import get_logger
    std_logger = get_logger(name='scepter')
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    cfg = Config(load=True, parser_ins=parser)

    for file_sys in cfg.FILE_SYSTEM:
        FS.init_fs_client(file_sys)
    model = SD3TextEmbedder(cfg.COND_STAGE_MODEL,
                            logger=std_logger).to(we.device_id)
    text = ['a dog is eating food.']
    ctx, pooled = model(text)
    print(ctx.shape, pooled.shape)
