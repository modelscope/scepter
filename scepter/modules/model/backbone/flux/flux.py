# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# This file contains code that is adapted from
# https://github.com/black-forest-labs/flux.git
import math
from collections import OrderedDict
from functools import partial

import torch
from einops import rearrange, repeat
from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.registry import BACKBONES
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn.utils.rnn import pad_sequence
from .layers import (DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder,
                     SingleStreamBlock, timestep_embedding)
@BACKBONES.register_class()
class Flux(BaseModel):
    """
    Transformer backbone Diffusion model with RoPE.
    """
    para_dict = {
        'IN_CHANNELS': {
            'value': 64,
            'description': "model's input channels."
        },
        'OUT_CHANNELS': {
            'value': 64,
            'description': "model's output channels."
        },
        'HIDDEN_SIZE': {
            'value': 1024,
            'description': "model's hidden size."
        },
        'NUM_HEADS': {
            'value': 16,
            'description': 'number of heads in the transformer.'
        },
        'AXES_DIM': {
            'value': [16, 56, 56],
            'description': 'dimensions of the axes of the positional encoding.'
        },
        'THETA': {
            'value': 10_000,
            'description': 'theta for positional encoding.'
        },
        'VEC_IN_DIM': {
            'value': 768,
            'description': 'dimension of the vector input.'
        },
        'GUIDANCE_EMBED': {
            'value': False,
            'description': 'whether to use guidance embedding.'
        },
        'CONTEXT_IN_DIM': {
            'value': 4096,
            'description': 'dimension of the context input.'
        },
        'MLP_RATIO': {
            'value': 4.0,
            'description': 'ratio of mlp hidden size to hidden size.'
        },
        'QKV_BIAS': {
            'value': True,
            'description': 'whether to use bias in qkv projection.'
        },
        'DEPTH': {
            'value': 19,
            'description': 'number of transformer blocks.'
        },
        'DEPTH_SINGLE_BLOCKS': {
            'value':
            38,
            'description':
            'number of transformer blocks in the single stream block.'
        },
        'USE_GRAD_CHECKPOINT': {
            'value': False,
            'description': 'whether to use gradient checkpointing.'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.in_channels = cfg.IN_CHANNELS
        self.out_channels = cfg.get('OUT_CHANNELS', self.in_channels)
        hidden_size = cfg.get('HIDDEN_SIZE', 1024)
        num_heads = cfg.get('NUM_HEADS', 16)
        axes_dim = cfg.AXES_DIM
        theta = cfg.THETA
        vec_in_dim = cfg.VEC_IN_DIM
        self.guidance_embed = cfg.GUIDANCE_EMBED
        context_in_dim = cfg.CONTEXT_IN_DIM
        mlp_ratio = cfg.MLP_RATIO
        qkv_bias = cfg.QKV_BIAS
        depth = cfg.DEPTH
        depth_single_blocks = cfg.DEPTH_SINGLE_BLOCKS
        self.use_grad_checkpoint = cfg.get("USE_GRAD_CHECKPOINT", False)
        self.attn_backend = cfg.get("ATTN_BACKEND", "pytorch")
        self.cache_pretrain_model = cfg.get("CACHE_PRETRAIN_MODEL", False)
        self.lora_model = cfg.get("DIFFUSERS_LORA_MODEL", None)
        self.comfyui_lora_model = cfg.get("COMFYUI_LORA_MODEL", None)
        self.swift_lora_model = cfg.get("SWIFT_LORA_MODEL", None)
        self.blackforest_lora_model = cfg.get("BLACKFOREST_LORA_MODEL", None)
        self.pretrain_adapter = cfg.get("PRETRAIN_ADAPTER", None)

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(
                f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = (MLPEmbedder(in_dim=256,
                                        hidden_dim=self.hidden_size)
                            if self.guidance_embed else nn.Identity())
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    backend=self.attn_backend
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio, backend=self.attn_backend)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def prepare_input(self, x, context, y, x_shape=None):
        # x.shape [6, 16, 16, 16] target is [6, 16, 768, 1360]
        bs, c, h, w = x.shape
        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        x_id = torch.zeros(h // 2, w // 2, 3)
        x_id[..., 1] = x_id[..., 1] + torch.arange(h // 2)[:, None]
        x_id[..., 2] = x_id[..., 2] + torch.arange(w // 2)[None, :]
        x_ids = repeat(x_id, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros(bs, context.shape[1], 3)
        return x, x_ids.to(x), context.to(x), txt_ids.to(x), y.to(x), h, w

    def unpack(self, x: Tensor, height: int, width: int) -> Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height/2),
            w=math.ceil(width/2),
            ph=2,
            pw=2,
        )

    def merge_diffuser_lora(self, ori_sd, lora_sd, scale=1.0):
        key_map = {
            "single_blocks.{}.linear1.weight": {"key_list": [
                ["transformer.single_transformer_blocks.{}.attn.to_q.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.attn.to_q.lora_B.weight", [0, 3072]],
                ["transformer.single_transformer_blocks.{}.attn.to_k.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.attn.to_k.lora_B.weight", [3072, 6144]],
                ["transformer.single_transformer_blocks.{}.attn.to_v.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.attn.to_v.lora_B.weight", [6144, 9216]],
                ["transformer.single_transformer_blocks.{}.proj_mlp.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.proj_mlp.lora_B.weight", [9216, 21504]]
            ], "num": 38},
            "single_blocks.{}.modulation.lin.weight": {"key_list": [
                ["transformer.single_transformer_blocks.{}.norm.linear.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.norm.linear.lora_B.weight", [0, 9216]],
            ], "num": 38},
            "single_blocks.{}.linear2.weight": {"key_list": [
                ["transformer.single_transformer_blocks.{}.proj_out.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.proj_out.lora_B.weight", [0, 3072]],
            ], "num": 38},
            "double_blocks.{}.txt_attn.qkv.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.add_q_proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.add_q_proj.lora_B.weight", [0, 3072]],
                ["transformer.transformer_blocks.{}.attn.add_k_proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.add_k_proj.lora_B.weight", [3072, 6144]],
                ["transformer.transformer_blocks.{}.attn.add_v_proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.add_v_proj.lora_B.weight", [6144, 9216]],
            ], "num": 19},
            "double_blocks.{}.img_attn.qkv.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.to_q.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_q.lora_B.weight", [0, 3072]],
                ["transformer.transformer_blocks.{}.attn.to_k.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_k.lora_B.weight", [3072, 6144]],
                ["transformer.transformer_blocks.{}.attn.to_v.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_v.lora_B.weight", [6144, 9216]],
            ], "num": 19},
            "double_blocks.{}.img_attn.proj.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.to_out.0.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_out.0.lora_B.weight", [0, 3072]]
            ], "num": 19},
            "double_blocks.{}.txt_attn.proj.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.to_add_out.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_add_out.lora_B.weight", [0, 3072]]
            ], "num": 19},
            "double_blocks.{}.img_mlp.0.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff.net.0.proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff.net.0.proj.lora_B.weight", [0, 12288]]
            ], "num": 19},
            "double_blocks.{}.img_mlp.2.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff.net.2.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff.net.2.lora_B.weight", [0, 3072]]
            ], "num": 19},
            "double_blocks.{}.txt_mlp.0.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff_context.net.0.proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff_context.net.0.proj.lora_B.weight", [0, 12288]]
            ], "num": 19},
            "double_blocks.{}.txt_mlp.2.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff_context.net.2.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff_context.net.2.lora_B.weight", [0, 3072]]
            ], "num": 19},
            "double_blocks.{}.img_mod.lin.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.norm1.linear.lora_A.weight",
                 "transformer.transformer_blocks.{}.norm1.linear.lora_B.weight", [0, 18432]]
            ], "num": 19},
            "double_blocks.{}.txt_mod.lin.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.norm1_context.linear.lora_A.weight",
                 "transformer.transformer_blocks.{}.norm1_context.linear.lora_B.weight", [0, 18432]]
            ], "num": 19}
        }
        cover_lora_keys = set()
        cover_ori_keys = set()
        for k, v in key_map.items():
            key_list = v["key_list"]
            block_num = v["num"]
            for block_id in range(block_num):
                for k_list in key_list:
                    if k_list[0].format(block_id) in lora_sd and k_list[1].format(block_id) in lora_sd:
                        cover_lora_keys.add(k_list[0].format(block_id))
                        cover_lora_keys.add(k_list[1].format(block_id))
                        current_weight = torch.matmul(lora_sd[k_list[0].format(block_id)].permute(1, 0),
                                                      lora_sd[k_list[1].format(block_id)].permute(1, 0)).permute(1, 0)
                        ori_sd[k.format(block_id)][k_list[2][0]:k_list[2][1], ...] += scale * current_weight
                        cover_ori_keys.add(k.format(block_id))
                        # lora_sd.pop(k_list[0].format(block_id))
                        # lora_sd.pop(k_list[1].format(block_id))
        self.logger.info(f"merge_blackforest_lora loads lora'parameters lora-paras: \n"
                         f"cover-{len(cover_lora_keys)} vs total {len(lora_sd)} \n"
                         f"cover ori-{len(cover_ori_keys)} vs total {len(ori_sd)}")
        return ori_sd

    def merge_swift_lora(self, ori_sd, lora_sd, scale = 1.0):
        have_lora_keys = {}
        for k, v in lora_sd.items():
            k = k[len("model."):] if k.startswith("model.") else k
            ori_key = k.split("lora")[0] + "weight"
            if ori_key not in ori_sd:
                raise f"{ori_key} should in the original statedict"
            if ori_key not in have_lora_keys:
                have_lora_keys[ori_key] = {}
            if "lora_A" in k:
                have_lora_keys[ori_key]["lora_A"] = v
            elif "lora_B" in k:
                have_lora_keys[ori_key]["lora_B"] = v
            else:
                raise NotImplementedError
        self.logger.info(f"merge_swift_lora loads lora'parameters {len(have_lora_keys)}")
        for key, v in have_lora_keys.items():
            current_weight = torch.matmul(v["lora_A"].permute(1, 0), v["lora_B"].permute(1, 0)).permute(1, 0)
            ori_sd[key] += scale * current_weight
        return ori_sd


    def merge_blackforest_lora(self, ori_sd, lora_sd, scale = 1.0):
        have_lora_keys = {}
        cover_lora_keys = set()
        cover_ori_keys = set()
        for k, v in lora_sd.items():
            if "lora" in k:
                ori_key = k.split("lora")[0] + "weight"
                if ori_key not in ori_sd:
                    raise f"{ori_key} should in the original statedict"
                if ori_key not in have_lora_keys:
                    have_lora_keys[ori_key] = {}
                if "lora_A" in k:
                    have_lora_keys[ori_key]["lora_A"] = v
                    cover_lora_keys.add(k)
                    cover_ori_keys.add(ori_key)
                elif "lora_B" in k:
                    have_lora_keys[ori_key]["lora_B"] = v
                    cover_lora_keys.add(k)
                    cover_ori_keys.add(ori_key)
            else:
                if k in ori_sd:
                    ori_sd[k] = v
                    cover_lora_keys.add(k)
                    cover_ori_keys.add(k)
                else:
                    print("unsurpport keys: ", k)
        self.logger.info(f"merge_blackforest_lora loads lora'parameters lora-paras: \n"
                         f"cover-{len(cover_lora_keys)} vs total {len(lora_sd)} \n"
                         f"cover ori-{len(cover_ori_keys)} vs total {len(ori_sd)}")

        for key, v in have_lora_keys.items():
            current_weight = torch.matmul(v["lora_A"].permute(1, 0), v["lora_B"].permute(1, 0)).permute(1, 0)
            # print(key, ori_sd[key].shape, current_weight.shape)
            ori_sd[key] += scale * current_weight
        return ori_sd

    def merge_comfyui_lora(self, ori_sd, lora_sd, scale = 1.0):
        ori_key_map = {key.replace("_", ".") : key for key in ori_sd.keys()}
        parse_ckpt = OrderedDict()
        for k, v in lora_sd.items():
            if "alpha" in k:
                continue
            k = k.replace("lora_unet_", "").replace("_", ".")
            map_k = ori_key_map[k.split(".lora")[0] + ".weight"]
            if map_k not in parse_ckpt:
                parse_ckpt[map_k] = {}
            if "lora.up" in k:
                parse_ckpt[map_k]["lora_up"] = v
            elif "lora.down" in k:
                parse_ckpt[map_k]["lora_down"] = v
        if self.cache_pretrain_model:
            self.lora_dict[self.comfyui_lora_model] = {}

        for key, v in parse_ckpt.items():
            current_weight = torch.matmul(v["lora_down"].permute(1, 0), v["lora_up"].permute(1, 0)).permute(1, 0)
            self.lora_dict[self.comfyui_lora_model] = current_weight
            ori_sd[key] += scale * current_weight
        return ori_sd

    def easy_lora_merge(self, ori_sd, lora_sd, scale = 1.0):
        for key, v in lora_sd.items():
            ori_sd[key] += scale * v
        return ori_sd

    def load_pretrained_model(self, pretrained_model, lora_scale = 1.0):
        if next(self.parameters()).device.type == 'meta':
            map_location = torch.device(we.device_id)
            safe_device = we.device_id
        else:
            map_location = "cpu"
            safe_device = "cpu"

        if pretrained_model is not None:
            if not hasattr(self, "ckpt"):
                with FS.get_from(pretrained_model, wait_finish=True) as local_model:
                    if local_model.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        ckpt = load_safetensors(local_model, device=safe_device)
                    else:
                        ckpt = torch.load(local_model, map_location=map_location, weights_only=True)
                if "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                if "model" in ckpt:
                    ckpt = ckpt["model"]["model"]
                if self.cache_pretrain_model:
                    self.ckpt = ckpt
                    self.lora_dict = {}
            else:
                ckpt = self.ckpt

            new_ckpt = OrderedDict()
            for k, v in ckpt.items():
                if k in ("img_in.weight"):
                    model_p = self.state_dict()[k]
                    if v.shape != model_p.shape:
                        expanded_state_dict_weight = torch.zeros_like(model_p, device=v.device)
                        slices = tuple(slice(0, dim) for dim in v.shape)
                        expanded_state_dict_weight[slices] = v
                        new_ckpt[k] = expanded_state_dict_weight
                    else:
                        new_ckpt[k] = v
                else:
                    new_ckpt[k] = v


            if self.lora_model is not None:
                with FS.get_from(self.lora_model, wait_finish=True) as local_model:
                    if local_model.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        lora_sd = load_safetensors(local_model, device=safe_device)
                    else:
                        lora_sd = torch.load(local_model, map_location=map_location, weights_only=True)
                new_ckpt = self.merge_diffuser_lora(new_ckpt, lora_sd, scale=lora_scale)
            if self.swift_lora_model is not None:
                if not isinstance(self.swift_lora_model, list):
                    self.swift_lora_model = [(self.swift_lora_model, 1.0)]
                for lora_model in self.swift_lora_model:
                    if isinstance(lora_model, str):
                        lora_model = (lora_model, 1.0/len(self.swift_lora_model))
                    print(lora_model)
                    self.logger.info(f"load swift lora model: {lora_model}")
                    with FS.get_from(lora_model[0], wait_finish=True) as local_model:
                        if local_model.endswith('safetensors'):
                            from safetensors.torch import load_file as load_safetensors
                            lora_sd = load_safetensors(local_model, device=safe_device)
                        else:
                            lora_sd = torch.load(local_model, map_location=map_location, weights_only=True)
                    new_ckpt = self.merge_swift_lora(new_ckpt, lora_sd, scale=lora_model[1])

            if self.blackforest_lora_model is not None:
                with FS.get_from(self.blackforest_lora_model, wait_finish=True) as local_model:
                    if local_model.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        lora_sd = load_safetensors(local_model, device=safe_device)
                    else:
                        lora_sd = torch.load(local_model, map_location=map_location, weights_only=True)
                new_ckpt = self.merge_blackforest_lora(new_ckpt, lora_sd, scale=lora_scale)

            if self.comfyui_lora_model is not None:
                if hasattr(self, "current_lora") and self.current_lora == self.comfyui_lora_model:
                    return
                if hasattr(self, "lora_dict") and self.comfyui_lora_model in self.lora_dict:
                    new_ckpt = self.easy_lora_merge(new_ckpt, self.lora_dict[self.comfyui_lora_model], scale=lora_scale)
                else:
                    with FS.get_from(self.comfyui_lora_model, wait_finish=True) as local_model:
                        if local_model.endswith('safetensors'):
                            from safetensors.torch import load_file as load_safetensors
                            lora_sd = load_safetensors(local_model, device=safe_device)
                        else:
                            lora_sd = torch.load(local_model, map_location=map_location, weights_only=True)
                    new_ckpt = self.merge_comfyui_lora(new_ckpt, lora_sd, scale=lora_scale)
                if self.comfyui_lora_model:
                    self.current_lora = self.comfyui_lora_model


            adapter_ckpt = {}
            if self.pretrain_adapter is not None:
                with FS.get_from(self.pretrain_adapter, wait_finish=True) as local_adapter:
                    if local_adapter.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        adapter_ckpt = load_safetensors(local_adapter, device=safe_device)
                    else:
                        adapter_ckpt = torch.load(local_adapter, map_location=map_location, weights_only=True)
            new_ckpt.update(adapter_ckpt)

            missing, unexpected = self.load_state_dict(new_ckpt, strict=False, assign=True)
            self.logger.info(
                f'Restored from {pretrained_model} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond: dict = {},
        guidance: Tensor | None = None,
        gc_seg: int = 0
    ) -> Tensor:
        x, x_ids, txt, txt_ids, y, h, w = self.prepare_input(x, cond["context"], cond["y"])
        # running on sequences img
        x = self.img_in(x)
        vec = self.time_in(timestep_embedding(t, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, x_ids), dim=1)
        pe = self.pe_embedder(ids)
        kwargs = dict(
            vec=vec,
            pe=pe,
            txt_length=txt.shape[1],
        )
        x = torch.cat((txt, x), 1)
        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.double_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.double_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.double_blocks:
                x = block(x, **kwargs)

        kwargs = dict(
            vec=vec,
            pe=pe,
        )

        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.single_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.single_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.single_blocks:
                x = block(x, **kwargs)
        x = x[:, txt.shape[1] :, ...]
        x = self.final_layer(x, vec)  # (N, T, patch_size ** 2 * out_channels) 6 64 64
        x = self.unpack(x, h, w)
        return x

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            Flux.para_dict,
                            set_name=True)
@BACKBONES.register_class()
class FluxMR(Flux):
    def prepare_input(self, x, cond):
        if isinstance(cond['context'], list):
            context, y = torch.cat(cond["context"], dim=0).to(x), torch.cat(cond["y"], dim=0).to(x)
        else:
            context, y = cond['context'].to(x), cond['y'].to(x)
        batch_frames, batch_frames_ids = [], []
        for ix, shape in zip(x, cond["x_shapes"]):
            # unpack image from sequence
            ix = ix[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            c, h, w = ix.shape
            ix = rearrange(ix, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            ix_id = torch.zeros(h // 2, w // 2, 3)
            ix_id[..., 1] = ix_id[..., 1] + torch.arange(h // 2)[:, None]
            ix_id[..., 2] = ix_id[..., 2] + torch.arange(w // 2)[None, :]
            ix_id = rearrange(ix_id, "h w c -> (h w) c")
            batch_frames.append([ix])
            batch_frames_ids.append([ix_id])

        x_list, x_id_list, mask_x_list, x_seq_length = [], [], [], []
        for frames, frame_ids in zip(batch_frames, batch_frames_ids):
            proj_frames = []
            for idx, one_frame in enumerate(frames):
                one_frame = self.img_in(one_frame)
                proj_frames.append(one_frame)
            ix = torch.cat(proj_frames, dim=0)
            if_id = torch.cat(frame_ids, dim=0)
            x_list.append(ix)
            x_id_list.append(if_id)
            mask_x_list.append(torch.ones(ix.shape[0]).to(ix.device, non_blocking=True).bool())
            x_seq_length.append(ix.shape[0])
        x = pad_sequence(tuple(x_list), batch_first=True)
        x_ids = pad_sequence(tuple(x_id_list), batch_first=True).to(x)  # [b,pad_seq,2] pad (0.,0.) at dim2
        mask_x = pad_sequence(tuple(mask_x_list), batch_first=True)

        txt = self.txt_in(context)
        txt_ids = torch.zeros(context.shape[0], context.shape[1], 3).to(x)
        mask_txt = torch.ones(context.shape[0], context.shape[1]).to(x.device, non_blocking=True).bool()

        return x, x_ids, txt, txt_ids, y, mask_x, mask_txt, x_seq_length

    def unpack(self, x: Tensor, cond: dict = None, x_seq_length: list = None) -> Tensor:
        x_list = []
        image_shapes = cond["x_shapes"]
        for u, shape, seq_length in zip(x, image_shapes, x_seq_length):
            height, width = shape
            h, w = math.ceil(height / 2), math.ceil(width / 2)
            u = rearrange(
                u[seq_length-h*w:seq_length, ...],
                "(h w) (c ph pw) -> (h ph w pw) c",
                h=h,
                w=w,
                ph=2,
                pw=2,
            )
            x_list.append(u)
        x = pad_sequence(tuple(x_list), batch_first=True).permute(0, 2, 1)
        return x

    def forward(
            self,
            x: Tensor,
            t: Tensor,
            cond: dict = {},
            guidance: Tensor | None = None,
            gc_seg: int = 0,
            **kwargs
    ) -> Tensor:
        x, x_ids, txt, txt_ids, y, mask_x, mask_txt, seq_length_list = self.prepare_input(x, cond)
        # running on sequences img
        vec = self.time_in(timestep_embedding(t, 256))
        if self.guidance_embed and guidance[-1] >= 0:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        ids = torch.cat((txt_ids, x_ids), dim=1)
        pe = self.pe_embedder(ids)

        mask_aside = torch.cat((mask_txt, mask_x), dim=1)
        mask = mask_aside[:, None, :] * mask_aside[:, :, None]

        kwargs = dict(
            vec=vec,
            pe=pe,
            mask=mask,
            txt_length = txt.shape[1],
        )
        x = torch.cat((txt, x), 1)
        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.double_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.double_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.double_blocks:
                x = block(x, **kwargs)

        kwargs = dict(
            vec=vec,
            pe=pe,
            mask=mask,
        )

        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.single_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.single_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.single_blocks:
                x = block(x, **kwargs)
        x = x[:, txt.shape[1]:, ...]
        x = self.final_layer(x, vec)  # (N, T, patch_size ** 2 * out_channels) 6 64 64
        x = self.unpack(x, cond, seq_length_list)
        return x

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            FluxMR.para_dict,
                            set_name=True)
@BACKBONES.register_class()
class FluxMRFill(FluxMR):
    def __init__(self, cfg, logger = None):
        super().__init__(cfg, logger)
    def prepare_input(self, x, cond):
        context, y = cond["context"], cond["y"]
        batch_frames, batch_frames_ids = [], []
        for ix, shape, imask, ie, ie_mask in zip(x, cond["x_shapes"], cond["x_mask"],
                                                 cond["edit"], cond["edit_mask"]):
            # unpack image from sequence
            ix = ix[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            imask = torch.ones_like(ix[[0], :, :]) if imask is None else imask.squeeze(0)
            if len(ie) > 0:
                ie = ie[0].squeeze(0)
                ie_mask = torch.ones((ix.shape[0] * 4, ix.shape[1], ix.shape[2])) if ie_mask is None else ie_mask[0].squeeze(0)
            else:
                ie, ie_mask = torch.zeros_like(ix).to(x), torch.ones_like(imask).to(x)
            ix = torch.cat([ix, ie, ie_mask], dim=0)
            c, h, w = ix.shape
            ix = rearrange(ix, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            ix_id = torch.zeros(h // 2, w // 2, 3)
            ix_id[..., 1] = ix_id[..., 1] + torch.arange(h // 2)[:, None]
            ix_id[..., 2] = ix_id[..., 2] + torch.arange(w // 2)[None, :]
            ix_id = rearrange(ix_id, "h w c -> (h w) c")
            batch_frames.append([ix])
            batch_frames_ids.append([ix_id])
        x_list, x_id_list, mask_x_list, x_seq_length = [], [], [], []
        for frames, frame_ids in zip(batch_frames, batch_frames_ids):
            proj_frames = []
            for idx, one_frame in enumerate(frames):
                one_frame = self.img_in(one_frame)
                proj_frames.append(one_frame)
            ix = torch.cat(proj_frames, dim=0)
            if_id = torch.cat(frame_ids, dim=0)
            x_list.append(ix)
            x_id_list.append(if_id)
            mask_x_list.append(torch.ones(ix.shape[0]).to(ix.device, non_blocking=True).bool())
            x_seq_length.append(ix.shape[0])
        # if len(x_list) < 1: import pdb;pdb.set_trace()
        x = pad_sequence(tuple(x_list), batch_first=True)
        x_ids = pad_sequence(tuple(x_id_list), batch_first=True).to(x)  # [b,pad_seq,2] pad (0.,0.) at dim2
        mask_x = pad_sequence(tuple(mask_x_list), batch_first=True)
        # import pdb;pdb.set_trace()
        if isinstance(context, list):
            txt_list, mask_txt_list, y_list = [], [], []
            for sample_id, (ctx, yy) in enumerate(zip(context, y)):
                txt_list.append(self.txt_in(ctx.to(x)))
                mask_txt_list.append(torch.ones(txt_list[-1].shape[0]).to(ctx.device, non_blocking=True).bool())
                y_list.append(yy.to(x))
            txt = pad_sequence(tuple(txt_list), batch_first=True)
            txt_ids = torch.zeros(txt.shape[0], txt.shape[1], 3).to(x)
            mask_txt = pad_sequence(tuple(mask_txt_list), batch_first=True)
            y = torch.cat(y_list, dim=0)
            assert y.ndim == 2 and txt.ndim == 3
        else:
            txt = self.txt_in(context)
            txt_ids = torch.zeros(context.shape[0], context.shape[1], 3).to(x)
            mask_txt = torch.ones(context.shape[0], context.shape[1]).to(x.device, non_blocking=True).bool()
        return x, x_ids, txt, txt_ids, y, mask_x, mask_txt, x_seq_length

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            FluxMRFill.para_dict,
                            set_name=True)
@BACKBONES.register_class()
class FluxMRRedux(FluxMR):
    '''
        ref_image_siglip + projector
    '''
    def __init__(self, cfg, logger = None):
        super().__init__(cfg, logger)
        self.redux_dim = cfg.get("REDUX_DIM", 1152)
        self.context_in_dim = cfg.CONTEXT_IN_DIM
        self.redux_up = nn.Linear(self.redux_dim, self.context_in_dim * 3)
        self.redux_down = nn.Linear(self.context_in_dim * 3, self.context_in_dim)


    def prepare_input(self, x, cond):
        ref_x = cond.get("ref_x", None)
        context, y = torch.cat(cond["context"], dim=0).to(x), torch.cat(cond["y"], dim=0).to(x)
        if ref_x is not None:
            ref_x = [torch.cat(ref_ix, dim=0).mean(dim=0, keepdim=True) for ref_ix in ref_x]
            ref_x = self.redux_down(nn.functional.silu(self.redux_up(torch.cat(ref_x, dim=0))))
            context = torch.cat((context, ref_x), dim=-2)

        batch_frames, batch_frames_ids = [], []
        for ix, shape in zip(x, cond["x_shapes"]):
            # unpack image from sequence
            ix = ix[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            c, h, w = ix.shape
            ix = rearrange(ix, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            ix_id = torch.zeros(h // 2, w // 2, 3)
            ix_id[..., 1] = ix_id[..., 1] + torch.arange(h // 2)[:, None]
            ix_id[..., 2] = ix_id[..., 2] + torch.arange(w // 2)[None, :]
            ix_id = rearrange(ix_id, "h w c -> (h w) c")
            batch_frames.append([ix])
            batch_frames_ids.append([ix_id])

        x_list, x_id_list, mask_x_list, x_seq_length = [], [], [], []
        for frames, frame_ids in zip(batch_frames, batch_frames_ids):
            proj_frames = []
            for idx, one_frame in enumerate(frames):
                one_frame = self.img_in(one_frame)
                proj_frames.append(one_frame)
            ix = torch.cat(proj_frames, dim=0)
            if_id = torch.cat(frame_ids, dim=0)
            x_list.append(ix)
            x_id_list.append(if_id)
            mask_x_list.append(torch.ones(ix.shape[0]).to(ix.device, non_blocking=True).bool())
            x_seq_length.append(ix.shape[0])
        x = pad_sequence(tuple(x_list), batch_first=True)
        x_ids = pad_sequence(tuple(x_id_list), batch_first=True).to(x)  # [b,pad_seq,2] pad (0.,0.) at dim2
        mask_x = pad_sequence(tuple(mask_x_list), batch_first=True)

        txt = self.txt_in(context)
        txt_ids = torch.zeros(context.shape[0], context.shape[1], 3).to(x)
        mask_txt = torch.ones(context.shape[0], context.shape[1]).to(x.device, non_blocking=True).bool()

        return x, x_ids, txt, txt_ids, y, mask_x, mask_txt, x_seq_length
    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            FluxMRRedux.para_dict,
                            set_name=True)
@BACKBONES.register_class()
class FluxMRControl(FluxMR):
    '''
        cat([x, ie]) ensure the same size bettwn the x and ie
    '''
    def prepare_input(self, x, cond, *args, **kwargs ):
        context, y = torch.cat(cond["context"], dim=0).to(x), torch.cat(cond["y"], dim=0).to(x)
        x_list, x_id_list, mask_x_list, x_seq_length = [], [], [], []
        for ix, shape, ie in zip(x, cond["x_shapes"], cond["edit"]):
            ix = ix[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            ix = torch.cat([ix, ie], dim=0)
            c, h, w = ix.shape
            ix = rearrange(ix, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            ix_id = torch.zeros(h // 2, w // 2, 3)
            ix_id[..., 1] = ix_id[..., 1] + torch.arange(h // 2)[:, None]
            ix_id[..., 2] = ix_id[..., 2] + torch.arange(w // 2)[None, :]
            ix_id = rearrange(ix_id, "h w c -> (h w) c")
            x_list.append(self.img_in(ix))
            x_id_list.append(ix_id)
            mask_x_list.append(torch.ones(ix.shape[0]).to(ix.device, non_blocking=True).bool())
            x_seq_length.append(ix.shape[0])
        # if len(x_list) < 1: import pdb;pdb.set_trace()
        x = pad_sequence(tuple(x_list), batch_first=True)
        x_ids = pad_sequence(tuple(x_id_list), batch_first=True).to(x)  # [b,pad_seq,2] pad (0.,0.) at dim2
        mask_x = pad_sequence(tuple(mask_x_list), batch_first=True)
        txt = self.txt_in(context)
        txt_ids = torch.zeros(context.shape[0], context.shape[1], 3).to(x)
        mask_txt = torch.ones(context.shape[0], context.shape[1]).to(x.device, non_blocking=True).bool()
        return x, x_ids, txt, txt_ids, y, mask_x, mask_txt, x_seq_length

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            FluxMRControl.para_dict,
                            set_name=True)
