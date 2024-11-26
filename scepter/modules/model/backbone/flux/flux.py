# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
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
        self.use_grad_checkpoint = cfg.get('USE_GRAD_CHECKPOINT', False)

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

        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            ) for _ in range(depth)
        ])

        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(self.hidden_size,
                              self.num_heads,
                              mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
        ])

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def prepare_input(self, x, context, y, x_shape=None):
        # x.shape [6, 16, 16, 16] target is [6, 16, 768, 1360]
        bs, c, h, w = x.shape
        x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=2, pw=2)
        x_id = torch.zeros(h // 2, w // 2, 3)
        x_id[..., 1] = x_id[..., 1] + torch.arange(h // 2)[:, None]
        x_id[..., 2] = x_id[..., 2] + torch.arange(w // 2)[None, :]
        x_ids = repeat(x_id, 'h w c -> b (h w) c', b=bs)
        txt_ids = torch.zeros(bs, context.shape[1], 3)
        return x, x_ids.to(x), context.to(x), txt_ids.to(x), y.to(x), h, w

    def unpack(self, x: Tensor, height: int, width: int) -> Tensor:
        return rearrange(
            x,
            'b (h w) (c ph pw) -> b c (h ph) (w pw)',
            h=math.ceil(height / 2),
            w=math.ceil(width / 2),
            ph=2,
            pw=2,
        )

    def load_pretrained_model(self, pretrained_model):
        if next(self.parameters()).device.type == 'meta':
            map_location = we.device_id
        else:
            map_location = 'cpu'
        if pretrained_model is not None:
            with FS.get_from(pretrained_model,
                             wait_finish=True) as local_model:
                if local_model.endswith('safetensors'):
                    from safetensors.torch import load_file as load_safetensors
                    sd = load_safetensors(local_model, device=map_location)
                else:
                    sd = torch.load(local_model, map_location=map_location)
            missing, unexpected = self.load_state_dict(sd,
                                                       strict=False,
                                                       assign=True)
            self.logger.info(
                f'Restored from {pretrained_model} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')  # noqa
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')  # noqa

    def forward(self,
                x: Tensor,
                t: Tensor,
                cond: dict = {},
                guidance: Tensor | None = None,
                gc_seg: int = 0) -> Tensor:
        x, x_ids, txt, txt_ids, y, h, w = self.prepare_input(
            x, cond['context'], cond['y'])
        # running on sequences img
        x = self.img_in(x)
        vec = self.time_in(timestep_embedding(t, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
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
                functions=[
                    partial(block, **kwargs) for block in self.double_blocks
                ],
                segments=gc_seg if gc_seg > 0 else len(self.double_blocks),
                input=x,
                use_reentrant=False)
        else:
            for block in self.double_blocks:
                x = block(x, **kwargs)

        kwargs = dict(
            vec=vec,
            pe=pe,
        )

        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[
                    partial(block, **kwargs) for block in self.single_blocks
                ],
                segments=gc_seg if gc_seg > 0 else len(self.single_blocks),
                input=x,
                use_reentrant=False)
        else:
            for block in self.single_blocks:
                x = block(x, **kwargs)
        x = x[:, txt.shape[1]:, ...]
        x = self.final_layer(
            x, vec)  # (N, T, patch_size ** 2 * out_channels) 6 64 64
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
        context, y = cond["context"].to(x), cond["y"].to(x)
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
        if self.guidance_embed:
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
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            FluxMR.para_dict,
                            set_name=True)
