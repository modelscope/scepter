# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange
from scepter.modules.model.backbone.flux import FluxMR
from scepter.modules.model.registry import BACKBONES
from scepter.modules.utils.config import dict_to_yaml


@BACKBONES.register_class()
class FluxMRACEPlus(FluxMR):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)

    def prepare_input(self, x, cond):
        context, y = cond['context'], cond['y']
        batch_frames, batch_frames_ids = [], []
        for ix, shape, imask, ie, ie_mask in zip(x, cond['x_shapes'],
                                                 cond['x_mask'], cond['edit'],
                                                 cond['edit_mask']):
            # unpack image from sequence
            ix = ix[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            imask = torch.ones_like(
                ix[[0], :, :]) if imask is None else imask.squeeze(0)
            if len(ie) > 0:
                ie = [iie.squeeze(0) for iie in ie]
                ie_mask = [
                    torch.ones(
                        (ix.shape[0] * 4, ix.shape[1],
                         ix.shape[2])) if iime is None else iime.squeeze(0)
                    for iime in ie_mask
                ]
                ie = torch.cat(ie, dim=-1)
                ie_mask = torch.cat(ie_mask, dim=-1)
            else:
                ie, ie_mask = torch.zeros_like(ix).to(x), torch.ones_like(
                    imask).to(x)
            ix = torch.cat([ix, ie, ie_mask], dim=0)
            c, h, w = ix.shape
            ix = rearrange(ix,
                           'c (h ph) (w pw) -> (h w) (c ph pw)',
                           ph=2,
                           pw=2)
            ix_id = torch.zeros(h // 2, w // 2, 3)
            ix_id[..., 1] = ix_id[..., 1] + torch.arange(h // 2)[:, None]
            ix_id[..., 2] = ix_id[..., 2] + torch.arange(w // 2)[None, :]
            ix_id = rearrange(ix_id, 'h w c -> (h w) c')
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
            mask_x_list.append(
                torch.ones(ix.shape[0]).to(ix.device,
                                           non_blocking=True).bool())
            x_seq_length.append(ix.shape[0])
        # if len(x_list) < 1: import pdb;pdb.set_trace()
        x = pad_sequence(tuple(x_list), batch_first=True)
        x_ids = pad_sequence(tuple(x_id_list), batch_first=True).to(
            x)  # [b,pad_seq,2] pad (0.,0.) at dim2
        mask_x = pad_sequence(tuple(mask_x_list), batch_first=True)
        # import pdb;pdb.set_trace()
        if isinstance(context, list):
            txt_list, mask_txt_list, y_list = [], [], []
            for sample_id, (ctx, yy) in enumerate(zip(context, y)):
                txt_list.append(self.txt_in(ctx.to(x)))
                mask_txt_list.append(
                    torch.ones(txt_list[-1].shape[0]).to(
                        ctx.device, non_blocking=True).bool())
                y_list.append(yy.to(x))
            txt = pad_sequence(tuple(txt_list), batch_first=True)
            txt_ids = torch.zeros(txt.shape[0], txt.shape[1], 3).to(x)
            mask_txt = pad_sequence(tuple(mask_txt_list), batch_first=True)
            y = torch.cat(y_list, dim=0)
            assert y.ndim == 2 and txt.ndim == 3
        else:
            txt = self.txt_in(context)
            txt_ids = torch.zeros(context.shape[0], context.shape[1], 3).to(x)
            mask_txt = torch.ones(context.shape[0], context.shape[1]).to(
                x.device, non_blocking=True).bool()
        return x, x_ids, txt, txt_ids, y, mask_x, mask_txt, x_seq_length

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            FluxMRACEPlus.para_dict,
                            set_name=True)
