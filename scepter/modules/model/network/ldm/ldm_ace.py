# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from scepter.modules.model.network.ldm import LatentDiffusion
from scepter.modules.model.registry import MODELS
from scepter.modules.model.utils.basic_utils import check_list_of_list
from scepter.modules.model.utils.basic_utils import \
    pack_imagelist_into_tensor_v2 as pack_imagelist_into_tensor
from scepter.modules.model.utils.basic_utils import (
    to_device, unpack_tensor_into_imagelist)
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we


class TextEmbedding(nn.Module):
    def __init__(self, embedding_shape):
        super().__init__()
        self.pos = nn.Parameter(data=torch.zeros(embedding_shape))


@MODELS.register_class()
class LatentDiffusionACE(LatentDiffusion):
    para_dict = LatentDiffusion.para_dict
    para_dict['DECODER_BIAS'] = {'value': 0, 'description': ''}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.interpolate_func = lambda x: (F.interpolate(
            x.unsqueeze(0),
            scale_factor=1 / self.size_factor,
            mode='nearest-exact') if x is not None else None)

        self.text_indentifers = cfg.get('TEXT_IDENTIFIER', [])
        self.use_text_pos_embeddings = cfg.get('USE_TEXT_POS_EMBEDDINGS',
                                               False)
        if self.use_text_pos_embeddings:
            self.text_position_embeddings = TextEmbedding(
                (10, 4096)).eval().requires_grad_(False)
        else:
            self.text_position_embeddings = None

        self.logger.info(self.model)

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        return [
            self.scale_factor *
            self.first_stage_model._encode(i.unsqueeze(0).to(torch.float16))
            for i in x
        ]

    @torch.no_grad()
    def decode_first_stage(self, z):
        return [
            self.first_stage_model._decode(1. / self.scale_factor *
                                           i.to(torch.float16)) for i in z
        ]

    def cond_stage_embeddings(self, prompt, edit_image, cont, cont_mask):
        if self.use_text_pos_embeddings and not torch.sum(
                self.text_position_embeddings.pos) > 0:
            identifier_cont, identifier_cont_mask = getattr(
                self.cond_stage_model, 'encode')(self.text_indentifers,
                                                 return_mask=True)
            self.text_position_embeddings.load_state_dict(
                {'pos': identifier_cont[:, 0, :]})
        cont_, cont_mask_ = [], []
        for pp, edit, c, cm in zip(prompt, edit_image, cont, cont_mask):
            if isinstance(pp, list):
                cont_.append([c[-1], *c] if len(edit) > 0 else [c[-1]])
                cont_mask_.append([cm[-1], *cm] if len(edit) > 0 else [cm[-1]])
            else:
                raise NotImplementedError

        return cont_, cont_mask_

    def limit_batch_data(self, batch_data_list, log_num):
        if log_num and log_num > 0:
            batch_data_list_limited = []
            for sub_data in batch_data_list:
                if sub_data is not None:
                    sub_data = sub_data[:log_num]
                batch_data_list_limited.append(sub_data)
            return batch_data_list_limited
        else:
            return batch_data_list

    def forward_train(self,
                      edit_image=[],
                      edit_image_mask=[],
                      image=None,
                      image_mask=None,
                      noise=None,
                      prompt=[],
                      **kwargs):
        '''
        Args:
            edit_image: list of list of edit_image
            edit_image_mask: list of list of edit_image_mask
            image: target image
            image_mask: target image mask
            noise: default is None, generate automaticly
            prompt: list of list of text
            **kwargs:
        Returns:
        '''
        assert check_list_of_list(prompt) and check_list_of_list(
            edit_image) and check_list_of_list(edit_image_mask)
        assert len(edit_image) == len(edit_image_mask) == len(prompt)
        assert self.cond_stage_model is not None
        gc_seg = kwargs.pop('gc_seg', [])
        gc_seg = int(gc_seg[0]) if len(gc_seg) > 0 else 0
        context = {}

        # process image
        image = to_device(image)
        x_start = self.encode_first_stage(image, **kwargs)
        x_start, x_shapes = pack_imagelist_into_tensor(x_start)  # B, C, L
        n, _, _ = x_start.shape
        t = torch.randint(0, self.num_timesteps, (n, ),
                          device=x_start.device).long()
        context['x_shapes'] = x_shapes

        # process image mask
        image_mask = to_device(image_mask, strict=False)
        context['x_mask'] = [self.interpolate_func(i) for i in image_mask
                             ] if image_mask is not None else [None] * n

        # process text
        # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        prompt_ = [[pp] if isinstance(pp, str) else pp for pp in prompt]
        try:
            cont, cont_mask = getattr(self.cond_stage_model,
                                      'encode_list')(prompt_, return_mask=True)
        except Exception as e:
            print(e, prompt_)
        cont, cont_mask = self.cond_stage_embeddings(prompt, edit_image, cont,
                                                     cont_mask)
        context['crossattn'] = cont

        # process edit image & edit image mask
        edit_image = [to_device(i, strict=False) for i in edit_image]
        edit_image_mask = [to_device(i, strict=False) for i in edit_image_mask]
        e_img, e_mask = [], []
        for u, m in zip(edit_image, edit_image_mask):
            if m is None:
                m = [None] * len(u) if u is not None else [None]
            e_img.append(
                self.encode_first_stage(u, **kwargs) if u is not None else u)
            e_mask.append([
                self.interpolate_func(i) if i is not None else None for i in m
            ])
        context['edit'], context['edit_mask'] = e_img, e_mask

        # process loss
        loss = self.diffusion.loss(
            x_0=x_start,
            t=t,
            noise=noise,
            model=self.model,
            model_kwargs={
                'cond':
                context,
                'mask':
                cont_mask,
                'gc_seg':
                gc_seg,
                'text_position_embeddings':
                self.text_position_embeddings.pos if hasattr(
                    self.text_position_embeddings, 'pos') else None
            },
            **kwargs)
        loss = loss.mean()
        ret = {'loss': loss, 'probe_data': {'prompt': prompt}}
        return ret

    @torch.no_grad()
    def forward_test(self,
                     edit_image=[],
                     edit_image_mask=[],
                     image=None,
                     image_mask=None,
                     prompt=[],
                     n_prompt=[],
                     sampler='ddim',
                     sample_steps=20,
                     guide_scale=4.5,
                     guide_rescale=0.5,
                     log_num=-1,
                     seed=2024,
                     **kwargs):

        assert check_list_of_list(prompt) and check_list_of_list(
            edit_image) and check_list_of_list(edit_image_mask)
        assert len(edit_image) == len(edit_image_mask) == len(prompt)
        assert self.cond_stage_model is not None
        # gc_seg is unused
        kwargs.pop('gc_seg', -1)
        # prepare data
        context, null_context = {}, {}

        prompt, n_prompt, image, image_mask, edit_image, edit_image_mask = self.limit_batch_data(
            [prompt, n_prompt, image, image_mask, edit_image, edit_image_mask],
            log_num)
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        n_prompt = copy.deepcopy(prompt)
        # only modify the last prompt to be zero
        for nn_p_id, nn_p in enumerate(n_prompt):
            if isinstance(nn_p, str):
                n_prompt[nn_p_id] = ['']
            elif isinstance(nn_p, list):
                n_prompt[nn_p_id][-1] = ''
            else:
                raise NotImplementedError
        # process image
        image = to_device(image)
        x = self.encode_first_stage(image, **kwargs)
        noise = [
            torch.empty(*i.shape, device=we.device_id).normal_(generator=g)
            for i in x
        ]
        noise, x_shapes = pack_imagelist_into_tensor(noise)
        context['x_shapes'] = null_context['x_shapes'] = x_shapes

        # process image mask
        image_mask = to_device(image_mask, strict=False)
        cond_mask = [self.interpolate_func(i) for i in image_mask
                     ] if image_mask is not None else [None] * len(image)
        context['x_mask'] = null_context['x_mask'] = cond_mask
        # process text
        # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        prompt_ = [[pp] if isinstance(pp, str) else pp for pp in prompt]
        cont, cont_mask = getattr(self.cond_stage_model,
                                  'encode_list')(prompt_, return_mask=True)
        cont, cont_mask = self.cond_stage_embeddings(prompt, edit_image, cont,
                                                     cont_mask)
        null_cont, null_cont_mask = getattr(self.cond_stage_model,
                                            'encode_list')(n_prompt,
                                                           return_mask=True)
        null_cont, null_cont_mask = self.cond_stage_embeddings(
            prompt, edit_image, null_cont, null_cont_mask)
        context['crossattn'] = cont
        null_context['crossattn'] = null_cont

        # processe edit image & edit image mask
        edit_image = [to_device(i, strict=False) for i in edit_image]
        edit_image_mask = [to_device(i, strict=False) for i in edit_image_mask]
        e_img, e_mask = [], []
        for u, m in zip(edit_image, edit_image_mask):
            if u is None:
                continue
            if m is None:
                m = [None] * len(u)
            e_img.append(self.encode_first_stage(u, **kwargs))
            e_mask.append([self.interpolate_func(i) for i in m])
        null_context['edit'] = context['edit'] = e_img
        null_context['edit_mask'] = context['edit_mask'] = e_mask

        # process sample
        model = self.model_ema if self.use_ema and self.eval_ema else self.model
        embedding_context = model.no_sync if isinstance(model, torch.distributed.fsdp.FullyShardedDataParallel) \
            else nullcontext
        with embedding_context():
            samples = self.diffusion.sample(
                sampler=sampler,
                noise=noise,
                model=model,
                model_kwargs=[{
                    'cond':
                    context,
                    'mask':
                    cont_mask,
                    'text_position_embeddings':
                    self.text_position_embeddings.pos if hasattr(
                        self.text_position_embeddings, 'pos') else None
                }, {
                    'cond':
                    null_context,
                    'mask':
                    null_cont_mask,
                    'text_position_embeddings':
                    self.text_position_embeddings.pos if hasattr(
                        self.text_position_embeddings, 'pos') else None
                }] if guide_scale is not None and guide_scale > 1 else {
                    'cond':
                    context,
                    'mask':
                    cont_mask,
                    'text_position_embeddings':
                    self.text_position_embeddings.pos if hasattr(
                        self.text_position_embeddings, 'pos') else None
                },
                steps=sample_steps,
                guide_scale=guide_scale,
                guide_rescale=guide_rescale,
                show_progress=True,
                **kwargs)

        samples = unpack_tensor_into_imagelist(samples, x_shapes)
        x_samples = self.decode_first_stage(samples)
        outputs = list()
        for i in range(len(prompt)):
            rec_img = torch.clamp(
                (x_samples[i] + 1.0) / 2.0 + self.decoder_bias / 255,
                min=0.0,
                max=1.0)
            rec_img = rec_img.squeeze(0)
            edit_imgs, edit_img_masks = [], []
            if edit_image is not None and edit_image[i] is not None:
                if edit_image_mask[i] is None:
                    edit_image_mask[i] = [None] * len(edit_image[i])
                for edit_img, edit_mask in zip(edit_image[i],
                                               edit_image_mask[i]):
                    edit_img = torch.clamp((edit_img + 1.0) / 2.0,
                                           min=0.0,
                                           max=1.0)
                    edit_imgs.append(edit_img.squeeze(0))
                    if edit_mask is None:
                        edit_mask = torch.ones_like(edit_img[[0], :, :])
                    edit_img_masks.append(edit_mask)
            one_tup = {
                'reconstruct_image': rec_img,
                'instruction': prompt[i],
                'edit_image': edit_imgs if len(edit_imgs) > 0 else None,
                'edit_mask': edit_img_masks if len(edit_imgs) > 0 else None
            }
            if image is not None:
                if image_mask is None:
                    image_mask = [None] * len(image)
                ori_img = torch.clamp((image[i] + 1.0) / 2.0, min=0.0, max=1.0)
                one_tup['target_image'] = ori_img.squeeze(0)
                one_tup['target_mask'] = image_mask[i] if image_mask[
                    i] is not None else torch.ones_like(ori_img[[0], :, :])
            outputs.append(one_tup)
        return outputs

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            LatentDiffusionACE.para_dict,
                            set_name=True)
