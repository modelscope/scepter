# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from scepter.modules.model.registry import DIFFUSIONS
from scepter.modules.model.utils.basic_utils import check_list_of_list
from scepter.modules.model.utils.basic_utils import \
    pack_imagelist_into_tensor_v2 as pack_imagelist_into_tensor
from scepter.modules.model.utils.basic_utils import (
    to_device, unpack_tensor_into_imagelist)
from scepter.modules.utils.distribute import we
from scepter.modules.utils.logger import get_logger

from .diffusion_inference import DiffusionInference, get_model


def process_edit_image(images,
                       masks,
                       tasks,
                       max_seq_len=1024,
                       max_aspect_ratio=4,
                       d=16,
                       **kwargs):

    if not isinstance(images, list):
        images = [images]
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(tasks, list):
        tasks = [tasks]

    img_tensors = []
    mask_tensors = []
    for img, mask, task in zip(images, masks, tasks):
        if mask is None or mask == '':
            mask = Image.new('L', img.size, 0)
        W, H = img.size
        if H / W > max_aspect_ratio:
            img = TF.center_crop(img, [int(max_aspect_ratio * W), W])
            mask = TF.center_crop(mask, [int(max_aspect_ratio * W), W])
        elif W / H > max_aspect_ratio:
            img = TF.center_crop(img, [H, int(max_aspect_ratio * H)])
            mask = TF.center_crop(mask, [H, int(max_aspect_ratio * H)])

        H, W = img.height, img.width
        scale = min(1.0, math.sqrt(max_seq_len / ((H / d) * (W / d))))
        rH = int(H * scale) // d * d  # ensure divisible by self.d
        rW = int(W * scale) // d * d

        img = TF.resize(img, (rH, rW),
                        interpolation=TF.InterpolationMode.BICUBIC)
        mask = TF.resize(mask, (rH, rW),
                         interpolation=TF.InterpolationMode.NEAREST_EXACT)

        mask = np.asarray(mask)
        mask = np.where(mask > 128, 1, 0)
        mask = mask.astype(
            np.float32) if np.any(mask) else np.ones_like(mask).astype(
                np.float32)

        img_tensor = TF.to_tensor(img).to(we.device_id)
        img_tensor = TF.normalize(img_tensor,
                                  mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        mask_tensor = TF.to_tensor(mask).to(we.device_id)
        if task in ['inpainting', 'Try On', 'Inpainting']:
            mask_indicator = mask_tensor.repeat(3, 1, 1)
            img_tensor[mask_indicator == 1] = -1.0
        img_tensors.append(img_tensor)
        mask_tensors.append(mask_tensor)
    return img_tensors, mask_tensors


class TextEmbedding(nn.Module):
    def __init__(self, embedding_shape):
        super().__init__()
        self.pos = nn.Parameter(data=torch.zeros(embedding_shape))


class ACEInference(DiffusionInference):
    def __init__(self, logger=None):
        if logger is None:
            logger = get_logger(name='scepter')
        self.logger = logger
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model'
        ]

    def init_from_cfg(self, cfg):
        self.name = cfg.NAME
        self.is_default = cfg.get('IS_DEFAULT', False)
        module_paras = self.load_default(cfg.get('DEFAULT_PARAS', None))
        assert cfg.have('MODEL')

        self.diffusion_model = self.infer_model(
            cfg.MODEL.DIFFUSION_MODEL, module_paras.get(
                'DIFFUSION_MODEL',
                None)) if cfg.MODEL.have('DIFFUSION_MODEL') else None
        self.first_stage_model = self.infer_model(
            cfg.MODEL.FIRST_STAGE_MODEL,
            module_paras.get(
                'FIRST_STAGE_MODEL',
                None)) if cfg.MODEL.have('FIRST_STAGE_MODEL') else None
        self.cond_stage_model = self.infer_model(
            cfg.MODEL.COND_STAGE_MODEL,
            module_paras.get(
                'COND_STAGE_MODEL',
                None)) if cfg.MODEL.have('COND_STAGE_MODEL') else None
        self.diffusion = DIFFUSIONS.build(cfg.MODEL.DIFFUSION,
                                          logger=self.logger)

        self.interpolate_func = lambda x: (F.interpolate(
            x.unsqueeze(0),
            scale_factor=1 / self.size_factor,
            mode='nearest-exact') if x is not None else None)
        self.text_indentifers = cfg.MODEL.get('TEXT_IDENTIFIER', [])
        self.use_text_pos_embeddings = cfg.MODEL.get('USE_TEXT_POS_EMBEDDINGS',
                                                     False)
        if self.use_text_pos_embeddings:
            self.text_position_embeddings = TextEmbedding(
                (10, 4096)).eval().requires_grad_(False).to(we.device_id)
        else:
            self.text_position_embeddings = None

        self.max_seq_len = cfg.MODEL.DIFFUSION_MODEL.MAX_SEQ_LEN
        self.scale_factor = cfg.get('SCALE_FACTOR', 0.18215)
        self.size_factor = cfg.get('SIZE_FACTOR', 8)
        self.decoder_bias = cfg.get('DECODER_BIAS', 0)
        self.default_n_prompt = cfg.get('DEFAULT_N_PROMPT', '')

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        _, dtype = self.get_function_info(self.first_stage_model, 'encode')
        with torch.autocast('cuda',
                            enabled=(dtype != 'float32'),
                            dtype=getattr(torch, dtype)):
            z = [
                self.scale_factor * get_model(self.first_stage_model)._encode(
                    i.unsqueeze(0).to(getattr(torch, dtype))) for i in x
            ]
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        _, dtype = self.get_function_info(self.first_stage_model, 'decode')
        with torch.autocast('cuda',
                            enabled=(dtype != 'float32'),
                            dtype=getattr(torch, dtype)):
            x = [
                get_model(self.first_stage_model)._decode(
                    1. / self.scale_factor * i.to(getattr(torch, dtype)))
                for i in z
            ]
        return x

    @torch.no_grad()
    def __call__(self,
                 image=None,
                 mask=None,
                 prompt='',
                 task=None,
                 negative_prompt='',
                 output_height=512,
                 output_width=512,
                 sampler='ddim',
                 sample_steps=20,
                 guide_scale=4.5,
                 guide_rescale=0.5,
                 seed=-1,
                 history_io=None,
                 tar_index=0,
                 **kwargs):
        input_image, input_mask = image, mask
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(int(seed))

        if input_image is not None:
            # assert isinstance(input_image, list) and isinstance(input_mask, list)
            if task is None:
                task = [''] * len(input_image)
            if not isinstance(prompt, list):
                prompt = [prompt] * len(input_image)
            if history_io is not None and len(history_io) > 0:
                his_image, his_maks, his_prompt, his_task = history_io[
                    'image'], history_io['mask'], history_io[
                        'prompt'], history_io['task']
                assert len(his_image) == len(his_maks) == len(
                    his_prompt) == len(his_task)
                input_image = his_image + input_image
                input_mask = his_maks + input_mask
                task = his_task + task
                prompt = his_prompt + [prompt[-1]]
                prompt = [
                    pp.replace('{image}', f'{{image{i}}}') if i > 0 else pp
                    for i, pp in enumerate(prompt)
                ]

            edit_image, edit_image_mask = process_edit_image(
                input_image, input_mask, task, max_seq_len=self.max_seq_len)

            image, image_mask = edit_image[tar_index], edit_image_mask[
                tar_index]
            edit_image, edit_image_mask = [edit_image], [edit_image_mask]

        else:
            edit_image = edit_image_mask = [[]]
            image = torch.zeros(
                size=[3, int(output_height),
                      int(output_width)])
            image_mask = torch.ones(
                size=[1, int(output_height),
                      int(output_width)])
            if not isinstance(prompt, list):
                prompt = [prompt]

        image, image_mask, prompt = [image], [image_mask], [prompt]
        assert check_list_of_list(prompt) and check_list_of_list(
            edit_image) and check_list_of_list(edit_image_mask)
        # Assign Negative Prompt
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0]
        assert isinstance(negative_prompt, str)

        n_prompt = copy.deepcopy(prompt)
        for nn_p_id, nn_p in enumerate(n_prompt):
            assert isinstance(nn_p, list)
            n_prompt[nn_p_id][-1] = negative_prompt

        ctx, null_ctx = {}, {}

        # Get Noise Shape
        self.dynamic_load(self.first_stage_model, 'first_stage_model')
        image = to_device(image)
        x = self.encode_first_stage(image)
        self.dynamic_unload(self.first_stage_model,
                            'first_stage_model',
                            skip_loaded=True)
        noise = [
            torch.empty(*i.shape, device=we.device_id).normal_(generator=g)
            for i in x
        ]
        noise, x_shapes = pack_imagelist_into_tensor(noise)
        ctx['x_shapes'] = null_ctx['x_shapes'] = x_shapes

        image_mask = to_device(image_mask, strict=False)
        cond_mask = [self.interpolate_func(i) for i in image_mask
                     ] if image_mask is not None else [None] * len(image)
        ctx['x_mask'] = null_ctx['x_mask'] = cond_mask

        # Encode Prompt
        self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
        function_name, dtype = self.get_function_info(self.cond_stage_model)
        cont, cont_mask = getattr(get_model(self.cond_stage_model),
                                  function_name)(prompt)
        cont, cont_mask = self.cond_stage_embeddings(prompt, edit_image, cont,
                                                     cont_mask)
        null_cont, null_cont_mask = getattr(get_model(self.cond_stage_model),
                                            function_name)(n_prompt)
        null_cont, null_cont_mask = self.cond_stage_embeddings(
            prompt, edit_image, null_cont, null_cont_mask)
        self.dynamic_unload(self.cond_stage_model,
                            'cond_stage_model',
                            skip_loaded=False)
        ctx['crossattn'] = cont
        null_ctx['crossattn'] = null_cont

        # Encode Edit Images
        self.dynamic_load(self.first_stage_model, 'first_stage_model')
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
        self.dynamic_unload(self.first_stage_model,
                            'first_stage_model',
                            skip_loaded=True)
        null_ctx['edit'] = ctx['edit'] = e_img
        null_ctx['edit_mask'] = ctx['edit_mask'] = e_mask

        # Diffusion Process
        self.dynamic_load(self.diffusion_model, 'diffusion_model')
        function_name, dtype = self.get_function_info(self.diffusion_model)
        with torch.autocast('cuda',
                            enabled=dtype in ('float16', 'bfloat16'),
                            dtype=getattr(torch, dtype)):
            latent = self.diffusion.sample(
                noise=noise,
                sampler=sampler,
                model=get_model(self.diffusion_model),
                model_kwargs=[{
                    'cond':
                    ctx,
                    'mask':
                    cont_mask,
                    'text_position_embeddings':
                    self.text_position_embeddings.pos if hasattr(
                        self.text_position_embeddings, 'pos') else None
                }, {
                    'cond':
                    null_ctx,
                    'mask':
                    null_cont_mask,
                    'text_position_embeddings':
                    self.text_position_embeddings.pos if hasattr(
                        self.text_position_embeddings, 'pos') else None
                }] if guide_scale is not None and guide_scale > 1 else {
                    'cond':
                    null_ctx,
                    'mask':
                    cont_mask,
                    'text_position_embeddings':
                    self.text_position_embeddings.pos if hasattr(
                        self.text_position_embeddings, 'pos') else None
                },
                steps=sample_steps,
                show_progress=True,
                seed=seed,
                guide_scale=guide_scale,
                guide_rescale=guide_rescale,
                return_intermediate=None,
                **kwargs)
        self.dynamic_unload(self.diffusion_model,
                            'diffusion_model',
                            skip_loaded=False)

        # Decode to Pixel Space
        self.dynamic_load(self.first_stage_model, 'first_stage_model')
        samples = unpack_tensor_into_imagelist(latent, x_shapes)
        x_samples = self.decode_first_stage(samples)
        self.dynamic_unload(self.first_stage_model,
                            'first_stage_model',
                            skip_loaded=False)

        imgs = [
            torch.clamp((x_i + 1.0) / 2.0 + self.decoder_bias / 255,
                        min=0.0,
                        max=1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            for x_i in x_samples
        ]
        imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in imgs]
        return imgs

    def cond_stage_embeddings(self, prompt, edit_image, cont, cont_mask):
        if self.use_text_pos_embeddings and not torch.sum(
                self.text_position_embeddings.pos) > 0:
            identifier_cont, _ = getattr(get_model(self.cond_stage_model),
                                         'encode')(self.text_indentifers,
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
