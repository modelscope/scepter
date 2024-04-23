# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os.path
import random
from collections import OrderedDict

import gradio as gr
import torch
import torchvision.transforms.functional as TF
from scepter.modules.model.utils.data_utils import crop_back
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

from .diffusion_inference import DiffusionInference


def get_model(model_tuple):
    assert 'model' in model_tuple
    return model_tuple['model']


class LargenInference(DiffusionInference):
    '''
        define vae, unet, text-encoder, tuner, refiner components
        support to load the components dynamicly.
        create and load model when run this model at the first time.
    '''
    def __init__(self, logger=None):
        self.logger = logger
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model'
        ]

    def redefine_paras(self, cfg):
        if cfg.get('PRETRAINED_MODEL', None):
            assert FS.isfile(cfg.PRETRAINED_MODEL)
            with FS.get_from(cfg.PRETRAINED_MODEL,
                             wait_finish=True) as local_path:
                if local_path.endswith('safetensors'):
                    from safetensors.torch import load_file as load_safetensors
                    sd = load_safetensors(local_path)
                else:
                    sd = torch.load(local_path, map_location='cpu')

                if 'model' in sd:
                    sd = sd['model']

                first_stage_model_path = os.path.join(
                    os.path.dirname(local_path), 'first_stage_model.pth')
                cond_stage_model_path = os.path.join(
                    os.path.dirname(local_path), 'cond_stage_model.pth')
                diffusion_model_path = os.path.join(
                    os.path.dirname(local_path), 'diffusion_model.pth')
                if (not os.path.exists(first_stage_model_path)
                        or not os.path.exists(cond_stage_model_path)
                        or not os.path.exists(diffusion_model_path)):
                    self.logger.info(
                        'Now read the whole model and rearrange the modules, it may take several mins.'
                    )
                    first_stage_model = OrderedDict()
                    cond_stage_model = OrderedDict()
                    diffusion_model = OrderedDict()
                    for k, v in sd.items():
                        if k.startswith('first_stage_model.'):
                            first_stage_model[k.replace(
                                'first_stage_model.', '')] = v
                        elif k.startswith('conditioner.'):
                            cond_stage_model[k.replace('conditioner.', '')] = v
                        elif k.startswith('cond_stage_model.'):
                            if k.startswith('cond_stage_model.model.'):
                                cond_stage_model[k.replace(
                                    'cond_stage_model.model.', '')] = v
                            else:
                                cond_stage_model[k.replace(
                                    'cond_stage_model.', '')] = v
                        elif k.startswith('model.diffusion_model.'):
                            diffusion_model[k.replace('model.diffusion_model.',
                                                      '')] = v
                        elif k.startswith('model.'):
                            diffusion_model[k.replace('model.', '')] = v
                        else:
                            continue
                    if cfg.have('FIRST_STAGE_MODEL'):
                        with open(first_stage_model_path + 'cache', 'wb') as f:
                            torch.save(first_stage_model, f)
                        os.rename(first_stage_model_path + 'cache',
                                  first_stage_model_path)
                        self.logger.info(
                            'First stage model has been processed.')
                    if cfg.have('COND_STAGE_MODEL'):
                        with open(cond_stage_model_path + 'cache', 'wb') as f:
                            torch.save(cond_stage_model, f)
                        os.rename(cond_stage_model_path + 'cache',
                                  cond_stage_model_path)
                        self.logger.info(
                            'Cond stage model has been processed.')
                    if cfg.have('DIFFUSION_MODEL'):
                        with open(diffusion_model_path + 'cache', 'wb') as f:
                            torch.save(diffusion_model, f)
                        os.rename(diffusion_model_path + 'cache',
                                  diffusion_model_path)
                        self.logger.info('Diffusion model has been processed.')
                if not cfg.FIRST_STAGE_MODEL.get('PRETRAINED_MODEL', None):
                    cfg.FIRST_STAGE_MODEL.PRETRAINED_MODEL = first_stage_model_path
                else:
                    cfg.FIRST_STAGE_MODEL.RELOAD_MODEL = first_stage_model_path
                if not cfg.COND_STAGE_MODEL.get('PRETRAINED_MODEL', None):
                    cfg.COND_STAGE_MODEL.PRETRAINED_MODEL = cond_stage_model_path
                else:
                    cfg.COND_STAGE_MODEL.RELOAD_MODEL = cond_stage_model_path
                if not cfg.DIFFUSION_MODEL.get('PRETRAINED_MODEL', None):
                    cfg.DIFFUSION_MODEL.PRETRAINED_MODEL = diffusion_model_path
                else:
                    cfg.DIFFUSION_MODEL.RELOAD_MODEL = diffusion_model_path
        return cfg

    @torch.no_grad()
    def __call__(self,
                 input,
                 num_samples=1,
                 intermediate_callback=None,
                 refine_strength=0,
                 img_to_img_strength=0,
                 cat_uc=True,
                 tuner_model=None,
                 control_model=None,
                 largen_state=False,
                 **kwargs):
        if not largen_state:
            raise gr.Error('LARGEN model must be used with LAR-Gen settings')

        value_input = copy.deepcopy(self.input)
        value_input.update(input)
        print(value_input)
        height, width = value_input['target_size_as_tuple']
        value_output = copy.deepcopy(self.output)
        batch, batch_uc = self.get_batch(value_input, num_samples=1)

        # first stage encode
        task = kwargs.get('largen_task', 'Text_Guided_Inpainting')
        image_scale = kwargs.get('largen_image_scale', 1.0)
        tar_image = kwargs.get('largen_tar_image', None)
        tar_mask = kwargs.get('largen_tar_mask', None)
        masked_image = kwargs.get('largen_masked_image', None)
        ref_image = kwargs.get('largen_ref_image', None)
        ref_mask = kwargs.get('largen_ref_mask', None)
        ref_clip = kwargs.get('largen_ref_clip', None)

        base_image = kwargs.get('largen_base_image', None)
        extra_sizes = kwargs.get('largen_extra_sizes', None)
        bbox_yyxx = kwargs.get('largen_bbox_yyxx', None)

        device = we.device_id
        tar_image = tar_image.to(device)
        tar_mask = tar_mask.to(device)
        masked_image = masked_image.to(device)
        if 'Subject' in task:
            ref_image = ref_image.to(device)
            ref_mask = ref_mask.to(device)
            ref_clip = ref_clip.to(device)

        self.dynamic_load(self.first_stage_model, 'first_stage_model')

        tar_x0 = self.encode_first_stage(tar_image)
        masked_x0 = self.encode_first_stage(masked_image)
        b, _, h, w = tar_x0.shape
        tar_mask_latent = TF.resize(tar_mask, (h, w), antialias=True)
        tar_mask_latent = (tar_mask_latent > 0.5).float()

        batch.update({
            'tar_x0': tar_x0,
            'tar_mask_latent': tar_mask_latent,
            'masked_x0': masked_x0,
            'task': task
        })
        batch_uc.update({
            'tar_x0': tar_x0,
            'tar_mask_latent': tar_mask_latent,
            'masked_x0': masked_x0,
            'task': task
        })

        if 'Subject' in task and ref_image is not None:
            ref_x0 = self.encode_first_stage(ref_image)
            batch.update({
                'ref_ip': ref_clip,
                'ref_detail': ref_clip,
                'ref_x0': ref_x0,
                'ref_mask': ref_mask,
                'image_scale': image_scale,
            })
            batch_uc.update({
                'ref_ip': torch.zeros_like(ref_clip),
                'ref_detail': ref_clip,
                'ref_x0': ref_x0,
                'ref_mask': ref_mask,
                'image_scale': image_scale,
            })

        self.dynamic_unload(self.first_stage_model,
                            'first_stage_model',
                            skip_loaded=False)

        # cond stage
        self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
        function_name, dtype = self.get_function_info(self.cond_stage_model)
        with torch.autocast('cuda',
                            enabled=dtype == 'float16',
                            dtype=getattr(torch, dtype)):
            context = getattr(get_model(self.cond_stage_model),
                              function_name)(batch)
            null_context = getattr(get_model(self.cond_stage_model),
                                   function_name)(batch_uc)

        self.dynamic_unload(self.cond_stage_model,
                            'cond_stage_model',
                            skip_loaded=False)

        # get noise
        seed = kwargs.pop('seed', -1)
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        if 'seed' in value_output:
            value_output['seed'] = seed
        for sample_id in range(num_samples):
            if self.diffusion_model is not None:
                noise = torch.empty(
                    1,
                    4,
                    height // self.first_stage_model['paras']['size_factor'],
                    width // self.first_stage_model['paras']['size_factor'],
                    device=we.device_id).normal_(generator=g)

                self.dynamic_load(self.diffusion_model, 'diffusion_model')
                # UNet use input n_prompt
                function_name, dtype = self.get_function_info(
                    self.diffusion_model)
                with torch.autocast('cuda',
                                    enabled=dtype == 'float16',
                                    dtype=getattr(torch, dtype)):
                    latent = self.diffusion.sample(
                        noise=noise,
                        x=None,
                        denoising_strength=1.0,
                        refine_strength=refine_strength,
                        solver=value_input.get('sample', 'ddim'),
                        model=get_model(self.diffusion_model),
                        model_kwargs=[{
                            'cond': context
                        }, {
                            'cond': null_context
                        }],
                        steps=value_input.get('sample_steps', 50),
                        guide_scale=value_input.get('guide_scale', 7.5),
                        guide_rescale=value_input.get('guide_rescale', 0.5),
                        discretization=value_input.get('discretization',
                                                       'trailing'),
                        show_progress=True,
                        seed=seed,
                        condition_fn=None,
                        clamp=None,
                        percentile=None,
                        t_max=None,
                        t_min=None,
                        discard_penultimate_step=None,
                        intermediate_callback=intermediate_callback,
                        cat_uc=cat_uc,
                        **kwargs)

                self.dynamic_unload(self.diffusion_model,
                                    'diffusion_model',
                                    skip_loaded=True)

            if 'latent' in value_output:
                if value_output['latent'] is None or (
                        isinstance(value_output['latent'], list)
                        and len(value_output['latent']) < 1):
                    value_output['latent'] = []
                value_output['latent'].append(latent)

            self.dynamic_load(self.first_stage_model, 'first_stage_model')
            x_samples = self.decode_first_stage(latent).float()
            self.dynamic_unload(self.first_stage_model,
                                'first_stage_model',
                                skip_loaded=False)
            images = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            if base_image is not None:
                stitch_images = []
                for img in images:
                    stitch_img = crop_back(img, copy.deepcopy(base_image),
                                           extra_sizes, bbox_yyxx)
                    stitch_images.append(stitch_img)
                images = torch.stack(stitch_images, dim=0)
            if 'images' in value_output:
                if value_output['images'] is None or (
                        isinstance(value_output['images'], list)
                        and len(value_output['images']) < 1):
                    value_output['images'] = []
                value_output['images'].append(images)

        for k, v in value_output.items():
            if isinstance(v, list):
                value_output[k] = torch.cat(v, dim=0)
            if isinstance(v, torch.Tensor):
                value_output[k] = v.cpu()

        return value_output
