# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import numbers
import re
from collections import OrderedDict

import numpy as np
import torch
from scepter.modules.model.network.train_module import TrainModule
from scepter.modules.model.registry import BACKBONES, LOSSES, MODELS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from einops import repeat
import math
import torch.nn.functional as F
class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var +
                    self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar +
                               torch.pow(sample - self.mean, 2) / self.var,
                               dim=dims)

    def mode(self):
        return self.mean


@MODELS.register_class()
class AutoencoderKL(TrainModule):
    para_dict = {
        'ENCODER': {},
        'DECODER': {},
        'LOSS': {},
        'EMBED_DIM': {
            'value': 4,
            'description': ''
        },
        'PRETRAINED_MODEL': {
            'value': None,
            'description': ''
        },
        'IGNORE_KEYS': {
            'value': [],
            'description': ''
        },
        'BATCH_SIZE': {
            'value': 16,
            'description': ''
        },
        'SCALE_FACTOR': {
            'value': None,
            'description':
            'if is not None, will used to scale the latent space.'
        },
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.encoder_cfg = self.cfg.ENCODER
        self.decoder_cfg = self.cfg.DECODER
        self.loss_cfg = self.cfg.get('LOSS', None)
        self.embed_dim = self.cfg.get('EMBED_DIM', 4)
        self.pretrained_model = self.cfg.get('PRETRAINED_MODEL', None)
        self.ignore_keys = self.cfg.get('IGNORE_KEYS', [])
        self.batch_size = self.cfg.get('BATCH_SIZE', 16)
        self.use_conv = self.cfg.get('USE_CONV', True)
        self.scale_factor = self.cfg.get('SCALE_FACTOR', None)

        self.construct_network()
        self.init_network()

    def construct_network(self):
        z_channels = self.encoder_cfg.Z_CHANNELS
        self.encoder = BACKBONES.build(self.encoder_cfg, logger=self.logger)
        self.decoder = BACKBONES.build(self.decoder_cfg, logger=self.logger)
        self.conv1 = torch.nn.Conv2d(
            2 * z_channels, 2 *
            self.embed_dim, 1) if self.use_conv else torch.nn.Identity()
        self.conv2 = torch.nn.Conv2d(
            self.embed_dim, z_channels,
            1) if self.use_conv else torch.nn.Identity()

        if self.loss_cfg is not None:
            self.loss = LOSSES.build(self.loss_cfg, logger=self.logger)

    def init_network(self):
        if self.pretrained_model is not None:
            with FS.get_from(self.pretrained_model,
                             wait_finish=True) as local_model:
                self.init_from_ckpt(local_model, ignore_keys=self.ignore_keys)

    def init_from_ckpt(self, path, ignore_keys):
        if path.find('.safetensors') > -1:
            from safetensors import safe_open
            sd = OrderedDict()
            with safe_open(path, framework='pt', device='cpu') as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        else:
            sd = torch.load(path, map_location='cpu')
            if path.find('.pt') > -1 and 'state_dict' in sd:
                sd = sd['state_dict']
            elif path.find('.ckpt') > -1 and 'state_dict' in sd:
                sd = sd['state_dict']

        new_sd = OrderedDict()

        for k, v in sd.items():
            if self.ignore_keys is not None:
                if (isinstance(self.ignore_keys, str) and re.match(self.ignore_keys, k)) or \
                        (isinstance(self.ignore_keys, list) and k in self.ignore_keys):
                    continue
            k = k.replace('post_quant_conv',
                          'conv2') if 'post_quant_conv' in k else k
            k = k.replace('quant_conv', 'conv1') if 'quant_conv' in k else k
            k = k.replace('first_stage_model.', '')
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

    def encode(self, x, return_mom=False):
        return torch.cat([
            self._encode(batch, return_mom=return_mom)
            for batch in x.split(self.batch_size, dim=0)
        ],
                         dim=0)

    def decode(self, z):
        return torch.cat(
            [self._decode(batch) for batch in z.split(self.batch_size, dim=0)],
            dim=0)

    def sample(self, moments):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        z = posterior.sample()
        return z

    def _encode(self, x, return_mom=False):

        h = self.encoder(x)
        moments = self.conv1(h)
        if return_mom:
            return moments
        mean, logvar = torch.chunk(moments, 2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        z = posterior.sample()
        if self.scale_factor is not None and isinstance(
                self.scale_factor, numbers.Number):
            z = self.scale_factor * z
        return z

    def _decode(self, z):
        if self.scale_factor is not None and isinstance(
                self.scale_factor, numbers.Number):
            z = z / self.scale_factor
        z = self.conv2(z)
        dec = self.decoder(z)
        return dec

    def share_forward(self, image, sample_posterior=True):
        posterior = self.encode(image)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def forward(self, **kwargs):
        if self.training:
            ret = self.forward_train(**kwargs)
        else:
            ret = self.forward_test(**kwargs)
        return ret

    def forward_train(self,
                      image=None,
                      sample_posterior=True,
                      optimizer_idx=0,
                      **kwargs):
        reconstructions, posterior = self.share_forward(
            image, sample_posterior)
        ret = {}
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(image,
                                            reconstructions,
                                            posterior,
                                            optimizer_idx,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split='train')
            ret['loss'] = aeloss
            ret.update(log_dict_ae)

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                image,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split='train')
            ret['loss'] = discloss
            ret.update(log_dict_disc)

        return ret

    def forward_test(self, image=None, sample_posterior=True, **kwargs):
        reconstructions, posterior = self.share_forward(
            image, sample_posterior)
        ret = {}
        aeloss, log_dict_ae = self.loss(image,
                                        reconstructions,
                                        posterior,
                                        0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split='val')
        discloss, log_dict_disc = self.loss(image,
                                            reconstructions,
                                            posterior,
                                            1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split='val')

        ret.update(log_dict_ae)
        ret.update(log_dict_disc)

        return ret

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            AutoencoderKL.para_dict,
                            set_name=True)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = torch.mm(timesteps.float().unsqueeze(1), freqs.unsqueeze(0)).view(timesteps.shape[0], len(freqs))
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

@MODELS.register_class()
class AutoencoderKLFlux(TrainModule):
    para_dict = {
        "ENCODER": {},
        "DECODER": {},
        "LOSS": {},
        "EMBED_DIM": {
            "value": 4,
            "description": ""
        },
        "PRETRAINED_MODEL": {
            "value": None,
            "description": ""
        },
        "IGNORE_KEYS": {
            "value": [],
            "description": ""
        },
        "BATCH_SIZE": {
            "value": 16,
            "description": ""
        },
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.encoder_cfg = self.cfg.ENCODER
        self.decoder_cfg = self.cfg.DECODER
        self.loss_cfg = self.cfg.get("LOSS", None)
        self.embed_dim = self.cfg.get("EMBED_DIM", 4)
        self.pretrained_model = self.cfg.get("PRETRAINED_MODEL", None)
        #
        self.ignore_keys = self.cfg.get("IGNORE_KEYS", [])
        self.batch_size = self.cfg.get("BATCH_SIZE", 16)
        self.resize_nx = self.cfg.get("RESIZE_NX", 1)
        self.use_rembed = self.cfg.get("USE_REMBED", True)
        self.use_conv = self.cfg.get('USE_CONV', True)
        self.scale_factor = self.cfg.get('SCALE_FACTOR', None)
        self.shift_factor = self.cfg.get('SHIFT_FACTOR', None)
        self.construct_network()

    def construct_network(self):
        z_channels = self.encoder_cfg.Z_CHANNELS
        self.encoder = BACKBONES.build(self.encoder_cfg, logger=self.logger)
        self.decoder = BACKBONES.build(self.decoder_cfg, logger=self.logger)
        self.conv1 = torch.nn.Conv2d(2 * z_channels, 2 * self.embed_dim, 1) if self.use_conv else torch.nn.Identity()
        self.conv2 = torch.nn.Conv2d(self.embed_dim, z_channels,1) if self.use_conv else torch.nn.Identity()

        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model is not None:
            with FS.get_from(pretrained_model, wait_finish=True) as local_model:
                self.init_from_ckpt(local_model)
    def init_from_ckpt(self, path, ignore_keys=list()):
        if path.find('.safetensors') > -1:
            from safetensors import safe_open
            sd = OrderedDict()
            with safe_open(path, framework="pt", device='cpu') as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        else:
            sd = torch.load(path, map_location="cpu")
            if path.find('.pt') > -1 and 'state_dict' in sd:
                sd = sd['state_dict']
            elif path.find('.ckpt') > -1 and 'state_dict' in sd:
                sd = sd['state_dict']
            elif path.find('.pth') > -1 and 'model' in sd:
                sd = sd['model']

        new_sd = OrderedDict()
        for k, v in sd.items():
            ignored = False
            for ik in ignore_keys:
                if ik in k:
                    if we.rank == 0:
                        self.logger.info("ignore key {} from state_dict.".format(k))
                    ignored = True
                    break
            k = k.replace("post_quant_conv", "conv2") if "post_quant_conv" in k else k
            k = k.replace("quant_conv", "conv1") if "quant_conv" in k else k
            if not ignored:
                new_sd[k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if we.rank == 0:
            self.logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                self.logger.info(f"Missing Keys:\n {missing}")
            if len(unexpected) > 0:
                self.logger.info(f"\nUnexpected Keys:\n {unexpected}")

    @torch.no_grad()
    def encode(self, x, sample_posterior = True):
        h = self.encoder(x)
        moments = self.conv1(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if self.shift_factor is not None and isinstance(self.shift_factor, numbers.Number):
            z = z - self.shift_factor
        if self.scale_factor is not None and isinstance(self.scale_factor, numbers.Number):
            z = self.scale_factor * z
        return z, posterior

    def decode(self, z, **kwargs):
        b, c, h, w = z.size()
        if kwargs.get('resize_nx', None) is not None and self.use_rembed:
            resize_nx = kwargs['resize_nx']
            if not torch.is_tensor(resize_nx):
                resize_nx = torch.full((b,), resize_nx, device=we.device_id, dtype=z.dtype)
            rembed = timestep_embedding(resize_nx, dim=self.decoder_cfg.CH_MULT[-1] * self.decoder_cfg.CH)
        else:
            rembed = None
        if self.scale_factor is not None and isinstance(self.scale_factor, numbers.Number):
            z = z / self.scale_factor
        if self.shift_factor is not None and isinstance(self.shift_factor, numbers.Number):
            z = z + self.shift_factor
        z = self.conv2(z)
        # add grad;
        if rembed is not None:
            dec = self.decoder(z, rembed)
        else:
            dec = self.decoder(z)
        return dec

    def share_forward(self, image=None, sample_posterior=True, **kwargs):
        # rembed: resize embedding
        if image is not None:
            z, posterior = self.encode(image, sample_posterior = sample_posterior)
        else:
            latent = kwargs.pop("latent", None)
            assert latent is not None
            z = latent
            posterior = None
        if self.shift_factor is not None and isinstance(self.scale_factor, numbers.Number):
            z = z - self.shift_factor
        if self.scale_factor is not None and isinstance(self.scale_factor, numbers.Number):
            z = self.scale_factor * z
        dec = self.decode(z, **kwargs)
        return dec, posterior

    def forward(self, **kwargs):
        if self.training:
            ret = self.forward_train(**kwargs)
        else:
            ret = self.forward_test(**kwargs)
        return ret

    def forward_train(self,
                      image=None,
                      gt_image=None,
                      sample_posterior=True,
                      optimizer_idx=0,
                      global_step=0,
                      **kwargs):

        if gt_image is None:
            gt_image = copy.deepcopy(image)
        reconstructions, posterior = self.share_forward(image, sample_posterior, **kwargs)

        ret = {}
        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(gt_image, reconstructions, posterior, optimizer_idx,
                                            global_step, last_layer=self.get_last_layer(), split="train")
            # self.logger.info(f"aeloss: {aeloss.detach().cpu().item()}, ")
            ret["loss"] = aeloss
            ret.update(log_dict_ae)

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(gt_image, reconstructions, posterior, optimizer_idx,
                                                global_step, last_layer=self.get_last_layer(), split="train")
            # self.logger.info(f"discloss: {discloss.detach().cpu().item()}, ")
            ret["loss"] = discloss
            ret.update(log_dict_disc)

        return ret

    @torch.no_grad()
    def forward_test(self,
                     image=None,
                     gt_image=None,
                     sample_posterior=True,
                     **kwargs):
        resize_nx_ = 1
        if image is not None:
            b, c, h, w = image.size()
            if kwargs.get('resize_ex'):
                resize_nx_ = kwargs.pop('resize_ex')
                image = F.interpolate(image, (int(float(h) / resize_nx_), int(float(w) / resize_nx_)), mode='bicubic')
                image = F.interpolate(image, (h, w), mode='bicubic')
                kwargs["resize_nx"] = resize_nx_
            elif kwargs.get('resize_nx', None) is not None:
                resize_nx_ = kwargs['resize_nx']

        if gt_image is None:
            if image is not None:
                gt_image = copy.deepcopy(image)

        # kwargs["resize_nx"] = resize_nx_
        reconstructions, posterior = self.share_forward(image, sample_posterior, **kwargs)
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)
        if gt_image is not None:
            gt_image = torch.clamp((gt_image + 1.0) / 2.0, min=0.0, max=1.0)
        else:
            gt_image = [None for _ in range(reconstructions.shape[0])]
        if image is not None:
            lr_image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        else:
            lr_image = [None for _ in range(reconstructions.shape[0])]

        ret = list()
        if torch.is_tensor(resize_nx_):
            resize_nx = [nx.item() for nx, in zip(resize_nx_.cpu())]
        else:
            resize_nx = [resize_nx_ for _ in range(reconstructions.size(0))]

        for img, ori, gt_img, nx in zip(reconstructions, lr_image, gt_image, resize_nx):
            ret.append({
                "prompt": "",
                "n_prompt": "",
                "image": img,
                "lr_image": ori,
                "gt_image": gt_img,
                "resize_nx": nx
            })

        return ret

    def get_last_layer(self):
        if hasattr(self.decoder, 'conv_out'):
            return self.decoder.conv_out.weight
        else:
            return self.decoder.head[-1].weight

    @staticmethod
    def get_config_template():
        return dict_to_yaml("MODEL", __class__.__name__, AutoencoderKLFlux.para_dict, set_name=True)
