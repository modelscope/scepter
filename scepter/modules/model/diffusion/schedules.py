import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import numpy as np

from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.math_plot import plot_multi_curves
from scepter.modules.model.registry import NOISE_SCHEDULERS
from torch import Tensor

from .util import _i

@dataclass
class ScheduleOutput(object):
    x_t: torch.Tensor
    x_0: torch.Tensor
    t: torch.Tensor
    sigma: torch.Tensor
    alpha: torch.Tensor
    custom_fields: dict = field(default_factory=dict)

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


@NOISE_SCHEDULERS.register_class()
class BaseNoiseScheduler(object):
    '''
    In the diffusion model, the parameters related to the noise schedule are alpha, beta,
    and sigma. The following are the definitions of the above three parameters, which should
    be the basic property for the instance of noise scheduler.
        \alpha_{t} = \sqrt{1 - \beta_{t}^2} \alpha is the strength of signal and \beta is the strength of noise
        \sigma_{t} = \sqrt{1 - \overline\alpha} = \sqrt{1 - \prod_{i=1}^{t}\alpha^2_{i}} (P(x_{t}|x_{0}) ~ N(\overline\alpha x_{0}, \sigma^2))

    where sigma_{t} is the var of p(x_{t-1}|x_{t}, x_{0}).

    (reference to https://arxiv.org/abs/2010.02502)
    let sigma transfer to beta:
        square_\beta = 1 - \frac{1 - square_\sigma_{t}}{1 - square_\sigma_{t - 1 }}

    '''
    para_dict = {
        "NUM_TIMESTEPS": {
            "value": 1000,
            "description": "The number of timesteps for sampling."
        },
    }
    def __init__(self, cfg, logger=None):
        super(BaseNoiseScheduler, self).__init__()
        self.logger = logger
        self.cfg = cfg
        self.init_params()
        self.get_schedule()
        # self.check_function()

    def init_params(self):
        self.num_timesteps = self.cfg.get("NUM_TIMESTEPS", 1000)
        self._sample_steps = torch.arange(self.num_timesteps, dtype=torch.float32)
        self._sigmas, self._betas, self._alphas, self._timesteps = None, None, None, None
    def check_function(self):
        # for the same t, we should gurantee t_to_sigma and sigma_to_t is aligned
        try:
            predict_timestamps = self.sigma_to_t(self.sigmas)
            predict_sigmas = self.t_to_sigma(self._timesteps)
            diff_sigmas = torch.sum(torch.abs(predict_sigmas - self.sigmas))
            diff_timestamps = torch.sum(torch.abs(predict_timestamps - self._timesteps))
            if diff_sigmas > 1e-3 or diff_timestamps > 1:
                self.logger.info(f"The noise scheduler {self.__class__.__name__} is not correct, "
                                 f"please check the function sigma_to_t or t_to_sigma."
                                 f"Info: diff sigmas {diff_sigmas}, diff timestamps {diff_timestamps}")
                raise "The noise scheduler checked failed."
            else:
                self.logger.info(f"The noise scheduler {self.__class__.__name__} is checked and passed.")
        except Exception as e:
            if isinstance(e, NotImplementedError):
                self.logger.info("Not implemented function sigma_to_t or t_to_sigma, skip check.")
            else:
                self.logger.info(f"The noise scheduler {self.__class__.__name__} is not correct, "
                                   f"please check the function sigma_to_t or t_to_sigma. Error: {e}")
                raise e

    def get_schedule(self):
        raise NotImplementedError(f'NoiseScheduler get_schedule function not implemented')

    def square_betas_to_sigmas(self, square_betas):
        return torch.sqrt(1 - torch.cumprod(1 - square_betas, dim=0))
    def sigmas_to_square_betas(self, sigmas):
        square_alphas = 1 - sigmas ** 2
        betas = 1 - torch.cat([square_alphas[:1], square_alphas[1:] / square_alphas[:-1]])
        return betas
    def sigma_to_t(self, sigma, **kwargs):
        if sigma == float('inf'):
            t = torch.full_like(sigma, len(self._sigmas) - 1)
        else:
            log_sigmas = torch.sqrt(self._sigmas**2 /
                                    (1 - self._sigmas**2)).log().to(sigma)
            log_sigma = sigma.log()
            dists = log_sigma - log_sigmas[:, None]
            low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(
                max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
            low, high = log_sigmas[low_idx], log_sigmas[high_idx]
            w = (low - log_sigma) / (low - high)
            w = w.clamp(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            t = t.view(sigma.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t
    def t_to_sigma(self, t, **kwargs):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = torch.sqrt(self.sigmas**2 /
                                (1 - self.sigmas**2)).log().to(t)
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        log_sigma[torch.isnan(log_sigma)
                  | torch.isinf(log_sigma)] = float('inf')
        return log_sigma.exp()

    def t_to_alpha(self, t, **kwargs):
        sigma = self.t_to_sigma(t)
        square_beta = self.sigmas_to_square_betas(sigma)
        return torch.sqrt(1 - square_beta)

    def t_to_beta(self, t, **kwargs):
        sigma = self.t_to_sigma(t)
        square_beta = self.sigmas_to_square_betas(sigma)
        return torch.sqrt(square_beta)

    def add_noise(self, x_0, noise = None, t = None):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device).long()
        alpha = _i(self.alphas, t, x_0)
        sigma = _i(self.sigmas, t, x_0)
        x_t = alpha * x_0 + sigma * noise

        return ScheduleOutput(x_0 = x_0, x_t = x_t, t = t, alpha=alpha, sigma=sigma)

    def t_to_alpha_init(self, t, **kwargs):
        indices = t.long()
        indices[indices >= self.num_timesteps] = self.num_timesteps - 1
        timesteps = self.timesteps.to(t)[indices]
        step_indices = [(self.timesteps.to(t) == t).nonzero().item() for t in timesteps]
        alpha = self.alphas[step_indices].flatten().to(t)
        return alpha

    def t_to_beta_init(self, t, **kwargs):
        indices = t.long()
        indices[indices >= self.num_timesteps] = self.num_timesteps - 1
        timesteps = self.timesteps.to(t)[indices]
        step_indices = [(self.timesteps.to(t) == t).nonzero().item() for t in timesteps]
        beta = self.betas[step_indices].flatten().to(t)
        return beta

    def t_to_sigma_init(self, t, **kwargs):
        indices = t.long()
        indices[indices >= self.num_timesteps] = self.num_timesteps - 1
        timesteps = self.timesteps.to(t)[indices]
        step_indices = [(self.timesteps.to(t) == t).nonzero().item() for t in timesteps]
        sigma = self.sigmas[step_indices].flatten().to(t)
        return sigma

    def rescale_zero_terminal_snr(self, alphas_cumprod):
        """
        Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)
        Args:
            betas (`torch.Tensor`):
                the betas that the scheduler is being initialized with.
        Returns:
            `torch.Tensor`: rescaled betas with zero terminal SNR
        """
        alphas_bar_sqrt = alphas_cumprod.sqrt()
        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
        return alphas_bar

    @property
    def sigmas(self):
        return self._sigmas

    @property
    def betas(self):
        return self._betas

    @property
    def alphas(self):
        return self._alphas

    @property
    def timesteps(self):
        return self._timesteps

    # plot the noise sampling map
    def plot_noise_sampling_map(self, save_path):
        y = [
            {"data": self._sigmas.cpu().numpy(), "label": "sigmas"},
            {"data": self._betas.cpu().numpy(), "label": "betas"},
            {"data": self._alphas.cpu().numpy(), "label": "alphas"},
            {"data": self._timesteps.cpu().numpy()/self.num_timesteps, "label": "timesteps"}
            ]
        plot_multi_curves(
            x=self._sample_steps.cpu().numpy(),
            y=y,
            x_label='timesteps',
            y_label=None,
            title=f"{self.__class__.__name__}'s noise sampling map",
            save_path=save_path
        )
        return save_path

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' + ' ' + super().__repr__()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('NOISE_SCHEDULER',
                            __class__.__name__,
                            BaseNoiseScheduler.para_dict,
                            set_name=True)


@NOISE_SCHEDULERS.register_class()
class ScaledLinearScheduler(BaseNoiseScheduler):
    para_dict = {}
    def init_params(self):
        super().init_params()
        self.beta_min = self.cfg.get('BETA_MIN', 0.00085)
        self.beta_max = self.cfg.get('BETA_MAX', 0.012)
        self.snr_shift_scale = self.cfg.get('SNR_SHIFT_SCALE', None)
        self.rescale_betas_zero_snr = self.cfg.get('RESCALE_BETAS_ZERO_SNR', False)

    def square_betas_to_sigmas(self, square_betas, snr_shift_scale=None, rescale_betas_zero_snr=False):
        if snr_shift_scale is not None or rescale_betas_zero_snr:
            alphas_cumprod = torch.cumprod(1 - square_betas, dim=0)
            if snr_shift_scale is not None and snr_shift_scale > 0:
                alphas_cumprod = alphas_cumprod / (snr_shift_scale + (1 - snr_shift_scale) * alphas_cumprod)
            if rescale_betas_zero_snr:
                alphas_cumprod = self.rescale_zero_terminal_snr(alphas_cumprod)
            return torch.sqrt(1 - alphas_cumprod)
        else:
            return torch.sqrt(1 - torch.cumprod(1 - square_betas, dim=0))

    def get_schedule(self):
        square_betas = torch.linspace(self.beta_min**0.5, self.beta_max**0.5, self.num_timesteps, dtype=torch.float32) ** 2
        self._sigmas = self.square_betas_to_sigmas(square_betas, self.snr_shift_scale, self.rescale_betas_zero_snr)
        self._betas = torch.sqrt(square_betas)
        self._alphas = torch.sqrt(1 - self._sigmas ** 2)
        self._timesteps = torch.arange(len(self._sigmas), dtype=torch.float32)


@NOISE_SCHEDULERS.register_class()
class FlowMatchUniformScheduler(BaseNoiseScheduler):
    def get_schedule(self):
        timesteps = np.linspace(1, self.num_timesteps, self.num_timesteps, dtype=np.float32).copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        self._timesteps = timesteps
        self._sigmas = self.t_to_sigma(timesteps)
        self._betas = torch.sqrt(self.sigmas_to_square_betas(self._sigmas))
        self._alphas = torch.sqrt(1 - self.betas ** 2)

    def add_noise(self, x_0, noise = None, t = None):
        if t is None:
            t = torch.rand((x_0.shape[0],), device=x_0.device)
        sigma = self.t_to_sigma(t)
        shape = (x_0.size(0),) + (1,) * (x_0.ndim - 1)
        x_t = (1 - sigma.view(shape)) * x_0 + sigma.view(shape) * noise
        return ScheduleOutput(x_0 = x_0, x_t = x_t, t = t, sigma=sigma, alpha=self.t_to_alpha(t))

    def sigma_to_t(self, sigma, **kwargs):
        return sigma * self.num_timesteps

    def t_to_sigma(self, t, **kwargs):
        return t/self.num_timesteps

    @staticmethod
    def get_config_template():
        return dict_to_yaml('NOISE_SCHEDULER',
                            __class__.__name__,
                            FlowMatchUniformScheduler.para_dict,
                            set_name=True)


@NOISE_SCHEDULERS.register_class()
class FlowMatchSigmoidScheduler(FlowMatchUniformScheduler):
    para_dict = {
        "SIGMOID_SCALE": {
            "value": 1,
            "description": "The scale for the sigmoid function."
        }
    }
    def init_params(self):
        super().init_params()
        self.sigmoid_scale = self.cfg.get("SIGMOID_SCALE", 1)

    def sigma_to_t(self, sigma, **kwargs):
        t = - torch.log(1/sigma - 1)/self.sigmoid_scale
        return t * self.num_timesteps

    def t_to_sigma(self, t, **kwargs):
        return torch.sigmoid(self.sigmoid_scale * t/self.num_timesteps)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('NOISE_SCHEDULER',
                            __class__.__name__,
                            FlowMatchSigmoidScheduler.para_dict,
                            set_name=True)

@NOISE_SCHEDULERS.register_class()
class FlowMatchShiftScheduler(FlowMatchUniformScheduler):
    para_dict = {
        "SHIFT": {
            "value": 3,
            "description": "The shift factor for the timestamp."
        },
        "SIGMOID_SCALE": {
            "value": 1,
            "description": "The scale for the sigmoid function."
        }
    }

    def init_params(self):
        super().init_params()
        self.shift = self.cfg.get("SHIFT", 3)
        self.sigmoid_scale = self.cfg.get("SIGMOID_SCALE", 1)

    def add_noise(self, x_0, noise = None, t = None):
        if t is None:
            logits_norm = torch.randn(x_0.shape[0], device=x_0.device)
            logits_norm = logits_norm * self.sigmoid_scale  # larger scale for more uniform sampling
            t = logits_norm.sigmoid() * self.num_timesteps
        sigma = self.t_to_sigma(t)
        shape = (x_0.size(0),) + (1,) * (x_0.ndim - 1)
        x_t = (1 - sigma.view(shape)) * x_0 + sigma.view(shape) * noise
        return ScheduleOutput(x_0 = x_0, x_t = x_t, t = t, sigma=sigma, alpha=self.t_to_alpha(t))

    def sigma_to_t(self, sigma, **kwargs):
        t = sigma/(sigma - self.shift * sigma + self.shift)
        return t * self.num_timesteps

    def t_to_sigma(self, t, **kwargs):
        t = t / self.num_timesteps
        return  (t * self.shift) / (1 + (self.shift - 1) * t)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('NOISE_SCHEDULER',
                            __class__.__name__,
                            FlowMatchShiftScheduler.para_dict,
                            set_name=True)

@NOISE_SCHEDULERS.register_class()
class FlowMatchFluxShiftScheduler(FlowMatchUniformScheduler):
    para_dict = {
        "SHIFT": {
            "value": True,
            "description": "Use timestamp shift or not, default is True."
        },
        "SIGMOID_SCALE": {
            "value": 1,
            "description": "The scale of sigmoid function for sampling timesteps."
        },
        "BASE_SHIFT": {
            "value": 0.5,
            "description": "The base shift factor for the timestamp."
        },
        "MAX_SHIFT": {
            "value": 1.15,
            "description": "The max shift factor for the timestamp."
        }
    }

    def init_params(self):
        super().init_params()
        self.shift = self.cfg.get("SHIFT", True)
        self.sigmoid_scale = self.cfg.get("SIGMOID_SCALE", 1)
        self.base_shift = self.cfg.get('BASE_SHIFT', 0.5)
        self.max_shift = self.cfg.get('MAX_SHIFT', 1.15)

    def time_shift(self, mu: float, sigma_scale: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma_scale)
    def sigma_shift(self, mu: float, sigma_scale: float, sigma: Tensor):
        return 1/(torch.pow((1-sigma) * math.exp(mu)/sigma, sigma_scale) + 1)

    def get_lin_function(self,
            x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def add_noise(self, x_0, noise = None, t = None):
        if x_0.ndim == 3:
            seq_len = x_0.shape[2] // 4
        else:
            n, _, h, w = x_0.shape
            seq_len = (h // 2 * w // 2)
        if t is None:
            logits_norm = torch.randn(x_0.shape[0], device=x_0.device)
            logits_norm = logits_norm * self.sigmoid_scale  # larger scale for more uniform sampling
            t = logits_norm.sigmoid() * self.num_timesteps
        sigma = self.t_to_sigma(t, seq_len=seq_len)
        shape = (x_0.size(0),) + (1,) * (x_0.ndim - 1)
        x_t = (1 - sigma.view(shape)) * x_0 + sigma.view(shape) * noise
        return ScheduleOutput(x_0 = x_0, x_t = x_t, t = t, sigma=sigma, alpha=self.t_to_alpha(t))

    def sigma_to_t(self, sigma, **kwargs):
        seq_len = kwargs.get('seq_len', 256)
        if self.shift:
            mu = self.get_lin_function(y1=self.base_shift, y2=self.max_shift)(seq_len)
            sigma = self.sigma_shift(mu, 1.0, sigma)
        t = torch.as_tensor(sigma, dtype=torch.float32)
        return t * self.num_timesteps

    def t_to_sigma(self, t, **kwargs):
        seq_len = kwargs.get('seq_len', 256)
        t = t/self.num_timesteps
        if self.shift:
            mu = self.get_lin_function(y1=self.base_shift, y2=self.max_shift)(seq_len)
            t = self.time_shift(mu, 1.0, t)
        sigma = torch.as_tensor(t, dtype=torch.float32)
        return sigma

    @staticmethod
    def get_config_template():
        return dict_to_yaml('NOISE_SCHEDULER',
                            __class__.__name__,
                            FlowMatchFluxShiftScheduler.para_dict,
                            set_name=True)

@NOISE_SCHEDULERS.register_class()
class FlowMatchSigmaScheduler(FlowMatchUniformScheduler):
    para_dict = {
       "WEIGHTING_SCHEME" : {
            "value": "logit_normal",
            "description": "The weighting scheme for sampling timesteps, "
                           "choose from ['sigma_sqrt', 'logit_normal', 'mode', 'cosmap', 'none']."
        },
        "SHIFT": {
            "value": 3.0,
            "description": "The shift factor for the timestamp."
        },
        "LOGIT_MEAN" : {
            "value": 0.0,
            "description": "The mean of the logit distribution for sampling timesteps."
        },
        "LOGIT_STD" : {
            "value": 1.0,
            "description": "The standard deviation of the logit distribution for sampling timesteps."
        },
        "MODE_SCALE" : {
            "value": 1.29,
            "description": "The scale factor for the mode of the logit distribution for sampling timesteps."
        }
    }

    def init_params(self):
        super().init_params()
        self.weighting_scheme = self.cfg.get("WEIGHTING_SCHEME", "logit_normal")
        self.logit_mean = self.cfg.get("LOGIT_MEAN", 0.0)
        self.logit_std = self.cfg.get("LOGIT_STD", 1.0)
        self.mode_scale = self.cfg.get("MODE_SCALE", 1.29)
        self.shift = self.cfg.get("SHIFT", 1.0)

    def get_schedule(self):
        timesteps = np.linspace(1, self.num_timesteps, self.num_timesteps, dtype=np.float32).copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        self._timesteps = timesteps
        timesteps = timesteps / self.num_timesteps
        self._sigmas = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)
        self._betas = torch.sqrt(self.sigmas_to_square_betas(self._sigmas))
        self._alphas = torch.sqrt(1 - self.betas ** 2)

    def add_noise(self, x_0, noise=None, t=None):
        if t is None:
            if self.weighting_scheme == "logit_normal":
                t = torch.normal(mean=self.logit_mean, std=self.logit_std, size=(x_0.shape[0],), device=x_0.device)
            else:
                t = torch.rand(x_0.shape[0], device=x_0.device)
            t = self.compute_density_for_timestep_sampling(t) * self.num_timesteps
        sigma = self.t_to_sigma(t)
        shape = (x_0.size(0),) + (1,) * (x_0.ndim - 1)
        x_t = (1 - sigma.view(shape)) * x_0 + sigma.view(shape) * noise
        return ScheduleOutput(x_0=x_0, x_t=x_t, t=t, sigma=sigma, alpha=self.t_to_alpha(t))

    def compute_density_for_timestep_sampling(self, t):
        """Compute the density for sampling the timesteps when doing SD3 training.
        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.
        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if self.weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            t = torch.nn.functional.sigmoid(t)
        elif self.weighting_scheme == "mode":
            t = 1 - t - self.mode_scale * (torch.cos(math.pi * t / 2) ** 2 - 1 + t)
        return t

    def sigma_to_t(self, sigma, **kwargs):
        raise NotImplementedError

    def t_to_sigma(self, t, **kwargs):
        indices = t.long()
        indices[indices >= self.num_timesteps] = self.num_timesteps - 1
        timesteps = self.timesteps.to(t)[indices]
        step_indices = [(self.timesteps.to(t) == t).nonzero().item() for t in timesteps]
        sigma = self.sigmas[step_indices].flatten().to(t)
        return sigma

    @staticmethod
    def get_config_template():
        return dict_to_yaml('NOISE_SCHEDULER',
                            __class__.__name__,
                            FlowMatchSigmaScheduler.para_dict,
                            set_name=True)


if __name__ == '__main__':
    from scepter.modules.utils.config import Config
    cfg = Config(cfg_dict={
        "NAME": "FlowMatchShiftScheduler",
        "SHIFT": 1.15
    }, load=False)

    scheduler = NOISE_SCHEDULERS.build(cfg)
