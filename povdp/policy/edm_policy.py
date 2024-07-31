from typing import Optional
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import pdb

from povdp.networks.diffusion_model import (
    AttentionConditionalUnet1d, 
    AttentionConditionalUnet2d, 
)
from povdp.networks.projection.point_cloud_to_2d_projector import (
    PointCloudTo2dProjector
)
from povdp.networks.helpers import (
    LowdimMaskGenerator, 
    bits2int
)
from povdp.networks.losses import (
    PolicyLosses, 
    SSIMLoss
)
from povdp.utils import (
    Progress, Silent
)
from povdp.utils.tensor_utils import (
    mean_flat, 
    append_dims
)
from povdp.utils.random_utils import (
    get_generator
)


torch.set_printoptions(threshold=sys.maxsize)


class ConditionalEDMPolicy(nn.Module):
    def __init__(
            self, 
            diffusion_model: AttentionConditionalUnet1d, 
            point_cloud_to_2d_projector: Optional[PointCloudTo2dProjector], 
            horizon: int, 
            action_dim: int, 
            n_action_steps: int, 
            n_obsv_steps: int, 
            pred_action_steps_only=False, 
            n_timesteps:int = 40, 
            sigma_data: float = .5, 
            sigma_max: float = 80., 
            sigma_min: float = .002, 
            rho: float = 7., 
            s_churn: float = 0., 
            s_tmin: float = 0., 
            s_tmax: float = float('inf'), 
            s_noise: float = 1., 
            weight_schedule='edm', 
            clip_denoised=False, 
            loss_norm='l2',
            bit_scale: float = 1.,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obsv_steps = n_obsv_steps
        self.pred_action_steps_only = pred_action_steps_only

        self.diffusion_model = diffusion_model
        self.point_cloud_to_2d_projector = point_cloud_to_2d_projector

        self.n_timesteps = n_timesteps
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho

        self.s_churn = s_churn, 
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

        self.weight_schedule = weight_schedule
        self.loss_norm = loss_norm

        self.clip_denoised = clip_denoised

        self.bit_scale = bit_scale

    def get_weights(self, sigma):
        snr = sigma ** -2.
        if self.weight_schedule == 'snr':
            weights = snr
        elif self.weight_schedule == 'snr+1':
            weights = snr + 1.
        elif self.weight_schedule == 'edm':
            weights = snr + 1. / self.sigma_data ** 2
        elif self.weight_schedule == 'truncated-snr':
            weights = torch.clamp(snr, min=1.)
        elif self.weight_schedule == 'uniform':
            weights = torch.ones_like(snr)
        else:
            raise NotImplementedError()
        return weights

    def get_scales(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** .5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** .5
        return c_skip, c_out, c_in

    def get_scales_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** .5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** .5
        return c_skip, c_out, c_in
    
    def get_cond_args(self, cond):
        local_cond = global_cond = None
        if 'env_maps' in cond.keys():
            global_cond = cond['env_maps'].float()
        else:
            keys = ['rgb', 'emb', 'valid_points', 'surf_xyz']
            if all(key in cond.keys() for key in keys):
                assert self.point_cloud_to_2d_projector is not None
                outputs = self.point_cloud_to_2d_projector(
                    rgb=cond['rgb'], 
                    emb=cond['emb'], 
                    surf_xyz=cond['surf_xyz'], 
                    valid_points=cond['valid_points'], 
                    index=cond['index'], 
                    target_emb=cond['target_emb'], 
                    batch_size=cond['target_emb'].shape[0], 
                )
                # global_cond = outputs['feature_map']
                global_cond = outputs.clone()

        return local_cond, global_cond

    def loss(
            self, 
            x_start, 
            sigma, 
            cond, 
            noise=None
        ):
        if noise is None:
            noise = torch.randn_like(
                x_start.float(), 
                device=x_start.device
            )
        local_cond, global_cond = self.get_cond_args(cond)
        x_start = x_start.float() * self.bit_scale
        x_t = x_start + noise * append_dims(sigma, x_start.ndim)

        _, denoised = self.denoise(
            x_t=x_t, 
            sigma=sigma, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        assert not torch.isnan(denoised).any()

        weights = append_dims(
            self.get_weights(sigma), x_start.ndim
        )

        loss_dict = dict()
        loss_dict["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        loss_dict["mse"] = mean_flat(
            weights * (denoised - x_start) ** 2
        )

        if "vb" in loss_dict:
            loss = loss_dict["mse"] + loss_dict["vb"]
        else:
            loss = loss_dict["mse"]

        return loss, loss_dict
    
    def denoise(
            self, 
            x_t, 
            sigma, 
            local_cond=None, 
            global_cond=None, 
        ):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scales(sigma)
        ]
        rescaled_t = 1000 * .25 + torch.log(sigma + 1e-4)

        model_output = self.diffusion_model(
            sample=c_in * x_t, 
            timestep=rescaled_t, 
            local_cond=local_cond, 
            global_cond=global_cond,
        )
        denoised = c_out * model_output + c_skip * x_t

        return model_output, denoised
    
    def get_sigma(self, device='cuda'):
        ramp = torch.linspace(0, 1, self.n_timesteps)
        min_inv_rho = self.sigma_min ** (1. / self.rho)
        max_inv_rho = self.sigma_max ** (1. / self.rho)
        sigma_seq = (
            max_inv_rho + ramp * (min_inv_rho - max_inv_rho)
        ) ** self.rho
        return torch.cat(
            [sigma_seq, sigma_seq.new_zeros([1])]).to(device)
    
    @staticmethod
    def get_ode_derivative(x, denoised, sigma):
        return (x - denoised) / append_dims(sigma, x.ndim)
    
    @torch.no_grad()
    def heun_sample_loop(
            self, 
            x, 
            sigma_seq, 
            generator, 
            local_cond, 
            global_cond, 
    ):
        s_in = x.new_ones([x.shape[0]])
        for i, sigma in enumerate(sigma_seq[:-1]):
            if isinstance(self.s_churn, tuple):
                self.s_churn = self.s_churn[0]
            gamma = (
                min(self.s_churn / (len(sigma_seq) - 1), 2 ** .5 - 1)
                if self.s_tmin <= sigma <= self.s_tmax
                else 0.
            )
            epsilon = generator.randn_like(x) * self.s_noise
            sigma_hat = sigma * (gamma + 1)
            
            if gamma > 0:
                x += epsilon * (sigma_hat**2 - sigma**2) ** .5
            
            _, denoised = self.denoise(
                x_t=x, 
                sigma=sigma_hat * s_in, 
                local_cond=local_cond, 
                global_cond=global_cond 
            )
            if self.clip_denoised:
                denoised = denoised.clamp(-1., 1.)
            d = self.get_ode_derivative(
                x=x, 
                denoised=denoised, 
                sigma=sigma_hat
            )
            dt = sigma_seq[i + 1] - sigma_hat
            if not sigma_seq[i + 1]:  ## Euler method
                x += d * dt
            else:  ## Heun method
                x_2 = x + d * dt
                _, denoised_2 = self.denoise(
                    x_t=x_2, 
                    sigma=sigma_seq[i + 1] * s_in, 
                    local_cond=local_cond, 
                    global_cond=global_cond
                )
                if self.clip_denoised:
                    denoised_2 = denoised_2.clamp(-1., 1.)
                d_2 = self.get_ode_derivative(
                    x=x_2, 
                    denoised=denoised_2, 
                    sigma=sigma_seq[i + 1]
                )
                d_prime = (d + d_2) / 2
                x += d_prime * dt

        return x
    
    @torch.no_grad()
    def conditional_sample(
            self, 
            cond, 
            generator=None, 
            **kwargs
        ):
        local_cond, global_cond = self.get_cond_args(cond)

        if generator is None:
            generator = get_generator('dummy')
        x_T = generator.randn(
            size=(
                global_cond.shape[0], 
                self.horizon, 
                self.action_dim
            ), 
            device=global_cond.device) * self.sigma_max
        
        sigma_seq = self.get_sigma(device=x_T.device)

        normed_pred_actions = self.heun_sample_loop(
            x=x_T, 
            sigma_seq=sigma_seq, 
            generator=generator, 
            local_cond=local_cond, 
            global_cond=global_cond, 
        ).clamp(-1., 1.)

        # unnormalize prediction
        pred_actions = bits2int(normed_pred_actions > 0)

        # get action
        if self.pred_action_steps_only:
            exe_actions = pred_actions
        else:
            exe_actions = pred_actions[:, :self.n_action_steps]
        
        plan = {
            'exe_actions': exe_actions,
            'pred_actions': pred_actions
        }

        return plan
    
    def forward(
            self, 
            cond, 
            *args, 
            **kwargs
        ):
        return self.conditional_sample(
            cond=cond, 
            *args, 
            **kwargs
        )