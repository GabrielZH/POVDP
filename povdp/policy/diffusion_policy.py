from typing import Union, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import pdb

from povdp.networks.diffusion_model import (
    ConditionalDiffusionUnet1d
)
from povdp.networks.vision import (
    EnvMapEncoder, 
    MultiGridObsvEncoder, 
    MultiImageObsEncoder,
)
from povdp.networks.helpers import (
    cosine_beta_schedule,
    extract,
    bits2int,
    unnormalize_env,
    get_goal_belief, 
    unzip_single_env_trajectory,
    LowdimMaskGenerator,
)
from povdp.networks.losses import PolicyLosses
import povdp.utils as utils


torch.set_printoptions(threshold=10000)


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


class ConditionalGaussianDiffusionPolicy(nn.Module):
    def __init__(
            self, 
            diffusion_model: ConditionalDiffusionUnet1d,
            env_map_encoder: EnvMapEncoder,
            multi_step_obs_encoder: Optional[
                Union[MultiGridObsvEncoder, 
                      MultiImageObsEncoder]
            ],
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_policy_obs_steps,
            n_env_recon_obs_steps, 
            env_cond_only=True, 
            obs_as_cond_type='global', 
            n_timesteps=1000, 
            loss_type='l1', 
            clip_denoised=False, 
            predict_epsilon=True, 
            pred_action_steps_only=False, 
            bit_scale=1., 
            action_weight=1., 
            loss_discount=1., 
            loss_weights=None, 
        ):
        super().__init__()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_policy_obs_steps = n_policy_obs_steps
        self.n_env_recon_obs_steps = n_env_recon_obs_steps
        self.max_obs_steps = max(n_policy_obs_steps, n_env_recon_obs_steps)
        self.env_cond_only = env_cond_only
        self.obs_as_local_cond = self.obs_as_global_cond = False
        if not env_cond_only:
            assert multi_step_obs_encoder is not None
            if obs_as_cond_type == 'local':
                self.obs_as_local_cond = True
            elif obs_as_cond_type == 'global':
                self.obs_as_global_cond = True

        self.diffusion_model = diffusion_model
        self.env_map_encoder = env_map_encoder
        self.multi_step_obs_encoder = multi_step_obs_encoder
        
        self.pred_action_steps_only = pred_action_steps_only
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if \
                (self.env_cond_only or \
                 self.obs_as_local_cond or \
                    self.obs_as_global_cond) else self.obs_dim,
            max_n_obs_steps=max(n_policy_obs_steps, n_env_recon_obs_steps), 
            fix_obs_steps=True, 
            action_visible=False,
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.bit_scale = bit_scale

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(
            action_weight=action_weight, 
            discount=loss_discount, 
            weights_dict=loss_weights)
        self.loss_fn = PolicyLosses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, local_cond=None, global_cond=None):
        x_recon = self.predict_start_from_noise(
            x_t=x, 
            t=t, 
            noise=self.diffusion_model(
                sample=x, 
                timestep=t, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
        )

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, local_cond=None, global_cond=None):
        b, *_, = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, 
            t=t, 
            local_cond=local_cond, 
            global_cond=global_cond)
        
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,  
        cond_data,
        local_cond=None,
        global_cond=None, 
        verbose=True, 
        return_diffusion=False
    ):
        """
        """
        batch_size = cond_data.shape[0]
        x = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype, 
            device=cond_data.device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timestep = torch.full(
                size=(batch_size,), 
                fill_value=i, 
                dtype=torch.long, 
                device=x.device)
            x = self.p_sample(
                x=x, 
                t=timestep, 
                local_cond=local_cond, 
                global_cond=global_cond)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        return x

    @torch.no_grad()
    def conditional_sample(
        self, 
        cond, 
        *args, 
        **kwargs
    ):
        """
        cond : [ 
            ('prev_obsvs', latest_obs), 
            ..., 
            ('curr_env', curr_feature_map), -> reconstructed
        ]
        The `env_cond` for predicting a plan (actions for the future steps)
        is the feature map corresponding to the current time-step.

        In the training phase, the feature map is the ground-truth at time tt 
        (using tt to distinguish it from diffusion step t) that is provided by 
        the training data. `p_losses` tackles this part.

        In the inference phase, the feature map is the reconstructed `env_map`
        from `env_reconstructor` at time tt. This method tackles this part.

        """
        device = self.betas.device
        env_cond = cond['env_maps'][:, self.n_env_recon_obs_steps].float()
        goal_belief = get_goal_belief(unnormalize_env(env_cond))[:, None]
        global_cond = self.env_map_encoder(
            torch.cat((env_cond, goal_belief), dim=1)
        )
        # global_cond = self.env_map_encoder(env_cond)
        local_cond = None

        if not self.env_cond_only:
            obs_cond = cond['observations'].float()
            dtype = obs_cond.dtype

            if self.obs_as_local_cond:
                # condition through local feature
                # all zero except first To timesteps
                local_cond = torch.zeros(
                    size=(obs_cond.shape[0], self.horizon, *obs_cond.shape[2:]), 
                    device=device, 
                    dtype=dtype
                )
                local_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps] = \
                    obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]
            elif self.obs_as_global_cond:
                obs_cond = obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]  # B,T,C,H,W
                obs_cond_emb = self.multi_step_obs_encoder(obs_cond)
                global_cond = torch.cat([global_cond, obs_cond_emb], dim=-1)
            else:
                raise ValueError(
                    "Observations should be either local condition or global condition")
            
        cond_data = torch.zeros(
            size=(global_cond.shape[0], self.n_action_steps, self.action_dim), 
            device=device, 
            dtype=global_cond.dtype
        ) if self.pred_action_steps_only else torch.zeros(
            size=(global_cond.shape[0], self.horizon, self.action_dim), 
            device=device, 
            dtype=global_cond.dtype
        )
        # cond_data[:, 0] = cond['a0'].clone()

        # run sampling
        normed_action_pred = self.p_sample_loop(
            cond_data=cond_data, 
            *args,
            local_cond=local_cond, 
            global_cond=global_cond, 
            **kwargs
        )
        
        # unnormalize prediction
        action_pred = bits2int(normed_action_pred > 0)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_policy_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result
    
    @torch.no_grad()
    def value_conditional_sample(
        self, 
        cond, 
        *args, 
        **kwargs
    ):
        """
        cond : [ 
            ('prev_obsvs', latest_obs), 
            ..., 
            ('curr_env', curr_feature_map), -> reconstructed
        ]
        The `env_cond` for predicting a plan (actions for the future steps)
        is the feature map corresponding to the current time-step.

        In the training phase, the feature map is the ground-truth at time tt 
        (using tt to distinguish it from diffusion step t) that is provided by 
        the training data. `p_losses` tackles this part.

        In the inference phase, the feature map is the reconstructed `env_map`
        from `env_reconstructor` at time tt. This method tackles this part.

        """
        device = self.betas.device
        env_cond = cond['env_maps'][:, self.n_env_recon_obs_steps].float()
        value_cond = cond['q_fn']
        global_cond = self.env_map_encoder(
            torch.cat((env_cond, value_cond), dim=1)
        )
        local_cond = None

        if not self.env_cond_only:
            obs_cond = cond['observations'].float()
            dtype = obs_cond.dtype

            if self.obs_as_local_cond:
                # condition through local feature
                # all zero except first To timesteps
                local_cond = torch.zeros(
                    size=(obs_cond.shape[0], self.horizon, *obs_cond.shape[2:]), 
                    device=device, 
                    dtype=dtype
                )
                local_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps] = \
                    obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]
            elif self.obs_as_global_cond:
                obs_cond = obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]  # B,T,C,H,W
                obs_cond_emb = self.multi_step_obs_encoder(obs_cond)
                global_cond = torch.cat([global_cond, obs_cond_emb], dim=-1)
            else:
                raise ValueError(
                    "Observations should be either local condition or global condition")
            
        cond_data = torch.zeros(
            size=(global_cond.shape[0], self.n_action_steps, self.action_dim), 
            device=device, 
            dtype=global_cond.dtype
        ) if self.pred_action_steps_only else torch.zeros(
            size=(global_cond.shape[0], self.horizon, self.action_dim), 
            device=device, 
            dtype=global_cond.dtype
        )
        # cond_data[:, 0] = cond['a0'].clone()

        # run sampling
        normed_action_pred = self.p_sample_loop(
            cond_data=cond_data, 
            *args,
            local_cond=local_cond, 
            global_cond=global_cond, 
            **kwargs
        )
        
        # unnormalize prediction
        action_pred = bits2int(normed_action_pred > 0)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_policy_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result
    
    @torch.no_grad()
    def value_guided_sample(
        self, 
        cond, 
        *args, 
        **kwargs
    ):
        """
        cond : [ 
            ('prev_obsvs', latest_obs), 
            ..., 
            ('curr_env', curr_feature_map), -> reconstructed
        ]
        The `env_cond` for predicting a plan (actions for the future steps)
        is the feature map corresponding to the current time-step.

        In the training phase, the feature map is the ground-truth at time tt 
        (using tt to distinguish it from diffusion step t) that is provided by 
        the training data. `p_losses` tackles this part.

        In the inference phase, the feature map is the reconstructed `env_map`
        from `env_reconstructor` at time tt. This method tackles this part.

        """
        device = self.betas.device
        env_cond = cond['env_maps'][:, self.n_env_recon_obs_steps].float()
        global_cond = self.env_map_encoder(env_cond)
        local_cond = None

        if not self.env_cond_only:
            obs_cond = cond['observations'].float()
            dtype = obs_cond.dtype

            if self.obs_as_local_cond:
                # condition through local feature
                # all zero except first To timesteps
                local_cond = torch.zeros(
                    size=(obs_cond.shape[0], self.horizon, *obs_cond.shape[2:]), 
                    device=device, 
                    dtype=dtype
                )
                local_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps] = \
                    obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]
            elif self.obs_as_global_cond:
                obs_cond = obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]  # B,T,C,H,W
                obs_cond_emb = self.multi_step_obs_encoder(obs_cond)
                global_cond = torch.cat([global_cond, obs_cond_emb], dim=-1)
            else:
                raise ValueError(
                    "Observations should be either local condition or global condition")
            
        cond_data = torch.zeros(
            size=(global_cond.shape[0], self.n_action_steps, self.action_dim), 
            device=device, 
            dtype=global_cond.dtype
        ) if self.pred_action_steps_only else torch.zeros(
            size=(global_cond.shape[0], self.horizon, self.action_dim), 
            device=device, 
            dtype=global_cond.dtype
        )
        # cond_data[:, 0] = cond['a0'].clone()

        # run sampling
        normed_action_pred = self.p_sample_loop(
            cond_data=cond_data, 
            *args,
            local_cond=local_cond, 
            global_cond=global_cond, 
            **kwargs
        )
        
        # unnormalize prediction
        action_pred = bits2int(normed_action_pred > 0)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_policy_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    def p_losses(self, x_start, cond, t):
        """
        x_start: action trajectory (plan)
        cond : [ 
            ('prev_obsvs', latest_obs), 
            ...,
            ('curr_env', curr_feature_map), -> ground-truth
        ]
        The `env_cond` for predicting a plan (actions for the future steps)
        is the feature map corresponding to the current time-step.

        In the training phase, the feature map is the ground-truth at time tt 
        (using tt to distinguish it from diffusion step t) that is provided by 
        the training data. This method tackles this part.

        In the inference phase, the feature map is the reconstructed `env_map`
        from `env_reconstructor` at time tt. `conditional_sample` tackles this
        part.
        """
        env_cond = cond['env_maps'][:, self.n_env_recon_obs_steps]
        goal_belief = get_goal_belief(unnormalize_env(env_cond))[:, None]
        global_cond = self.env_map_encoder(
            torch.cat((env_cond, goal_belief), dim=1)
        )
        local_cond = None

        if not self.env_cond_only:
            obs_cond = cond['observations'].float()

            if self.obs_as_local_cond:
                # zero out observations after `n_obs_steps`
                local_cond = obs_cond.clone()
                local_cond[:, :self.max_obs_steps - self.n_policy_obs_steps] = 0
                local_cond[:, self.max_obs_steps:] = 0
            elif self.obs_as_global_cond:
                obs_cond = obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]
                obs_cond_emb = self.multi_step_obs_encoder(obs_cond)
                global_cond = torch.cat([global_cond, obs_cond_emb], dim=-1)
                if self.pred_action_steps_only:
                    start = self.n_policy_obs_steps
                    end = start + self.n_action_steps
                    x_start = x_start[:, start:end, :]
            else:
                raise ValueError(
                    "Observations should be either local condition or global condition")

        noise = torch.randn_like(x_start, device=x_start.device)

        # generate impainting mask
        if self.pred_action_steps_only:
            cond_mask = torch.zeros_like(x_start, dtype=torch.bool).to(x_start.device)
        else:
            cond_mask = self.mask_generator(x_start.shape).to(x_start.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.diffusion_model(
            sample=x_noisy, 
            timestep=t, 
            local_cond=local_cond, 
            global_cond=global_cond)

        assert noise.shape == x_recon.shape

        loss_mask = ~cond_mask
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, mask=loss_mask)
        else:
            loss, info = self.loss_fn(x_recon, x_start, mask=loss_mask)

        return loss, info

    def value_conditional_p_losses(self, x_start, cond, t):
        """
        x_start: action trajectory (plan)
        cond : [ 
            ('prev_obsvs', latest_obs), 
            ...,
            ('curr_env', curr_feature_map), -> ground-truth
        ]
        The `env_cond` for predicting a plan (actions for the future steps)
        is the feature map corresponding to the current time-step.

        In the training phase, the feature map is the ground-truth at time tt 
        (using tt to distinguish it from diffusion step t) that is provided by 
        the training data. This method tackles this part.

        In the inference phase, the feature map is the reconstructed `env_map`
        from `env_reconstructor` at time tt. `conditional_sample` tackles this
        part.
        """
        env_cond = cond['env_maps'][:, self.n_env_recon_obs_steps]
        value_cond = cond['q_fn']
        global_cond = self.env_map_encoder(
            torch.cat((env_cond, value_cond), dim=1))
        local_cond = None

        if not self.env_cond_only:
            obs_cond = cond['observations'].float()

            if self.obs_as_local_cond:
                # zero out observations after `n_obs_steps`
                local_cond = obs_cond.clone()
                local_cond[:, :self.max_obs_steps - self.n_policy_obs_steps] = 0
                local_cond[:, self.max_obs_steps:] = 0
            elif self.obs_as_global_cond:
                obs_cond = obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]
                obs_cond_emb = self.multi_step_obs_encoder(obs_cond)
                global_cond = torch.cat([global_cond, obs_cond_emb], dim=-1)
                if self.pred_action_steps_only:
                    start = self.n_policy_obs_steps
                    end = start + self.n_action_steps
                    x_start = x_start[:, start:end, :]
            else:
                raise ValueError(
                    "Observations should be either local condition or global condition")

        noise = torch.randn_like(x_start, device=x_start.device)

        # generate impainting mask
        if self.pred_action_steps_only:
            cond_mask = torch.zeros_like(x_start, dtype=torch.bool).to(x_start.device)
        else:
            cond_mask = self.mask_generator(x_start.shape).to(x_start.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.diffusion_model(
            sample=x_noisy, 
            timestep=t, 
            local_cond=local_cond, 
            global_cond=global_cond)

        assert noise.shape == x_recon.shape

        loss_mask = ~cond_mask
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, mask=loss_mask)
        else:
            loss, info = self.loss_fn(x_recon, x_start, mask=loss_mask)

        return loss, info
    
    def value_guided_p_losses(self, x_start, cond, t):
        """
        x_start: action trajectory (plan)
        cond : [ 
            ('prev_obsvs', latest_obs), 
            ...,
            ('curr_env', curr_feature_map), -> ground-truth
        ]
        The `env_cond` for predicting a plan (actions for the future steps)
        is the feature map corresponding to the current time-step.

        In the training phase, the feature map is the ground-truth at time tt 
        (using tt to distinguish it from diffusion step t) that is provided by 
        the training data. This method tackles this part.

        In the inference phase, the feature map is the reconstructed `env_map`
        from `env_reconstructor` at time tt. `conditional_sample` tackles this
        part.
        """
        env_cond = cond['env_maps'][:, self.n_env_recon_obs_steps]
        global_cond = self.env_map_encoder(env_cond)
        local_cond = None

        if not self.env_cond_only:
            obs_cond = cond['observations'].float()

            if self.obs_as_local_cond:
                # zero out observations after `n_obs_steps`
                local_cond = obs_cond.clone()
                local_cond[:, :self.max_obs_steps - self.n_policy_obs_steps] = 0
                local_cond[:, self.max_obs_steps:] = 0
            elif self.obs_as_global_cond:
                obs_cond = obs_cond[:, self.max_obs_steps - self.n_policy_obs_steps:self.max_obs_steps]
                obs_cond_emb = self.multi_step_obs_encoder(obs_cond)
                global_cond = torch.cat([global_cond, obs_cond_emb], dim=-1)
                if self.pred_action_steps_only:
                    start = self.n_policy_obs_steps
                    end = start + self.n_action_steps
                    x_start = x_start[:, start:end, :]
            else:
                raise ValueError(
                    "Observations should be either local condition or global condition")

        noise = torch.randn_like(x_start, device=x_start.device)

        # generate impainting mask
        # if self.pred_action_steps_only:
        #     cond_mask = torch.zeros_like(x_start, dtype=torch.bool).to(x_start.device)
        # else:
        #     cond_mask = self.mask_generator(x_start.shape).to(x_start.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.diffusion_model(
            sample=x_noisy, 
            timestep=t, 
            local_cond=local_cond, 
            global_cond=global_cond)

        assert noise.shape == x_recon.shape

        # loss_mask = ~cond_mask
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        if type(x) is not torch.Tensor:
            assert type(x) is list or type(x) is tuple
            x = unzip_single_env_trajectory(x)

        batch_size = len(x)
        t = torch.randint(
            0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x * self.bit_scale, cond, t)
        # return self.value_guided_p_losses(x * self.bit_scale, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
        # return self.value_guided_sample(cond=cond, *args, **kwargs)

