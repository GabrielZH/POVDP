import math
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import pdb


#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_inpainting_conditioning_dict_indexing(x, conditions):
    for t, val in conditions.items():
        # x[:, t, action_dim:] = val.clone()
        x[:, t] = val.clone()
    return x

def apply_inpainting_conditioning_masking(x, condition, mask):
    x[mask] = condition[mask]
    return x

def apply_inpainting_conditioning_soft_masking(x, condition, mask):
    return x * (1. - mask) + condition * mask


#-----------------------------------------------------------------------------#
#-------------------------------- conditions ---------------------------------#
#-----------------------------------------------------------------------------#
def get_goal_belief(unnormed_env_map):
    goal_mask = unnormed_env_map[:, -1]
    explored_mask = torch.sum(unnormed_env_map[:, :2], dim=1)
    unexplored_mask = 1. - explored_mask
    belief = unexplored_mask + 1000. * goal_mask
    belief = belief / (belief.sum(dim=(1, 2), keepdim=True) + 1e-8)
    return belief


#-----------------------------------------------------------------------------#
#----------------------------- position encoding -----------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class SinusoidalPosEmb2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert not dim % 4, "target dimension must be a multiple of 4"
        self.dim = dim

    def forward(self, inp, height, width):
        device = inp.device
        quarter_dim = self.dim // 4
        emb = math.log(10000) / (quarter_dim - 1)

        h, w = torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device)
        )
        h = h.flatten().float()
        w = w.flatten().float()

        emb_h = torch.exp(torch.arange(quarter_dim, device=device) * -emb)
        emb_h = h[:, None] * emb[None]
        emb_h = torch.cat((emb_h.sin(), emb_h.cos()), dim=1)

        emb_w = torch.exp(torch.arange(quarter_dim, device=device) * -emb)
        emb_w = w[:, None] * emb[None]
        emb_w = torch.cat((emb_w.sin(), emb_w.cos()), dim=1)

        emb = torch.cat((emb_h, emb_w), dim=1).reshape(-1, height, width)
        emb = emb.unsqueeze(0).expand(inp.shape[0])
        emb = torch.cat((emb, inp), dim=1)

        return emb

#-----------------------------------------------------------------------------#
#------------------------------ mask generators ------------------------------#
#-----------------------------------------------------------------------------#

class LowdimMaskGenerator(nn.Module):
    def __init__(
            self,
            action_dim, 
            obsv_dim,
            # obs mask setup
            max_n_obsv_steps=2, 
            fix_obsv_steps=True, 
            # action mask
            action_visible=False
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obsv_dim = obsv_dim
        self.max_n_obsv_steps = max_n_obsv_steps
        self.fix_obsv_steps = fix_obsv_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        B, T, D = shape
        assert D == (self.action_dim + self.obsv_dim)

        # create all tensors on this device
        rng = torch.Generator()
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(
            size=shape, 
            dtype=torch.bool)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obsv_dim = ~is_action_dim

        # generate obsv mask
        if self.fix_obsv_steps:
            obsv_steps = torch.full((B,), 
            fill_value=self.max_n_obsv_steps)
        else:
            obsv_steps = torch.randint(
                low=1, 
                high=self.max_n_obsv_steps+1, 
                size=(B,), 
                generator=rng
            )
            
        steps = torch.arange(0, T).reshape(1,T).expand(B,T)
        obsv_mask = (steps.T < obsv_steps).T.reshape(B,T,1).expand(B,T,D)
        obsv_mask = obsv_mask & is_obsv_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obsv_steps - 1, 
                torch.tensor(0, dtype=obsv_steps.dtype))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obsv_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask


#-----------------------------------------------------------------------------#
#------------------------------ normalizations -------------------------------#
#-----------------------------------------------------------------------------#

def int2bits(x, n_bits=8):
    # Convert integers into the corresponding binary bits.
    x = (x.unsqueeze(-1) >> torch.arange(n_bits).to('cuda:0')) & 1
    return x * 2 - 1


def bits2int(x):
    # Convert binary bits into the corresponding integers.
    n_bits = x.shape[-1]
    x = torch.sum(x * (2 ** torch.arange(n_bits).to('cuda:0')), -1)
    return x.to(torch.int64)


def double_sigmoid(x, a=.25, b=.75, k=10):
    sigmoid = nn.Sigmoid()
    return sigmoid(k * (x - a)) * (1 - sigmoid(k * (x - b)))
    

def normalize_env(x):
    x_min = torch.amin(x, (2, 3))
    x_max = torch.amax(x, (2, 3))
    x = (x - x_min) / (x_max - x_min)
    return x * 2 - 1


def unnormalize_env(x, eps=1e-4):
    return (x + 1) / 2

#-----------------------------------------------------------------------------#
#------------------------ input/ouput manipulation ---------------------------#
#-----------------------------------------------------------------------------#

def unzip_trajectory(x, image_embedding_generator):
    x_actions, x_observations, x_states = zip(*x)
    
    fn = lambda z: torch.stack(z, dim=1)
    x_actions, x_observations, x_states = (
        fn(z) for z in (x_actions, x_observations, x_states)
    )
    batch_size = x_actions.shape[0]
    n_bits = x_actions.shape[-1]
    x_observations = rearrange(x_observations, 'b h c x y -> (b h) c x y')
    x_observations = image_embedding_generator(x_observations)
    x_observations = x_observations.reshape(batch_size, -1, n_bits)

    return torch.concat([x_actions, x_observations, x_states], dim=2)


def unzip_single_env_trajectory(x):
    x_actions, x_states = zip(*x)

    fn = lambda z: torch.stack(z, dim=1)
    x_actions, x_states = (
        fn(z) for z in (x_actions, x_states)
    )

    return torch.concat([x_actions, x_states], dim=-1)


def unzip_condition(observation, state, image_embedding_generator):
    observation = rearrange(observation, 'b h c x y -> (b h) c x y')
    observation = image_embedding_generator(observation)
    return observation, state
    

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def FiLM_scale1d(x, embed):
    """
    """
    assert x.shape[1] * 2 == embed.shape[1]  # channels
    embed = embed.reshape(x.shape[0], 2, x.shape[1], 1)
    scale = embed[:, 0, ...]
    bias = embed[:, 1, ...]
    return scale * x + bias

def FiLM_scale2d(x, embed):
    """
    """
    assert x.shape[1] * 2 == embed.shape[1]  # channels
    embed = embed.reshape(x.shape[0], 2, x.shape[1], 1, 1)
    scale = embed[:, 0, ...]
    bias = embed[:, 1, ...]
    return scale * x + bias