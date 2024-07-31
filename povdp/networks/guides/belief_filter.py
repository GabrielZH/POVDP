import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# def check_tensor(tensor, name):
#     if (tensor != tensor).any():  # Checks for NaNs
#         print(f"NaN value in {name}")
#     if (tensor == float('inf')).any() or (tensor == float('-inf')).any():  # Checks for infs
#         print(f"Infinite value in {name}")


class LocalObservationEncoder(nn.Module):
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            hidden_dim=128, 
            **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp_dim, hidden_dim, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(hidden_dim, out_dim, 3, padding=1), 
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(out_dim, out_dim), 
            nn.Softmax(),
        )

    def forward(self, obsv):
        return self.block(obsv)
    

class ObservationFunction(nn.Module):
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            hidden_dim=128, 
            dropout=0., 
            **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp_dim, hidden_dim, 3, padding=1), 
            # nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, out_dim, 1), 
            nn.Sigmoid(), 
        )

    def forward(self, env_map):
        return self.block(env_map)


class BeliefFilter(nn.Module):
    def __init__(
            self, 
            env_dim, 
            obs_dim, 
            n_observations, 
            n_actions, 
            motion_scale=10, 
            device='cuda', 
            **kwargs):
        super().__init__()
        
        self.device = device

        self.env_dim = env_dim
        self.n_actions = n_actions

        self.motion_scale = motion_scale
        self.w_trans_motion = Parameter(
            torch.randn(n_actions, 1, 3, 3, device=device), 
            requires_grad=True
        )

        self.local_obsv_encoder = LocalObservationEncoder(
            inp_dim=obs_dim, 
            out_dim=n_observations
        )

        self.obsv_fn = ObservationFunction(
            inp_dim=env_dim, 
            out_dim=n_observations, 
            **kwargs
        )

    def get_w_motion(self):
        return self.w_trans_motion * self.motion_scale
    
    def get_trans_motion_fn(self):
        w_motion = self.get_w_motion()
        flattened = w_motion.reshape(self.n_actions, -1)
        trans_motion_fn = F.softmax(
            flattened, dim=-1).reshape(w_motion.shape)
        return trans_motion_fn
    
    def get_obsv_fn(self, env_map):
        obsv_fn = self.obsv_fn(env_map)
        obsv_fn = obsv_fn / (obsv_fn.sum(dim=1, keepdim=True) + 1e-8)
        return obsv_fn
    
    def encode_observations(self, obsv):
        return self.local_obsv_encoder(obsv)
    
    def encode_actions(self, action):
        return F.one_hot(action, self.n_actions)
    
    def belief_update(
            self, 
            belief, 
            action, 
            obsv, 
            valid_actions, 
            env_map):
        belief = belief[:, None]
        ## prediction
        trans_motion_fn = self.get_trans_motion_fn()
        next_beliefs = F.conv2d(
            belief, 
            trans_motion_fn, 
            None, 1, 'same')
        next_beliefs = next_beliefs * valid_actions
        action_embed = self.encode_actions(action.to(torch.int64)).squeeze(dim=1)[:, :, None, None]
        next_belief = (next_beliefs * action_embed).sum(dim=1)
        
        ## correction
        obsv_fn = self.get_obsv_fn(env_map)
        obsv_embed = self.encode_observations(obsv)[:, :, None, None]
        # check_tensor(next_belief, "next_belief")

        post_pr_obsv = (obsv_fn * obsv_embed).sum(dim=1)
        # check_tensor(post_pr_obsv, "post_pr_obsv")
        next_belief = next_belief * post_pr_obsv
        # check_tensor(next_belief, "next_belief after multiplication")
        next_belief = next_belief / (next_belief.sum(dim=(1, 2), keepdim=True) + 1e-8)

        return next_belief
    
    def goal_belief_update(self, env_map):
        mask = torch.sum(env_map[:, :2], dim=1)
        belief = mask * env_map[:, -1] * 100 + (1 - mask)
        belief = belief / (belief.sum() + 1e-8)
        return belief
    
    def forward(
            self, 
            belief, 
            action, 
            obsv, 
            valid_actions,
            env_map):
        belief = self.belief_update(
            belief=belief, 
            action=action, 
            obsv=obsv, 
            valid_actions=valid_actions, 
            env_map=env_map
        )
        # TODO: whether we need belief over goal positions?
        # goal_belief = self.goal_belief_update(env_map)
        # next_belief = torch.stack((motion_belief, goal_belief), dim=1)

        return belief
