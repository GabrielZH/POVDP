import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import einops


class ValidActionFilter(nn.Module):
    def __init__(
            self, 
            inp_dim, 
            out_dim,
            hidden_dim_valid_action_filter=128,  
            dropout=0., 
            **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp_dim, hidden_dim_valid_action_filter, 3, padding=1, bias=True), 
            nn.Dropout(dropout), 
            nn.Mish(), 
            nn.Conv2d(hidden_dim_valid_action_filter, out_dim, 1, bias=True), 
        )

    def forward(self, x):
        return self.block(x)
    
class RewardFunction(nn.Module):
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            hidden_dim_reward_fn=128, 
            dropout=0., 
            **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp_dim, hidden_dim_reward_fn, 3, padding=1, bias=True), 
            nn.Dropout(dropout), 
            nn.Mish(), 
            nn.Conv2d(hidden_dim_reward_fn, out_dim, 1, bias=True), 
        )

    def forward(self, x):
        return self.block(x)


class ValueFunction(nn.Module):
    def __init__(
            self, 
            env_dim, 
            n_actions, 
            n_vi, 
            gamma=.99, 
            epsilon=.05,
            motion_scale=10, 
            device='cuda',
            **kwargs):
        super().__init__()

        self.device = device

        self.n_vi = n_vi
        self.gamma = gamma
        self.epsilon = epsilon

        self.env_dim = env_dim
        self.n_actions = n_actions

        self.motion_scale = motion_scale
        self.w_trans_motion = Parameter(
            torch.randn(n_actions, 1, 3, 3, device=device), 
            requires_grad=True
        )

        self.valid_action_filter = ValidActionFilter(
            inp_dim=env_dim, 
            out_dim=n_actions + 1,
            **kwargs
        )

        self.reward_function = RewardFunction(
            inp_dim=env_dim, 
            out_dim=n_actions, 
            **kwargs
        )

        self.r_failure = Parameter(torch.zeros((1,), device=device), requires_grad=True)
        self.r_trans_motion = Parameter(
            torch.zeros(n_actions, 1, 3, 3, device=device), 
            requires_grad=True)
        self.r_unexplored = Parameter(torch.zeros((1,), device=device), requires_grad=True)

    def get_w_motion(self):
        return self.w_trans_motion * self.motion_scale
    
    def get_trans_motion_fn(self):
        w_motion = self.get_w_motion()
        flattened = w_motion.reshape(self.n_actions, -1)
        trans_motion_fn = F.softmax(
            flattened, dim=-1).reshape(w_motion.shape)
        return trans_motion_fn
    
    def get_valid_actions(self, env_map):
        """
        As mentioned in CALVIN, there are:
        action_logit = action_filter_output[:, :-1]
        action_thresh = action_filter_output[:, -1:]
        By thresholding the logit, we determine which action(s)
        to be excluded due to invalidity.
        """
        output = self.valid_action_filter(env_map)
        action_logit = output[:, :-1]
        action_thresh = output[:, -1:]
        valid_action = torch.sigmoid(action_logit - action_thresh)
        return valid_action, action_logit, action_thresh
    
    def get_reward_fn(self, valid_actions, env_map):
        reward_failure = - F.softplus(self.r_failure)
        # reward_motion = einops.reduce(
        #     trans_motion_fn * self.r_trans_motion, 
        #     'a () x y -> () a () ()', 'sum', x=3, y=3)
        reward_motion = self.reward_function(env_map)
        return reward_failure * (1 - valid_actions) + reward_motion * valid_actions
    
    def estimate_q(self, valid_actions, trans_motion_fn, reward, value):
        q = reward.clone()
        if value is not None:
            next_value = F.conv2d(
                value, 
                trans_motion_fn, 
                None, 1, 'same')
            q += self.gamma * next_value * valid_actions

        return q

    def value_iteration(self, env_map, value=None, n_vi=None):
        """
        Value iteration
        """
        trans_motion_fn = self.get_trans_motion_fn()
        valid_actions, action_logit, action_thresh = self.get_valid_actions(env_map)
        reward_fn = self.get_reward_fn(
            valid_actions=valid_actions, 
            env_map=env_map
        )
        # mask = torch.sum(env_map[:, :2], dim=1, keepdim=True)
        # reward_fn = reward_fn * mask + self.r_unexplored * (1 - mask)
        if value is not None:
            value = value.clone().detach()
        q = self.estimate_q(
            valid_actions=valid_actions, 
            trans_motion_fn=trans_motion_fn, 
            reward=reward_fn, 
            value=value
        )
        v, _ = torch.max(q, dim=1, keepdim=True)

        if n_vi is None: n_vi = self.n_vi
        for _ in range(n_vi):
            q = self.estimate_q(
                valid_actions=valid_actions, 
                trans_motion_fn=trans_motion_fn, 
                reward=reward_fn, 
                value=v
            )
            v, _ = torch.max(q, dim=1, keepdim=True)

        return {
            'q_fn': q, 
            'v_fn': v, 
            'prev_v_fn': value if value is not None else torch.zeros_like(v), 
            'rewards': reward_fn, 
            'valid_actions': valid_actions, 
            'action_logit': action_logit, 
            'action_shreshold': action_thresh, 
            'trans_motion_fn': trans_motion_fn,
        }
    
    def action_value_loss(self, action_value, expert_action, belief):
        v_a = (action_value * belief[:, None]).sum(dim=(2, 3))
        return F.cross_entropy(v_a, expert_action.squeeze(dim=-1).long())
    
    def seq_action_value_loss(self, action_value_seq, action_label_seq, belief_seq):
        v_a = (action_value_seq * belief_seq[:, None]).sum(dim=(3, 4))
        return F.cross_entropy(v_a, action_label_seq.squeeze(dim=-1).long())
    
    def state_value_loss(self, state_value, env_map):
        mask = torch.sum(env_map[:, :2], dim=1, keepdim=True).bool()
        inp = state_value[mask]
        target = env_map[:, 1:2][mask]
        return F.mse_loss(torch.sigmoid(inp), target)

    def loss(self, q_fn, action_logit, expert_action, belief):
        loss_q_fn = self.action_value_loss(
            action_value=q_fn, 
            expert_action=expert_action, 
            belief=belief)
        loss_valid_actions = self.action_value_loss(
            action_value=action_logit, 
            expert_action=expert_action, 
            belief=belief)
        return loss_q_fn + loss_valid_actions
    
    def seq_loss(self, q_fns, action_logits, action_labels, beliefs):
        loss_q_fns = self.seq_action_value_loss(
            action_value_seq=q_fns, 
            action_label_seq=action_labels, 
            belief_seq=beliefs)
        loss_valid_actions = self.seq_action_value_loss(
            action_value_seq=action_logits, 
            action_label_seq=action_labels, 
            belief_seq=beliefs)
        return loss_q_fns + loss_valid_actions
    
    def forward(
            self, 
            env_map, 
            value=None, 
            n_vi=None):
        value_info = self.value_iteration(
            env_map=env_map, 
            value=value, 
            n_vi=n_vi)

        return value_info


    
