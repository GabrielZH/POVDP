import torch

from core.models.vin.vin_base import VINBase
from core.models.vin.vin_utils import label_belief
from core.mdp.actions import ActionSetBase


class CALVINBase(VINBase):
    def get_motion_model(self):
        raise NotImplementedError

    def get_available_actions(self, input_view, motion_model, target_map):
        """
        :param input_view: (batch_sz, l_i, *state_shape)
        :return:
            available actions A(s, a): (batch_sz, n_actions, *state_shape)
            available actions logit: (batch_sz, n_actions, *state_shape) or None
            available actions thresh: (batch_sz, n_actions, *state_shape) or None
        """
        raise NotImplementedError

    def get_target_map(self, feature_map, target=None):
        return None

    def get_reward_function(self, feature_map, available_actions, motion_model):
        """
        :param available_actions: (batch_sz, n_actions, *state_shape)
        :param motion_model: (n_actions, *state_shape)
        :param reward_map: (batch_sz, 1, *state_shape)
        :return: reward function R(s, a): (batch_sz, n_actions, *state_shape)
        """
        raise NotImplementedError

    def eval_q(self, available_actions, motion_model, reward, value=None):
        raise NotImplementedError

    def _forward(self, feature_map=None, k=None, prev_v=None, inference=False, target=None, **kwargs):
        """
        :param feature_map: (batch_sz, imsize, imsize)
        :param k: number of iterations. If None, use config.k
        :param prev_v: previously evaluated v (if it exists)
        :return: logits and softmaxed logits
        """
        # get reward map
        motion_model = self.get_motion_model() #.detach()
        # get target map
        target_map = self.get_target_map(feature_map, target=target)
        # get probability of available actions
        aa, aa_logit, aa_thresh, free, free_logit = self.get_available_actions(feature_map, motion_model, target_map)
        # get reward function
        r = self.get_reward_function(feature_map, aa, motion_model)

        q = self.eval_q(aa, motion_model, r, prev_v)  # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        # Update q and v values
        if k is None: k = self.kr if inference and self.kr else self.k
        for i in range(k):
            q = self.eval_q(aa, motion_model, r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        results = {"q": q, "v": v, "prev_v": prev_v if prev_v is not None else torch.zeros_like(v),
                "r_sa": r, "r": r[:, self.actions.done_index],
                "aa": aa, "aa_logit": aa_logit, "aa_thresh": aa_thresh, "mm": motion_model}
        if free is not None:
            results['free'] = free
        if free_logit is not None:
            results['free_logit'] = free_logit
        if target_map is not None:
            results['target_map'] = target_map
        return results


class POCALVINBase(CALVINBase):

    def get_obsv_model(self, obsv):
        """
        param feature_map: (batch_sz, channels, imsize, imsize)
        """
        raise NotImplementedError


    def update_belief(
            self, 
            prev_belief, 
            action, 
            obsv, 
            available_actions=None):
        """
        param prev_belief: (batch_sz, imsize, imsize)
        param action: (batch_sz, n_actions)
        param feature_map: (batch_sz, channels, imsize, imsize)
        """
        raise NotImplementedError


    def _forward(
            self, 
            post_feature_map=None, 
            feature_map=None, 
            curr_poses=None,
            poses=None, 
            curr_actions=None, 
            actions=None, 
            k=None, 
            prev_v=None, 
            inference=False, 
            target=None, 
            **kwargs
    ):
        """
        :param feature_map: (batch_sz, channels, imsize, imsize)
        :param actions: (batch_sz,)
        :param k: number of iterations. If None, use config.k
        :param prev_v: previously evaluated v (if it exists)
        :return: logits and softmaxed logits
        """
        # get reward map
        motion_model = self.get_motion_model() #.detach()
        obsv_model = self.get_obsv_model(post_feature_map)
        # get target map
        target_map = self.get_target_map(feature_map, target=target)
        # get probability of available actions
        aa, aa_logit, aa_thresh, free, free_logit = self.get_available_actions(
            feature_map, 
            motion_model, 
            target_map
        )

        # state estimation
        prev_belief = label_belief(
            poses=curr_poses, 
            belief_shape=post_feature_map[:, 0].size(), 
            belief_labeling=self.belief_labeling
        )
        if curr_actions is not None:
            prev_belief = prev_belief[:, None]
            pred_belief = self.update_belief(
                prev_belief=prev_belief, 
                action=curr_actions, 
                motion_model=motion_model, 
                obsv_model=obsv_model, 
                available_actions=aa
            )
            pred_belief = pred_belief.squeeze(dim=1)
        else:
            pred_belief = prev_belief.copy()

        # get reward function
        r = self.get_reward_function(feature_map, aa, motion_model)
        # value iteration to get optimal value function estimation
        q = self.eval_q(aa, motion_model, r, prev_v)  # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        # Update q and v values
        if k is None: k = self.kr if inference and self.kr else self.k
        for _ in range(k):
            q = self.eval_q(aa, motion_model, r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        results = {
            'q': q, 'v': v, 'prev_v': prev_v if prev_v is not None else torch.zeros_like(v),
            'pred_beliefs': pred_belief, 'r_sa': r, 'r': r[:, self.actions.done_index],
            'aa': aa, 'aa_logit': aa_logit, 'aa_thresh': aa_thresh, 'mm': motion_model
        }
        if free is not None:
            results['free'] = free
        if free_logit is not None:
            results['free_logit'] = free_logit
        if target_map is not None:
            results['target_map'] = target_map
        return results