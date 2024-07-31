from typing import Optional, Tuple, Any, List, Dict, Union

import numpy as np
import torch
from einops import rearrange

from core.domains.avd.dataset.data_classes import Scene
from core.domains.avd.navigation.pos_nav.pos_planner import AVDPosMDPMeta, AVDPosPlanner
from core.domains.avd.navigation.pose_nav.pose_planner import AVDPoseMDPMeta, AVDPosePlanner
from core.domains.avd.navigation.pos_nav.position_map import get_node_image_names
from core.env import Env
from core.utils.env_utils import NavStatus
from core.utils.image_utils import square_resize
from core.utils.tensor_utils import random_choice
# from povdp.dataset.segmentation import segment
from povdp.dataset.avd import get_3d_observations
# from swin3d.Swin3D.SemanticSeg.model.Swin3D_RGB import Swin3D
# from swin3d.Swin3D.data_utils.indoor3d_util import (
#     create_bev, init_bev, update_bev, get_bev_params, filter_outliers
# )
from povdp.dataset.avd import (
    create_bev, init_bev, update_bev, get_bev_params, filter_outliers
)


class AVDEnv(Env):
    def __init__(self, meta: Union[AVDPosMDPMeta, AVDPoseMDPMeta], *, split=None,
                 min_traj_len=0, max_steps=None, sample_free=None, done_explicit=None, **kwargs):
        self.min_traj_len = min_traj_len
        self.sample_free = sample_free

        self.split = split
        self.scenes: Dict[str, Scene] = meta.scenes.split[split]
        self.scene_name_gen = self.gen_scene_name()

        self.state = self.target = self.opt_traj = None
        self.planner: Union[AVDPosPlanner, AVDPosePlanner] = None

        self.image_obsv_history = list()

        self.done_explicit = done_explicit

        self.target_resize = square_resize((64, 64))
        self.bev_map = None

        super(AVDEnv, self).__init__(meta, max_steps)

    def gen_scene_name(self):
        while True:
            for scene_name in self.scenes:
                yield scene_name

    def _reset_avd(self) -> Tuple[dict, Any, Optional[List[Any]]]:
        """
        :return: tuple of episode_info (dict) and initial observation
        """
        self.opt_traj, self.planner = self._reset_traj()
        traj_states, opt_actions, _ = list(zip(*self.opt_traj))
        self.state = traj_states[0]
        self.target = traj_states[-1]
        values, best_action_maps, counts, motions = self.planner.get_values_and_best_actions(self.target)
        self.image_obsv_history.clear()
        target_object = random_choice(self.planner.target_objects)
        target_rgb = rearrange(torch.from_numpy(target_object.rgb()), "h w f -> f h w")
        
        self.bev_map = None
        
        return {
            'scene_name': self.planner.scene.name,
            'target': torch.tensor(self.meta.state_to_grid_index(self.target)).long(), 
            'targets_grid': torch.from_numpy(self.planner.targets_grid()).bool(), 
            'target_name': self.planner.target_name, 
            'target_rgb': self.target_resize(target_rgb), 
            'target_emb': target_object.embedding(), 
            'values': torch.from_numpy(values).float(),
            # 'target_image': target_object.rgb(),
            'occupancy': self.planner.grid
        }, self.obsv(), opt_actions

    def obsv(self):
        if self.meta.ori_res:
            state_info = self.state.image_name
            self.image_obsv_history.append(state_info)
        else:
            state_info = repr(self.state)
            self.image_obsv_history += get_node_image_names(state_info)
        pose = self.meta.state_to_index(self.state)
        observations = get_3d_observations(
            scene=self.planner.scene, 
            image_names=self.image_obsv_history, 
            target_obj_name=self.planner.target_name
        )
        # segmented_points = segment(
        #     point_cloud=observations['flat_points'], 
        #     model=self.meta.segmentation_model,
        #     args=self.meta.segment_config
        # )
        
        self.bev_map = create_bev(
            # points=segmented_points, 
            points=observations['flat_points'], 
            target_points=observations['flat_target_points'],
            target_grid_size=self.meta.map_res, 
            # visualize_grid=False
        )

        return {
            'state_info': state_info, 
            'poses': torch.tensor(pose).float(), 
            'bev_map': torch.from_numpy(self.bev_map).float()
        }

    def _step(self, action) -> Tuple[Any, float, bool, Any]:
        """
        :param action:
        :return: (obsv, reward, done, info)
        """
        status, reward, done = NavStatus.in_progress, 0, False
        if self.state in self.planner.target_states:
            if action == self.meta.actions.done or not self.done_explicit:
                status, reward, done = NavStatus.success, 1, True
                reward = 1
        if action != self.meta.actions.done:
            trans = self.planner.transition(self.state, action)
            if trans is not None:
                next_state, _, _ = trans
                self.state = next_state
        return self.obsv(), reward, done, {'status': status}

    def _init_planner(self, scene_name) -> Union[AVDPosPlanner, AVDPosePlanner]:
        return AVDPosePlanner(self.meta, scene_name) if self.meta.ori_res else AVDPosPlanner(self.meta, scene_name)

    def _reset_traj(self):
        scene_name = next(self.scene_name_gen)
        trajs = []
        planner = None
        while not trajs:
            planner = self._init_planner(scene_name)
            trajs = planner.sample_trajectories(
                self.min_traj_len, 
                n_trajs=1, 
            )
        return trajs[0], planner
