from typing import Union, List, Set, Tuple, Optional
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import pdb

# import diffuser.datasets.d4rl as d4rl
import povdp.dataset.grid as grid
import povdp.dataset.avd as avd
from povdp.dataset.normalization import DatasetNormalizer
from povdp.dataset.diffusion.buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
ConditionalBatch = namedtuple('ConditionalBatch', 'actions conditions')
ConditionalBeliefBatch = namedtuple(
    'ConditionalBeliefBatch', 'actions normed_actions beliefs conditions')


class DiscreteSequenceDataset(torch.utils.data.Dataset):

    def __init__(
            self,  
            map_res: Tuple[int] = (15, 15),
            # ori_res: int = None, 
            # resize: Optional[Tuple[int]] = None, 
            # target: str = None, 
            # target_size_ratio: Optional[float] = None, 
            # pcd_available=False, 
            # in_ram: Optional[bool] = None, 
            horizon: int = None, 
            n_action_steps: int = None, 
            n_obsv_steps: int = None, 
            n_bits: int = None, 
            normalizers: Set[str] = set(), 
            ds_path=None,
            ds_type='train',
            preprocess_fns=[], 
            max_path_length: int = None, 
            max_n_episodes: int = 10000, 
            termination_penalty: float = None,
            use_max_len_padding=False, 
            use_condition_padding=False, 
    ):
        self.preprocess_fn = None
        self.collate_fn = None

        self.map_res = map_res
        # self.ori_res = ori_res

        self.ds_type = ds_type
        
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.n_action_steps = n_action_steps
        self.n_obsv_steps = n_obsv_steps
        self.use_max_len_padding = use_max_len_padding
        self.use_condition_padding = use_condition_padding

        if 'grid' in ds_path:
            itr = grid.sequence_dataset(
                ds_path=ds_path, 
                preprocess_fn=self.preprocess_fn, 
                ds_type=ds_type
            )
            # self.collate_fn = grid.collate_fn
            fields = ReplayBuffer(
                max_n_episodes=max_n_episodes, 
                max_path_length=max_path_length, 
                n_action_steps=n_action_steps, 
                n_obsv_steps=n_obsv_steps,
                use_max_len_padding=use_max_len_padding, 
                use_condition_padding=use_condition_padding,
                termination_penalty=termination_penalty
            )
        elif 'avd' in ds_path:
            itr = avd.sequence_dataset(
                ds_path=ds_path, 
                preprocess_fn=self.preprocess_fn, 
                ds_type=ds_type, 
                # resize=resize, 
                # target=target, 
                # target_size_ratio=target_size_ratio, 
                # ori_res=ori_res, 
                # in_ram=in_ram
            )
            # self.collate_fn = avd.collate_fn
            fields = ReplayBuffer(
                max_n_episodes=max_n_episodes, 
                max_path_length=max_path_length, 
                n_action_steps=n_action_steps, 
                n_obsv_steps=n_obsv_steps,
                use_max_len_padding=use_max_len_padding, 
                use_condition_padding=use_condition_padding,
                termination_penalty=termination_penalty, 
            )
        # elif 'mp3d' in ds_path:
        #     itr = mp3d.sequence_dataset(
        #         ds_path=ds_path, 
        #         preprocess_fns=self.preprocess_fn, 
        #         ds_type=ds_type, 
        #         resize=resize, 
        #         target=target, 
        #         target_size_ratio=target_size_ratio, 
        #         pcd_available=pcd_available, 
        #         in_ram=in_ram
        #     )
        #     self.collate_fn = mp3d.collate_fn
        #     fields = ReplayBuffer(
        #         max_n_episodes=max_n_episodes, 
        #         max_path_length=max_path_length, 
        #         n_obs_steps=n_obsv_steps, 
        #         n_action_steps=n_action_steps, 
        #         use_max_len_padding=use_max_len_padding, 
        #         use_condition_padding=use_condition_padding,
        #         termination_penalty=termination_penalty
        #     )

        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizers = dict()
        for normalizer in normalizers:
            self.normalizers[normalizer] = DatasetNormalizer(
                dataset=fields, 
                normalizer=normalizer, 
                path_lengths=fields['path_lengths'], 
                n_bits=n_bits)

        self.indices = self.make_indices(
            fields.path_lengths, 
            horizon, 
            # fix_len=('grid' in ds_path)
        )
        self.action_dim = n_bits

        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        if 'grid' in ds_path:
            self.goal_reveal_indices = fields.goal_reveal_indices

        self.normalize(
            normalizer=self.normalizers['BitNormalizer'], 
            keys=['actions'])
        # if 'grid' in ds_path:
        #     self.normalize(
        #         normalizer=self.normalizers['LimitsNormalizer'], 
        #         keys=['feature_maps'])

        print(self.fields)
        
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, normalizer, keys=['actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            print(f"key: {key}")
            array = self.fields[key].reshape(
                self.n_episodes*self.max_path_length, *self.fields[key].shape[2:])
                
            if key == 'states':
                if array.shape[-1] > 1:
                    # only support pos coord (x,y) for now
                    assert array.shape[-1] == 2
                    array = np.ravel_multi_index(array.transpose(), self.map_res)
                    array = np.expand_dims(array, axis=-1)
            normed = normalizer(array.astype(np.int32), key)

            if normalizer.normalizer_name == 'BitNormalizer':
                self.fields[f'normed_{key}'] = normed.reshape(*self.fields[key].shape[:-1], -1)
            else:
                self.fields[f'normed_{key}'] = normed.reshape(self.fields[key].shape)

    def make_indices(
            self, 
            path_lengths, 
            horizon, 
            # fix_len=True
        ):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_max_len_padding:
                max_start = min(max_start, path_length - horizon)
            # if fix_len:
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
            # else:
            #     for start in range(max_start):
            #         for end in range(start + 1, path_length - horizon + 1):
            #             indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
    

class PartiallyObservableConditionalDiscreteDataset(DiscreteSequenceDataset):

    def __init__(self, ds_path=None, **kwargs):
        super().__init__(
            ds_path=ds_path, 
            **kwargs
        )
        self.ds_path = ds_path
        self.obsv_dim = self.fields.feature_maps.shape[2]  # image n_channels

    def get_conditions(self, **kwargs):
        return {**kwargs}
    
    def __getitem__(self, idx, eps=0.0001):
        path_ind, start, end = self.indices[idx]

        if self.ds_type == 'train':
            if 'grid' in self.ds_path:
                if random.random() < .1:
                    start = 0
                    end = self.horizon
                elif .9 <= random.random() < 1.:
                    start = self.goal_reveal_indices[path_ind]
                    end = start + self.horizon

        if 'grid' in self.ds_path:
            normed_actions = self.fields.normed_actions[path_ind, start:end]
            feature_maps = self.fields.feature_maps[path_ind, start:start+self.n_obsv_steps]
            conditions = self.get_conditions(
                env_maps=feature_maps,
            )
        elif 'avd' in self.ds_path:
            normed_actions = self.fields.normed_actions[path_ind, start:end]
            # target = self.fields.target[path_ind].astype(np.int32)
            # target_emb = self.fields.target_emb[path_ind]
            # target_grid = self.fields.target_grid[path_ind].astype(bool)
            # occupancy = self.fields.occupancy[path_ind]
            feature_maps = self.fields.feature_maps[path_ind, start:start+self.n_obsv_steps]
            # rgb = self.fields.rgb[path_ind, start:end]
            # emb = self.fields.emb[path_ind, start:end]
            # valid_points = self.fields.valid_points[path_ind, start:end].astype(bool)
            # surf_xyz = self.fields.surf_xyz[path_ind, start:end].astype(np.float64)

            conditions = self.get_conditions(
                # target=target, 
                # target_emb=target_emb, 
                # target_grid=target_grid, 
                # occupancy=occupancy, 
                env_maps=feature_maps,
                # rgb=rgb, 
                # emb=emb, 
                # valid_points=valid_points, 
                # surf_xyz=surf_xyz
            )
        else:
            raise NotImplementedError()
        # return ConditionalBatch(actions, normed_actions, conditions)
        # if normed_actions.shape[0] != self.horizon:
        #     print(f'start: {start}, end: {end}, total len: {len(self.fields.normed_actions[path_ind])}')
        #     print(f"normed_actions.shape: {normed_actions.shape}")
        #     raise RuntimeError("Error in __getitem__. Aborting...")
        return ConditionalBatch(normed_actions, conditions)
