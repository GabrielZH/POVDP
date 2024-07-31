import sys

import numpy as np
import torch


np.set_printoptions(threshold=sys.maxsize)


def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x


class ReplayBuffer:

    def __init__(
        self, 
        max_n_episodes, 
        max_path_length, 
        n_action_steps,
        n_obsv_steps, 
        use_max_len_padding=False, 
        use_condition_padding=False,
        termination_penalty=None,
    ):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=int), 
            'goal_reveal_indices': np.zeros(max_n_episodes, dtype=int), 
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.n_action_steps = n_action_steps
        self.n_obsv_steps = n_obsv_steps
        self.use_max_len_padding = use_max_len_padding
        self.use_condition_padding = use_condition_padding
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict.get(key)

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {
            k: v for k, v in self._dict.items()
            if k not in ['path_lengths', 'goal_reveal_indices']
        }.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        # dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, *(array.shape[1:]))
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}'

    def _pad_path(self, path, padding):
        for key, value in path.items():
            if key == 'actions':
                padding_before = torch.tensor([-1]).repeat(padding[key]['before'])
                padding_after = torch.tensor([8]).repeat(padding[key]['after'])
            elif key in [
                'states', 'next_states', 
                'poses', 'next_poses'
            ]:
                padding_before = torch.tensor(value[0]).repeat(padding[key]['before'], 1)
                padding_after = torch.tensor(value[-1]).repeat(padding[key]['after'], 1)
            # elif key in [
            #     'state_info', 'next_state_info'
            # ]:
            #     padding_before = [value[0]] * padding[key]['before'] if padding[key]['before'] else []
            #     padding_after = [value[-1]] * padding[key]['after'] if padding[key]['after'] else []
            elif key == 'rewards':
                padding_before = torch.tensor([0.]).repeat(padding[key]['before'])
                padding_after = torch.tensor([0.]).repeat(padding[key]['after'])
            elif key in [ 
                'rgb', 'next_rgb', 
                'emb', 'next_emb', 
                'surf_xyz', 'next_surf_xyz'
            ]:
                padding_before = torch.tensor(value[0]).repeat(padding[key]['before'], 1, 1, 1)
                padding_after = torch.tensor(value[-1]).repeat(padding[key]['after'], 1, 1, 1)
            elif key in [
                'valid_points', 'next_valid_points', 
            ]:
                padding_before = torch.tensor(value[0]).repeat(padding[key]['before'], 1, 1)
                padding_after = torch.tensor(value[-1]).repeat(padding[key]['after'], 1, 1)
            elif key in [
                'feature_maps', 'next_feature_maps'
            ]:
                padding_before = torch.zeros_like(value[0]).repeat(padding[key]['before'], 1, 1, 1)
                padding_after = torch.tensor(value[-1]).repeat(padding[key]['after'], 1, 1, 1)
            elif key in [
                'beliefs', 'next_beliefs'
            ]:
                padding_before = torch.zeros_like(value[0]).repeat(padding[key]['before'], 1, 1)
                padding_after = torch.tensor(value[-1]).repeat(padding[key]['after'], 1, 1)

            # if key in [
            #     'state_info', 
            #     'next_state_info'
            # ]:
            #     path[key] = padding_before + value + padding_after
            if key in [
                'actions', 'rewards', 'dones', 
                'feature_maps', 'next_feature_maps',
                'states', 'next_states', 
                'poses', 'next_poses', 
                'rgb', 'next_rgb', 
                'emb', 'next_emb', 
                'valid_points', 'next_valid_points', 
                'surf_xyz', 'next_surf_xyz'
            ]:
                path[key] = torch.cat(
                    (padding_before, value, padding_after), dim=0)
            
        return path

    def add_path(self, path):
        keys = list(path.keys())
        padding = {
            key: {'before': 0, 'after': 0} for key in keys
        }
        if self.use_max_len_padding:
            path_length = self.max_path_length
            if self.use_condition_padding:
                for key in [
                    'actions', 
                    'rewards', 
                    'dones', 
                    'beliefs', 
                    'next_beliefs'
                ]:
                    if key in keys:
                        padding[key]['before'] = 0
                        padding[key]['after'] = path_length - len(path[key]) - padding[key]['before']
                for key in [
                    'states', 'next_states', 
                    'poses', 'next_poses', 
                    # 'state_info', 'next_state_info', 
                    'rgb', 'next_rgb', 
                    'emb', 'next_emb', 
                    'valid_points', 'next_valid_points', 
                    'surf_xyz', 'next_surf_xyz'
                ]:
                    if key in keys:
                        padding[key]['before'] = 0
                        padding[key]['after'] = path_length - len(path[key]) - padding[key]['before']
                for key in [
                    'feature_maps', 
                    'next_feature_maps'
                ]:
                    if key in keys:
                        padding[key]['before'] = self.n_obsv_steps - 1
                        padding[key]['after'] = path_length - len(path[key]) - padding[key]['before']
                for key in keys:
                    assert padding[key]['after'] >= 0, \
                        f"Episode after padding larger than max length.\n"\
                        f" Key: {key}\n"\
                        f" Episode length before padding: {len(path[key])}\n{path[key]}."
                path = self._pad_path(path, padding)
            else:
                for key in keys:
                    padding[key]['after'] = path_length - len(path[key])
                path = self._pad_path(path, padding)
        elif self.use_condition_padding:
            max_path_len = -1
            for key in [
                'actions', 
                'rewards', 
                'beliefs', 
                'next_beliefs'
            ]:
                if key in keys:
                    padding[key]['before'] = 0
                    padding[key]['after'] = self.n_action_steps - 1
                    path_len = padding[key]['before'] + len(path[key]) + padding[key]['after']
                    if path_len > max_path_len: max_path_len = path_len

            for key in [
                'states', 'next_states', 
                'poses', 'next_poses', 
                # 'state_info', 'next_state_info', 
                'rgb', 'next_rgb', 
                'emb', 'next_emb', 
                'valid_points', 'next_valid_points', 
                'surf_xyz', 'next_surf_xyz'
            ]:
                if key in keys:
                    padding[key]['before'] = 0
                    path_len = padding[key]['before'] + len(path[key]) + padding[key]['after']
                    if path_len > max_path_len: max_path_len = path_len

            for key in [
                'feature_maps', 
                'next_feature_maps'
            ]:
                if key in keys:
                    padding[key]['before'] = 0
                    path_len = padding[key]['before'] + len(path[key]) + padding[key]['after']
                    if path_len > max_path_len: max_path_len = path_len
            
            for key in keys:
                path_len = padding[key]['before'] + len(path[key]) + padding[key]['after']
                if path_len < max_path_len:
                    padding[key]['after'] += max_path_len - path_len

            assert max_path_len <= self.max_path_length

            path = self._pad_path(path, padding)
            path_length = max_path_len
        else:
            path_length = len(path['actions'])

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            if key in [
                'target', 
                'target_emb', 
                'target_grid', 
                'occupancy'
            ]:
                if key not in self._dict:
                    shape = (self.max_n_episodes, *(path[key].shape))
                    self._dict[key] = np.zeros(shape, dtype=np.float32)
                self._dict[key][self._count] = path[key]
            else:
                array = atleast_2d(path[key])
                if key not in self._dict: self._allocate(key, array)
                try:
                    self._dict[key][self._count, :path_length] = array
                except IndexError:
                    print("key:", key)
                    print("count:", self._count)
                    print("dict[key] shape:", self._dict[key].shape)
                    assert False
                except ValueError:
                    print("key:", key)
                    print("count:", self._count)
                    print("dict[key] shape:", self._dict[key].shape)
                    assert False

        # ## penalize early termination
        # if path['terminals'].any() and self.termination_penalty is not None:
        #     assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
        #     self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        if 'feature_maps' in keys:
            for idx, obsv in enumerate(path['feature_maps']):
                if idx and not torch.equal(obsv[2], path['feature_maps'][idx - 1, 2]):
                    self._dict['goal_reveal_indices'][self._count] = idx

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths'] + ['goal_reveal_indices']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')
