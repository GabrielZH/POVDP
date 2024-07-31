import os
import sys
import json
from copy import deepcopy
import numpy as np
from os.path import join
import pdb

import torch

import povdp.dataset as dataset
import povdp.utils as utils

from calvin.core.domains.factory import get_factory
from calvin.core.domains.gridworld.actions import GridDirs
from exploration import frontier_exploration


action_list = GridDirs.DIRS_8 + [(0, 0)]

class Parser(utils.Parser):
    dataset: str = 'grid_maze_15x15_vr_2'
    config: str = 'configs.grid_maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

env_config = json.load(
    open(join(args.dataset_root_path, args.dataset, 'env_config.json')))
factory = get_factory(env_config['domain'])
meta = factory.meta(**env_config)
env = factory.env(meta, **env_config)
env_size = env_config['size']

ds_path = os.path.join(args.dataset_root_path, args.dataset)
episode_all, obsv_all, _, trans_all = dataset.grid.get_dataset(
    ds_path, 
    dataset_type='val')
env_map = obsv_all['0']['feature_map'][0, 0].numpy()
opt_action_indices = trans_all['0']['actions'].numpy()
opt_actions = [action_list[i] for i in opt_action_indices]

#---------------------------------- main loop ----------------------------------#

true_succ = 0
lenient_succ = 0
max_path_len = 0
n_trials = 5
n_eval = 100
max_path = expert_path = feature_map = start = goal = None

for i in range(n_trials):
    for j in range(n_eval):
        print(f"Trial {i+1} Eval {j+1}")
        episode_info, init_state, init_obsv, target, opt_actions = (
            output[0] for output in env.reset_grid()
        )
        exec_action_history = list()
        reach_goal = False
        obsv = done = info = None
        states = list()
        t = 0

        while t < args.max_episode_steps:
            if not t:
                # print(f"true env:\n{init_obsv['feature_map'].numpy()}")
                obsv = init_obsv['feature_map'].float().numpy()
                state = init_obsv['poses'].int().numpy()
            else:
                # print(f"true env:\n{obsv['feature_map'].numpy()}")
                true_env = obsv['feature_map'].float().numpy()
                state = obsv['poses'].int().numpy()

            action_seq, pred_state = frontier_exploration(obsv, tuple(state) if not t else tuple(pred_state))

            for pred_action in action_seq:
                exec_action_history.append(pred_action)
                obsv, _, done, info = env.step(pred_action)
                states.append(obsv['poses'].numpy())

                if np.array_equal(
                    obsv['poses'].numpy().astype(int), target):
                        if not reach_goal:
                            reach_goal = True
                            lenient_succ += 1
                            print(f"Reached goal.")
                            break

                if done:
                    if reach_goal:
                        true_succ += 1
                    break

        print(f"-- Trial {i + 1} Eval {j + 1} Result --")
        print(f"reach-goal rate: {lenient_succ / (n_eval * i + j + 1)}")
        print(f"success rate: {true_succ / (n_eval * i + j + 1)}")
            
    print(f"Trial {i + 1} Complete.")
    print("-- Overall Result --")
    print(f"reach-goal rate: {lenient_succ / (n_eval * (i + 1))}")
    print(f"success rate: {true_succ / (n_eval * (i + 1))}")
