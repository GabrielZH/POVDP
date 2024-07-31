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
from calvin.core.domains.gridworld.actions import GridDirs, GridActionSet
from calvin.core.models.calvin.calvin_conv2d import POCALVINConv2d


action_list = GridDirs.DIRS_8 + [(0, 0)]

class Parser(utils.Parser):
    dataset: str = 'grid_maze_15x15_vr_2'
    config: str = 'configs.grid_maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)
env_config = json.load(
    open(join(args.dataset_root_path, args.dataset, 'env_config.json')))
factory = get_factory(env_config['domain'])
meta = factory.meta(**env_config)
env = factory.env(meta, **env_config)
env_size = env_config['size']

# ds_path = os.path.join(args.dataset_root_path, args.dataset)
# episode_all, obsv_all, _, trans_all = dataset.grid.get_dataset(
#     ds_path, 
#     dataset_type='val')
# env_map = obsv_all['0']['feature_map'][0, 0].numpy()
# opt_action_indices = trans_all['0']['actions'].numpy()
# opt_actions = [action_list[i] for i in opt_action_indices]


#---------------------------------- loading ----------------------------------#

experiment = utils.load_diffusion_edm(
    args.logbase, 
    args.dataset, 
    args.arch_variation, 
    args.diffusion_loadpath, 
    epoch=args.diffusion_epoch, 
    ema_decay=args.ema_decay,
    schedule_sampler=args.schedule_sampler,
    train_dataset_config='train_dataset_config', 
    val_dataset_config='val_dataset_config',
    model1d_config='edm_model1d_config', 
    policy_config='edm_policy_config',
    trainer_config='edm_policy_trainer_config')

diffusion_policy = experiment.edm_policy
diffusion_policy.diffusion_model.eval()

value_loadpath = os.path.join(
    args.value_loadpath, f'epoch_{args.value_epoch}', 'checkpoint.pt'
)
value_config = json.load(
    open(os.path.join(args.value_loadpath, 'config.json'), 'r')
)
value_config['action_set'] = GridActionSet(env_config['four_way'])
checkpoint = torch.load(value_loadpath)
value_net = POCALVINConv2d(**value_config)
value_net.load_state_dict(checkpoint['model'])

#---------------------------------- main loop ----------------------------------#

true_succ = 0
lenient_succ = 0
max_path_len = 0
n_trials = 5
n_eval = 100

for i in range(n_trials):
    for j in range(n_eval):
        episode_info, init_state, init_obsv, target, opt_actions = (
            output[0] for output in env.reset_grid()
        )
        true_values = episode_info['values'].to('cuda')
        true_values = torch.where(
            torch.isinf(true_values), 
            torch.full_like(true_values, -100), 
            true_values
        )

        print(f"initial position: {init_state}; target: {target}")

        exec_action_history = list()
        # pred_states = list()
        reach_goal = False

        obsv = done = info = None
        # obsv_history = np.tile(
        #     init_obsv_map, 
        #     reps=[args.n_policy_obsv_steps + args.n_latency_steps, 1, 1, 1])

        cond = {
            'env_maps': torch.zeros(
                1, args.n_obsv_steps, args.env_dim, *env_size 
            ).float().to('cuda')
        }

        t_act = 0

        states = list()
        action_candidates = list()
        last_env_map = None
        obsv_history = list()
        while t_act < args.max_episode_steps:
            # print(f"obsv cond shape: {cond['observations'].shape}")
            # print(f"env cond shape: {cond['env_maps'].shape}")
            # cond['observations'] = torch.from_numpy(
            #     train_dataset.normalizers['LimitsNormalizer'](
            #         cond['observations'], 'obsv_maps')).float()[None].to('cuda')
            
            belief = torch.zeros_like(cond['env_maps'][0, 0, 0])[None].to('cuda')
            if not t_act:
                # print(f"true env:\n{init_obsv['feature_map'].numpy()}")
                env_map = init_obsv['feature_map'].float().to('cuda')
                cond['env_maps'][:, -1] = env_map[None]
                state = init_obsv['poses'].int()
                belief[:, state[0], state[1]] = 1.
            else:
                # print(f"true env:\n{obsv['feature_map'].numpy()}")
                if args.n_action_steps >= args.n_obsv_steps:
                    cond['env_maps'] = obsv_history[-args.n_obsv_steps:]
                else:
                    cond['env_maps'] = torch.cat(
                        [
                            cond['env_maps'][:, args.n_action_steps:],
                            obsv_history
                        ], 
                        dim=1
                    )
                state = obsv['poses'].int()
                belief[:, state[0], state[1]] = 1.

            result = diffusion_policy(cond)
            exe_actions = result['exe_actions'][0]
            pred_actions = result['pred_actions'][0]
            
            # print(f"values:\n{true_values}")

            best_candidate = None
            best_value = -torch.inf
            for n in range(20):
                result = diffusion_policy(cond)
                # print(f"actions: {result['action']}\npred_actions: {result['action_pred']}")
                
                exe_actions = result['exe_actions'][0]
                pred_actions = result['pred_actions'][0]
                
                sim_env = deepcopy(env)
                value_sum = 0.
                sim_belief = belief.clone()
                try:
                    for a_idx in pred_actions:
                        # print(f"simulated belief:\n{sim_belief}")
                        weighted_value = (true_values * sim_belief).sum(dim=(1, 2))
                        value_sum += weighted_value
                        sim_obsv, _, _, _ = sim_env.step(action_list[a_idx])
                        sim_state = sim_obsv['poses'].int()
                        sim_belief = torch.zeros_like(belief).to('cuda')
                        sim_belief[:, sim_state[0], sim_state[1]] = 1.
                except IndexError:
                    continue
                print(f"pred actions: {[action_list[a] for a in pred_actions]} expected value: {value_sum}")

                if value_sum > best_value:
                    best_value = value_sum.clone()
                    best_candidate = pred_actions

            action_candidates.append([action_list[a] for a in best_candidate])
            print(f"best pred actions:\n{[action_list[a] for a in best_candidate]}")

            obsv_history.clear()
            for act_i in best_candidate[:args.n_action_steps]:
                act = action_list[act_i]
                exec_action_history.append(act)
                obsv, _, done, info = env.step(act)
                states.append(obsv['poses'])
                last_env_map = obsv['feature_map']
                obsv_history.append(obsv['feature_map'][None])

                if np.array_equal(
                    obsv['poses'].numpy().astype(int), target):
                        if not reach_goal:
                            reach_goal = True
                            lenient_succ += 1

                if done:
                    break

                t_act += 1

            if done:
                print(f"The task finished in {t_act + 1} steps.")
                if not reach_goal:
                    print(f"last env map:\n{last_env_map}")
                    print(f"state history: {states}")
                    print(f"expert actions: {opt_actions}")
                    print(f"executed actions: {exec_action_history}")
                    print(f"best action candidate history:")
                    for act in action_candidates:
                        print(act)
                # elif 
                else:
                    true_succ += 1
                break

            obsv_history = torch.stack(obsv_history, dim=1).to('cuda')

        print(f"-- Trial {i + 1} Eval {j + 1} Result --")
        print(f"reach-goal rate: {lenient_succ / (n_eval * i + j + 1)}")
        print(f"success rate: {true_succ / (n_eval * i + j + 1)}")
            
    print(f"Trial {i + 1} Complete.")
    print("-- Overall Result --")
    print(f"reach-goal rate: {lenient_succ / (n_eval * (i + 1))}")
    print(f"success rate: {true_succ / (n_eval * (i + 1))}")
