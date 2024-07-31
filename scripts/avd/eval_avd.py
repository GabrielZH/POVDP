import os
import sys
import json
from copy import deepcopy
import numpy as np
import pdb

import torch
from einops import rearrange

import povdp.dataset as dataset
import povdp.utils as utils

from calvin.core.domains.factory import get_factory
from calvin.core.domains.gridworld.actions import GridDirs, GridActionSet
from calvin.core.domains.avd.navigation.pose_nav.actions import AVDPoseActionSet
from calvin.core.domains.avd.navigation.pos_nav.actions import AVDPosActionSet
from calvin.core.domains.avd.navigation.pos_nav.position_map import PositionNode
from calvin.core.domains.avd.dataset.scene_manager import AVDSceneManager
from calvin.core.domains.avd.dataset.data_classes import Scene
# from visualize import visualize

action_list = GridDirs.DIRS_8 + [(0, 0)]

class Parser(utils.Parser):
    dataset: str = 'avd_pos_nav'
    config: str = 'configs.avd'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)
experiment_params = (
    f"{args.n_episodes}_{'_'.join(map(str, args.map_bbox))}"
    f"__{'_'.join(map(str, args.map_res))}__{args.ori_res or 'pos'}"
    f"_{'_'.join(map(str, args.resize))}_{args.target}"
    f"_{args.min_traj_len}_{args.max_steps}_{args.sample_free}")
env_config = json.load(
    open(os.path.join(
        args.dataset_root_path, 
        args.dataset, 
        experiment_params, 
        'env_config.json'
    ))
)
env_config = {**env_config, 'split': args.split}
factory = get_factory(env_config['domain'])
meta = factory.meta(**env_config)
env = factory.env(meta, **env_config)
env_size = env_config['map_res']
# scenes = AVDSceneManager(
#     data_dir=os.path.join(
#         args.dataset_root_path, 'avd/src'), 
#     scene_resize=args.resize, 
#     target=args.target, 
#     in_ram=args.in_ram, 
#     target_size_ratio=args.target_size_ratio, 
#     avd_workers=4
# )
print(f"Finished preparing scenes.")

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

#---------------------------------- main loop ----------------------------------#

true_succ = 0
lenient_succ = 0
max_path_len = 0
n_trials = 5
n_eval = 10

trial_res = list()
for i in range(n_trials):
    for j in range(n_eval):
        episode_info, init_obsv, opt_actions = (
            output for output in env.reset_avd()
        )
        scene_name = episode_info['scene_name']
        targets_grid = episode_info['targets_grid']
        target_coords = torch.nonzero(targets_grid).tolist()
        # scene: Scene = scenes[scene_name]
        true_values = episode_info['values'].to('cuda')
        true_values = torch.where(
            torch.isinf(true_values),
            torch.full_like(true_values, -100),
            true_values
        )
        true_values = torch.where(
            true_values == float('inf'),
            torch.full_like(true_values, 100),
            true_values
        )

        print(
            f"initial pose: {init_obsv['poses']}; target: {episode_info['target']}")

        exec_action_history = list()
        # pred_states = list()
        reach_goal = False
        obsv = done = info = None
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
                env_map = init_obsv['bev_map'].float().to('cuda')
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
            curr_pos_map = env.state.pos_map
            curr_image_nodes = env.state.image_nodes[:]
            curr_neighbour_nodes = env.state.neighbour_nodes[:]
            curr_state = PositionNode(
                pos_map=curr_pos_map, 
                image_nodes=curr_image_nodes, 
                neighbour_nodes=curr_neighbour_nodes
            )
            curr_obsv_history = env.image_obsv_history[:]
            curr_total_rewards = env.total_rewards
            curr_count_steps = env.count_steps
            
            for n in range(20):
                result = diffusion_policy(cond)
                # print(f"actions: {result['action']}\npred_actions: {result['action_pred']}")
                
                exe_actions = result['exe_actions'][0]
                pred_actions = result['pred_actions'][0]
                print(f"pred_actions: {pred_actions}")
                
                env.state = PositionNode(
                    pos_map=curr_pos_map, 
                    image_nodes=curr_image_nodes, 
                    neighbour_nodes=curr_neighbour_nodes
                )
                env.image_obsv_history = curr_obsv_history[:]
                env.total_rewards = curr_total_rewards
                env.count_steps = curr_count_steps
                value_sum = 0.
                sim_belief = belief.clone()
                try:
                    for a_idx in pred_actions:
                        # print(f"simulated belief:\n{sim_belief}")
                        weighted_value = (true_values * sim_belief).sum(dim=(1, 2))
                        value_sum += weighted_value
                        sim_obsv, _, _, _ = env.step(action_list[a_idx])
                        sim_state = sim_obsv['poses'].int()
                        sim_belief = torch.zeros_like(belief).to('cuda')
                        sim_belief[:, sim_state[0], sim_state[1]] = 1.
                except IndexError as e:
                    print(e)
                    continue
                print(f"pred actions: {[action_list[a] for a in pred_actions]} expected value: {value_sum}")
                
                if value_sum > best_value:
                    best_value = value_sum.clone()
                    best_candidate = pred_actions

            action_candidates.append([action_list[a] for a in best_candidate])
            print(f"best pred actions:\n{[action_list[a] for a in best_candidate]}")

            obsv_history.clear()
            env.state = PositionNode(
                pos_map=curr_pos_map, 
                image_nodes=curr_image_nodes, 
                neighbour_nodes=curr_neighbour_nodes
            )
            env.image_obsv_history = curr_obsv_history[:]
            env.total_rewards = curr_total_rewards
            env.count_steps = curr_count_steps
            for act_i in best_candidate[:args.n_action_steps]:
                act = action_list[act_i]
                exec_action_history.append(act)
                obsv, _, done, info = env.step(act)
                states.append(obsv['poses'])
                last_env_map = obsv['bev_map']
                obsv_history.append(obsv['bev_map'][None])

                if obsv['poses'].numpy().astype(int).tolist() in target_coords:
                    if not reach_goal:
                        reach_goal = True
                        lenient_succ += 1

                if done:
                    break

                t_act += 1

            if done:
                print(f"The task finished in {t_act + 1} steps.")
                if not reach_goal:
                    # visualize(last_env_map[0], last_env_map[2])
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
        print(f"success rate: {true_succ / (n_eval * i + j + 1)}")
            
    print(f"Trial {i + 1} Complete.")
    print("-- Overall Result --")
    print(f"success rate: {true_succ / (n_eval * (i + 1))}")