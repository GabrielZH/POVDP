import os
import sys
import json
import random
from copy import deepcopy
import numpy as np
import pdb

import torch
from einops import rearrange

import povdp.dataset as dataset
import povdp.utils as utils

from calvin.core.domains.factory import get_factory
from calvin.core.domains.avd.navigation.pose_nav.actions import AVDPoseActionSet
from calvin.core.domains.avd.dataset.scene_manager import AVDSceneManager
from calvin.core.domains.avd.dataset.data_classes import Scene


action_list = AVDPoseActionSet()

class Parser(utils.Parser):
    dataset: str = 'avd_pose_nav'
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
scenes = AVDSceneManager(
    data_dir=os.path.join(
        args.dataset_root_path, 'avd/src'), 
    scene_resize=args.resize, 
    target=args.target, 
    in_ram=args.in_ram, 
    target_size_ratio=args.target_size_ratio, 
    avd_workers=4
)

#---------------------------------- loading ----------------------------------#

experiment = utils.load_diffusion_edm_policy_test(
    args.logbase, 
    args.dataset, 
    args.arch_variation, 
    args.diffusion_loadpath, 
    epoch=args.diffusion_epoch, 
    point_cloud_encoder_epoch=args.self_supervise_point_cloud_epoch, 
    flat_map_encoder_epoch=args.self_supervise_flat_map_epoch, 
    ema_decay=args.ema_decay, 
    schedule_sampler=args.schedule_sampler, 
    train_dataset_config='train_dataset_config', 
    val_dataset_config='val_dataset_config',
    model1d_config='edm_model1d_config', 
    point_cloud_to_2d_config='point_cloud_to_2d_projector', 
    policy_config='edm_policy_config',
    trainer_config='edm_policy_trainer_config')

policy = experiment.edm_policy
policy.diffusion_model.eval()
policy.point_cloud_to_2d_projector.eval()

#---------------------------------- main loop ----------------------------------#

succ = 0
max_path_len = 0
n_trials = 1
n_eval = 3

trial_res = list()
for i in range(n_trials):
    for j in range(n_eval):
        episode_info, init_obsv, opt_actions = (
            output for output in env.reset_avd()
        )
        scene_name = episode_info['scene_name']
        scene: Scene = scenes[scene_name]
        true_values = episode_info['values'].to('cuda')
        true_values = torch.where(
            true_values == float('-inf'),
            torch.full_like(true_values, -100),
            true_values
        )
        true_values = torch.where(
            true_values == float('inf'),
            torch.full_like(true_values, 100),
            true_values
        )
        print(f"true values:\n{true_values}")

        print(
            f"initial pose: {init_obsv['poses']}; target: {episode_info['target']}")

        exec_action_history = list()
        # pred_states = list()
        obsv = done = info = None
        image_name = init_obsv['state_info']
        image_names = [image_name]
        for _ in range(args.ori_res):
            act = action_list[5]
            obsv, _, done, info = env.step(act)
            image_names.append(image_name)
        indices = scene.names_to_indices(image_names)
        rgb = scene.rgb_images[indices]
        emb = scene.embeddings[indices]
        depth = scene.depth_images[indices] / scene.scale
        valid_points = depth != 0
        surf_xyz = scene.coords(image_names)
        rgb = rearrange(rgb, 'b h w f -> b f h w')
        rgb = torch.from_numpy(rgb).float() / 255
        valid_points = torch.from_numpy(valid_points).bool()
        surf_xyz = rearrange(surf_xyz, 'b h w f -> b f h w')
        surf_xyz = torch.from_numpy(surf_xyz).float()
        index = torch.full((len(rgb),), 0, dtype=torch.long)
        cond = {
            'target': episode_info['target'].to('cuda'),
            'target_emb': torch.from_numpy(episode_info['target_emb'])[None].to('cuda'), 
            'target_grid': episode_info['targets_grid'].to('cuda'), 
            'occupancy': torch.from_numpy(episode_info['occupancy'])[None].to('cuda'), 
            'rgb': rgb.to('cuda'), 
            'emb': emb.to('cuda'), 
            'valid_points': valid_points.to('cuda'), 
            'surf_xyz': surf_xyz.to('cuda'),
            'index': index.to('cuda'), 
        }

        t_act = 0

        states = list()
        buffer = None
        action_candidates = list()
        failed = False
        while t_act < args.max_steps:
            # print(f"obsv cond shape: {cond['observations'].shape}")
            # print(f"env cond shape: {cond['env_maps'].shape}")
            
            # _, unnormed_recon_env = env_reconstructor(cond)
            # unnormed_recon_env = unnormed_recon_env.detach().cpu().numpy()
            # print(f"recon env:\n{unnormed_recon_env[0]}")
            belief = torch.zeros(args.ori_res, *args.map_res)[None].to('cuda')
            if not t_act:
                # print(f"true env:\n{init_obsv['feature_map'].numpy()}")
                state = init_obsv['poses'].int()
                belief[:, state[0], state[1]] = 1.
            else:
                # print(f"true env:\n{obsv['feature_map'].numpy()}")
                state = obsv['poses'].int()
                belief[:, state[0], state[1]] = 1.

            plan = policy(cond)
                
            exec_actions = plan['exe_actions'][0]
            pred_actions = plan['pred_actions'][0]
            
            # print(f"values:\n{true_values}")

            best_candidate = None
            best_value = -torch.inf
            original_state = env.state
            viable_candidates = 0
            for n in range(20):
                candidate_failed = False
                env.state = original_state
                plan = policy(cond)
                # print(f"actions: {result['action']}\npred_actions: {result['action_pred']}")
                
                exec_actions = plan['exe_actions'][0]
                pred_actions = plan['pred_actions'][0]
                print(f"pred action index: {pred_actions}")

                value_sum = 0.
                sim_belief = belief.clone()
                valid_until = len(pred_actions)
                for ind, a_idx in enumerate(pred_actions):
                    # print(f"simulated belief:\n{sim_belief}")
                    weighted_value = (true_values * sim_belief).sum(dim=(1, 2, 3))
                    value_sum += weighted_value
                    try:
                        sim_obsv, _, _, _ = env.step(action_list[a_idx])
                    except IndexError:
                        if not ind:
                            candidate_failed = True
                            break
                        mean_value = value_sum / (ind)
                        valid_until = ind
                        break
                    sim_state = sim_obsv['poses'].int()
                    sim_belief = torch.zeros_like(belief).to('cuda')
                    sim_belief[:, sim_state[0], sim_state[1], sim_state[2]] = 1.
                if candidate_failed: break
                mean_value = value_sum / pred_actions.shape[0]
                print(f"valid until: {valid_until}")
                print(f"pred actions: {[action_list[a] for a in pred_actions[:valid_until]]} " 
                      f"mean expected value: {mean_value}")

                if mean_value > best_value:
                    best_value = mean_value.clone()
                    best_candidate = pred_actions[:valid_until].clone()

                viable_candidates += 1
            
            if not viable_candidates:
                failed = True
                break

            env.state = original_state
            if t_act and buffer.shape[0]:
                value_sum = 0.
                sim_belief = belief.clone()

                for a_idx in buffer:
                    weighted_value = (true_values * sim_belief).sum(dim=(1, 2, 3))
                    value_sum += weighted_value
                    sim_obsv, _, _, _ = env.step(action_list[a_idx])
                    sim_state = sim_obsv['poses'].int()
                    sim_belief = torch.zeros_like(belief).to('cuda')
                    sim_belief[:, sim_state[0], sim_state[1], sim_state[2]] = 1.
                mean_value = value_sum / buffer.shape[0]
                print(f"last best candidate: {[action_list[a] for a in buffer]} "
                      f"mean expected value: {mean_value}")
                
                if mean_value > best_value:
                    best_value = mean_value.clone()
                    best_candidate = buffer.clone()
                    action_exec_steps = min(action_exec_steps, buffer.shape[0])
                    buffer = buffer[1:]
                else:
                    buffer = best_candidate[1:]
                    action_exec_steps = args.n_action_steps
            else:
                buffer = best_candidate[1:]
                action_exec_steps = args.n_action_steps

            action_candidates.append([action_list[a] for a in best_candidate])
            print(f"best pred actions:\n{[action_list[a] for a in best_candidate]}")
            
            env.state = original_state
            if random.random() < .25:
                for _ in range(args.ori_res):
                    act = action_list[5]
                    obsv, _, done, info = env.step(act)
                    image_name = obsv['state_info']
                    image_names.append(image_name)
                indices = scene.names_to_indices(image_names)
                rgb = scene.rgb_images[indices]
                emb = scene.embeddings[indices]
                depth = scene.depth_images[indices] / scene.scale
                valid_points = depth != 0
                surf_xyz = scene.coords(image_names)
                rgb = rearrange(rgb, 'b h w f -> b f h w')
                rgb = torch.from_numpy(rgb).float() / 255
                valid_points = torch.from_numpy(valid_points).bool()
                surf_xyz = rearrange(surf_xyz, 'b h w f -> b f h w')
                surf_xyz = torch.from_numpy(surf_xyz).float()
                index = torch.full((len(rgb),), 0, dtype=torch.long)
            else:
                for act_i in best_candidate[:args.n_action_steps]:
                    act = action_list[act_i]
                    print(f"pred action: {act}")
                    exec_action_history.append(act)
                    obsv, _, done, info = env.step(act)
                    states.append(obsv['poses'])
                    print(f"pose history: {states}")
                    image_name = obsv['state_info']
                    image_names.append(image_name)
                    indices = scene.names_to_indices(image_names)
                    rgb = scene.rgb_images[indices]
                    emb = scene.embeddings[indices]
                    depth = scene.depth_images[indices] / scene.scale
                    valid_points = depth != 0
                    surf_xyz = scene.coords(image_names)
                    rgb = rearrange(rgb, 'b h w f -> b f h w')
                    rgb = torch.from_numpy(rgb).float() / 255
                    valid_points = torch.from_numpy(valid_points).bool()
                    surf_xyz = rearrange(surf_xyz, 'b h w f -> b f h w')
                    surf_xyz = torch.from_numpy(surf_xyz).float()
                    index = torch.full((len(rgb),), 0, dtype=torch.long)

                if done:
                    succ += 1
                    break

                t_act += 1
                # elif 

            cond = {
                'target': episode_info['target'].to('cuda'),
                'target_emb': torch.from_numpy(episode_info['target_emb'])[None].to('cuda'), 
                'target_grid': episode_info['targets_grid'].to('cuda'), 
                'occupancy': torch.from_numpy(episode_info['occupancy'])[None].to('cuda'), 
                'rgb': rgb.to('cuda'), 
                'emb': emb.to('cuda'), 
                'valid_points': valid_points.to('cuda'), 
                'surf_xyz': surf_xyz.to('cuda'),
                'index': index.to('cuda'),
            }

        print(f"action step: {t_act}")
        print(f"state history: {states}")
        print(f"expert actions: {opt_actions}")
        print(f"executed actions: {exec_action_history}")
        print(f"best action candidate history:")
        for act in action_candidates:
            print(act)

        print(f"-- Trial {i + 1} Eval {j + 1} Result --")
        true_succ_rate = succ / (n_eval * i + j + 1)
        print(f"success rate: {true_succ_rate}")

    trial_res.append(true_succ_rate)

    print(f"Trial {i + 1} Complete.")
    print("-- Accumulative Result --")
    print(f"success rate: {succ / (n_eval * (i + 1))}")

mean_true_succ_rate = np.array(trial_res).mean()
std_true_succ_rate = np.sqrt(
    np.array(
        [(x - mean_true_succ_rate) ** 2 
         for x in trial_res]
    ).sum()
)
print(f"-- Overall Result over {n_trials} trials --")
print(f"Mean success rate: {mean_true_succ_rate}")
print(f"Success rate std err: {std_true_succ_rate}")