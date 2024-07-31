import socket
import numpy as np

from povdp.utils import watch_diffusion

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_policy_diffusion_steps', 'Tp')
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_policy_diffusion_steps', 'Tp')
    # ('value_horizon', 'V'),
    # ('discount', 'd'),
    # ('normalizers', 'N'),
    # ('batch_size', 'b'),
    ##
]


value_args_to_watch = [
    ('prefix', ''),
    ('n_value_iterations', 'K'),
]


value_plan_args_to_watch = [
    ('prefix', ''), 
    ('n_value_iterations', 'K'),
]


base = {

    'diffusion': {
        ## DDPM model
        'diffusion_model1d': 'networks.ConditionalDiffusionUnet1d',
        'diffusion_model2d': 'networks.ConditionalDiffusionUnet2d',
        'diffusion_policy': 'policy.ConditionalGaussianDiffusionPolicy',

        ## EDM model
        'edm_model1d': 'networks.AttentionConditionalUnet1d', 
        'edm_model2d': 'networks.AttentionConditionalUnet2d', 
        'edm_policy': 'policy.ConditionalEDMPolicy',  

        ## consistency model
        'consistency_model1d': 'networks.AttentionConditionalConsistencyUnet1d', 
        'consistency_model2d': 'networks.AttentionConditionalConsistencyUnet2d', 
        'consistency_policy': 'policy.ConditionalConsistencyPolicy', 

        ## EMA and scale function
        'ema_scales_function': 'networks.EMAScalesFunction', 
        'target_ema_mode': 'adaptive', 
        'ema_decay': .999,
        'start_ema': .95, 
        'scale_mode': 'progressive', 
        'start_scales': 2, 
        'end_scales': 200, 
        'distill_steps_per_iter': 50000, 

        ## other vision models
        'env_map_encoder': 'networks.vision.env_map_encoder.EnvMapEncoder',
        'multi_step_obsv_encoder': 'networks.vision.multi_grid_obsv_encoder.MultiGridObsEncoder', 
        'image_obsv_encoder': 'networks.vision.multi_image_obsv_encoder.MultiImageObsEncoder',

        ## timestep schedule
        'sigma_min': .002, 
        'sigma_max': 80.,
        'sigma_data': .5, 
        'rho': 7.,
        'policy_schedule_sampler': 'lognormal', 

        ## weight schedule
        'weight_schedule': 'edm', 

        's_churn': 0., 
        's_tmin': 0, 
        's_tmax': float('inf'), 
        's_noise': 1., 

        ## attention
        'num_heads': 4, 
        'num_head_dim': -1, 
        'attention_resolutions': '32,16,8', 

        ## distillation
        
        'distillation': True, 
        'teacher_model_path': '', 

        ## architecture
        'num_res_blocks': 2,
        'use_scale_shift_norm': False,
        'dropout': 0., 
        'conv_resample': True, 
        'n_groups': 8, 

        # precision
        'use_fp16': False, 
        'fp16_scale_growth': 1e-3, 

        ## conditions
        'n_action_steps': 8,
        'n_latency_steps': 0, 
        'past_action_visible': False,
        'obsv_as_policy_cond_type': 'global',
        'cond_predict_scale': True,
        'pred_action_steps_only': False,
        'env_cond_only': False,
        'n_obsv_steps': np.inf, 

        ## dataset
        'loader': 'dataset.DiscreteGoalDataset',
        'decision_transformer_loader': None,
        'dataset': 'grid_maze_15x15',
        'map_res': (15, 15), 
        'env_dim': 3,
        'dataset_root_path': 'data',
        'termination_penalty': 0.,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer'},
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_max_len_padding': False,
        'use_condition_padding': False,
        'max_path_length': 200,
        'sequence_modeling_K': 4, 

        'horizon': 256,
        'n_policy_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'loss_norm': 'l2',
        'predict_epsilon': False,
        'diffusion_step_embed_dim': 256,
        'env_embed_dim': 256,
        'multi_step_obsv_embed_dim': 256,
        'policy_dim_mults': (1, 4, 8),
        'env_encoder_hidden_dims': (128, 256),
        'multi_step_obsv_encoder_hidden_dims': (128, 256),
        # 'renderer': 'utils.Maze2dRenderer',
        'n_bits': 8, # n_bits >= log2(K), where K=|A| or K=|O|
        'bit_scale': 1.,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch_diffusion(diffusion_args_to_watch),

        ## training
        'trainer': 'train.Trainer',
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'train_batch_size': 32,
        'val_batch_size': 1,
        'learning_rate': 2e-4,
        'policy_learning_rate': None, 
        'policy_learning_rate_anneal_steps': 0, 
        'lr_scheduler': 'cosine',
        'rollout_every': 50,
        'val_every': 10,
        'eval_every': 50,
        'gradient_accumulate_every': 2,
        'log_freq': 100, 
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'separate_training': False,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
        ## loading
        'load_from_ckpt': False,
        'diffusion_loadpath': 'f:diffusion/H{horizon}_Tp{n_policy_diffusion_steps}',
        'diffusion_epoch': 'latest',
        'decision_transformer_epoch': 'latest'
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## dataset
        'dataset': 'grid_maze_15x15_vr_2',
        'dataset_root_path': 'data',
        'map_res': (15, 15), 
        'env_dim': 3,
        'sequence_modeling_K': 4, 

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'max_episode_steps': 200,
        'normalizers': {'LimitsNormalizer',},
        'ema_decay': .999, 
        'schedule_sampler': 'lognormal', 

        ## conditional
        'n_action_steps': 8,
        'n_latency_steps': 0, 
        'env_cond_only': False,
        'past_action_visible': False,
        'obsv_as_policy_cond_type': 'global',
        'cond_predict_scale': True,
        'pred_action_steps_only': False,
        'n_obsv_steps': np.inf,

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch_diffusion(plan_args_to_watch),
        'suffix': '0',

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
        'value_loadpath': '',
        'value_epoch': ''
    },

    # 'value': {
    #     ## dataset
    #     'loader': 'dataset.ValueSequenceDataset', 
    #     'dataset': 'grid_maze_15x15_vr_2',
    #     'dataset_root_path': 'data',
    #     'max_path_length': 200,
    #     'use_max_len_padding': True,
    #     'horizon': 256, 

    #     ## domain
    #     'map_res': (15, 15), 
    #     'env_dim': 3,
    #     'n_actions': 9, 
    #     'n_observations': 256,

    #     ## model
    #     'value_function': 'networks.guides.ValueFunction',
    #     'belief_filter': 'networks.guides.BeliefFilter',
    #     'hidden_dim_valid_action_filter': 256, 
    #     'hidden_dim_reward_fn': 256, 
    #     'motion_scale': 10.,
    #     'belief_filter_dropout': 0.,
    #     'value_function_dropout': .2,
    #     'n_value_iterations': 60,
    #     'gamma': .99,
    #     'epsilon': None,

    #     ## training
    #     'trainer': 'train.ValueTrainer',
    #     'n_train_steps': 2e6,
    #     'n_steps_per_epoch': 10000,
    #     'train_batch_size': 32,
    #     'val_batch_size': 1,
    #     'learning_rate': .001,
    #     'lr_scheduler': 'cosine',
    #     'log_freq': 100, 
    #     'save_freq': 1000,
    #     'n_saves': 50,
    #     'gradient_accumulate_every': 2,
    #     'save_parallel': False,
    #     'n_reference': 50,
    #     'bucket': None,
    #     'device': 'cuda',

    #     ## serialization
    #     'logbase': 'logs',
    #     'prefix': 'value/',
    #     'exp_name': watch_value(value_args_to_watch),

    #      ## loading
    #     'load_from_ckpt': False,
    #     'value_loadpath': 'f:value/K{n_value_iterations}',
    #     'value_epoch': 'latest',

    # },

    # 'value_plan': {
    #     ## dataset
    #     'dataset': 'grid_maze_15x15_vr_2',
    #     'dataset_root_path': 'data',
    #     'map_res': (15, 15),
    #     'env_dim': 3,

    #     ## model
    #     'n_value_iterations': 60,

    #     ## evaluation
    #     'batch_size': 1,
    #     'device': 'cuda',
        
    #     ## serialization
    #     'vis_freq': 10,
    #     'logbase': 'logs',
    #     'prefix': 'value_plans/release',
    #     'exp_name': watch_value(value_plan_args_to_watch),
    #     'suffix': '0',

    #      ## loading
    #     'value_loadpath': 'f:value/K{n_value_iterations}',
    #     'value_epoch': 'latest',
    # }
}

base['diffusion']['arch_variation'] = ''
base['plan']['arch_variation'] = ''
# base['value']['arch_variation'] = ''
# base['value_plan']['arch_variation'] = ''

#------------------------ overrides ------------------------#

'''
    gridworld obstacle/maze episode steps: 
'''

grid_maze_15x15_vr_2 = {
    'diffusion': {
        'dataset': 'grid_maze_15x15_vr_2_single_episode_per_env',
        'loader': 'dataset.PartiallyObservableConditionalDiscreteDataset',
        'horizon': 8,
        'n_action_steps': 4, 
        'n_bits': 8,
        'obsv_as_policy_cond_type': 'global',
        'use_max_len_padding': True,
        'use_condition_padding': True,
        'env_cond_only': True,  
        'n_obsv_steps': 8, 
        'n_policy_diffusion_steps': 32, 
        'map_res': (15, 15), 
        'env_dim': 3, 
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',}, 
        'max_path_length': 60, 
        'arch_variation': 'edm_transformer_advanced_attn',  
        'learning_rate': 2e-5, 
        'policy_learning_rate': 2e-5, 
        'n_steps_per_epoch': 10000, 
        'n_train_steps': 6e5, 
        'eval_every': 1,
        'val_every': 1,
        'separate_training': True,
        'load_from_ckpt': False,
    },
    'plan': {
        'dataset': 'grid_maze_15x15_vr_2_single_episode_per_env',
        'horizon': 8,
        'n_action_steps': 1,
        'obsv_as_policy_cond_type': 'global',
        'env_cond_only': True,
        'n_obsv_steps': 8,
        'n_policy_diffusion_steps': 32,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',},
        'max_path_length': 60,
        'arch_variation': 'edm_transformer_advanced_attn', 
        'diffusion_loadpath': \
            'f:diffusion/H{horizon}_Tp{n_policy_diffusion_steps}',
        'diffusion_epoch': 'latest',
        'value_loadpath': 'data/grid_maze_15x15_vr_2_single_episode_per_env/models/POCALVINConv2d_k_60_i_3_h_150_adam_0.005_0.1_0.25_0728_024929_315227/',
        'value_epoch': '099'
    },
}

grid_maze_15x15_vr_1 = {
    'diffusion': {
        'dataset': 'grid_maze_15x15_vr_1',
        'loader': 'dataset.PartiallyObservableConditionalDiscreteDataset',
        'horizon': 8,
        'n_action_steps': 4,  
        'n_bits': 8,
        'obsv_as_policy_cond_type': 'global',
        'use_max_len_padding': True,
        'use_condition_padding': True,
        'env_cond_only': True, 
        'n_obsv_steps': 4,
        'n_policy_diffusion_steps': 32,
        'map_res': (15, 15),
        'env_dim': 3,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',},
        'max_path_length': 50,
        'arch_variation': 'edm',
        'learning_rate': 2e-5, 
        'policy_learning_rate': 2e-5, 
        'n_steps_per_epoch': 10000,
        'n_train_steps': 6e5,
        'eval_every': 1,
        'val_every': 1,
        'separate_training': True,
        'load_from_ckpt': False,
    },
    'plan': {
        'dataset': 'grid_maze_15x15_vr_1',
        'horizon': 8,
        'n_action_steps': 4,
        'obsv_as_policy_cond_type': 'global',
        'env_cond_only': True,
        'n_obsv_steps': 4,
        'n_policy_diffusion_steps': 32,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',},
        'max_path_length': 50,
        'arch_variation': 'edm', 
        'diffusion_loadpath': \
            'f:diffusion/H{horizon}_Tp{n_policy_diffusion_steps}',
        'diffusion_epoch': '293001',
    },
}

grid_maze_15x15_vr_3 = {
    'diffusion': {
        'dataset': 'grid_maze_15x15_vr_3',
        'loader': 'dataset.PartiallyObservableConditionalDiscreteDataset',
        'horizon': 8,
        'n_action_steps': 4,  
        'n_bits': 8,
        'obsv_as_policy_cond_type': 'global',
        'use_max_len_padding': True,
        'use_condition_padding': True,
        'env_cond_only': True,
        'n_obsv_steps': 4,
        'n_policy_diffusion_steps': 32,
        'map_res': (15, 15),
        'env_dim': 3,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',},
        'max_path_length': 50,
        'arch_variation': 'edm',
        'learning_rate': 2e-5, 
        'policy_learning_rate': 2e-5, 
        'n_steps_per_epoch': 10000,
        'n_train_steps': 6e5,
        'eval_every': 1,
        'val_every': 1,
        'separate_training': True,
        'load_from_ckpt': False,
    },
    'plan': {
        'dataset': 'grid_maze_15x15_vr_3',
        'horizon': 10,
        'n_action_steps': 4,
        'obsv_as_policy_cond_type': 'global',
        'env_cond_only': True,
        'n_obsv_steps': 4,
        'n_policy_diffusion_steps': 32,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',},
        'max_path_length': 50,
        'arch_variation': 'edm', 
        'diffusion_loadpath': \
            'f:diffusion/H{horizon}_Tp{n_policy_diffusion_steps}',
        'diffusion_epoch': '023001',
    },
}