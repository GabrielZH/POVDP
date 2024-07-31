import os
import socket
import numpy as np

from povdp.utils import watch_diffusion

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_policy_diffusion_steps', 'Tp'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_policy_diffusion_steps', 'Tp'),
    # ('discount', 'd'),
    # ('normalizers', 'N'),
    # ('batch_size', 'b'),
    ##
]


def parse_path(path):
    components = path.split('__')
    n_episodes = int(components[0].split('_')[0])
    map_bbox = list(map(int, components[0].split('_')[1:5]))
    map_res = list(map(int, components[1].split('_')))
    ori_res = components[2].split('_')[0]
    if ori_res == 'pos': ori_res = None
    resize = list(map(int, components[2].split('_')[1:]))
    target = components[3]
    min_traj_len = int(components[4])
    max_steps = int(components[5])
    sample_free = components[6]

    return {
        'n_episodes': n_episodes,
        'map_bbox': map_bbox,
        'map_res': map_res,
        'ori_res': ori_res,
        'resize': resize,
        'target': target,
        'min_traj_len': min_traj_len,
        'max_steps': max_steps,
        'sample_free': sample_free,
    }


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

        ## point cloud projected to 2D space
        'point_cloud_to_2d_projector': 'networks.PointCloudTo2dProjector', 

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

        ## 3D point cloud
        'pcn_i': 3,
        'pcn_h': 40,
        'pcn_f': 20,
        'xyz_to_h': 0, 
        'xyz_to_w': 2, 
        'pcn_sample_ratio': 1., 
        'pcn_use_group_norm': True, 
        'pcn_use_embeddings': True, 
        'dot_channels': 8, 
        'reduce_thresh': 1, 

        ## attention
        'num_heads': 8, 
        'num_head_dim': -1, 
        'attention_resolutions': '32,16,8', 

        ## distillation
        
        'distillation': True, 
        'teacher_model_path': '', 

        ## architecture
        'num_res_blocks': 2,
        'cond_encoder_hidden_dims': None,
        'use_scale_shift_norm': False,
        'dropout': .2, 
        'point_cloud_dropout': .2, 
        'conv_resample': True, 
        'n_groups': 8, 

        # precision
        'use_fp16': False, 
        'fp16_scale_growth': 1e-3, 

        ## conditions
        'n_obsv_steps': 2,
        'n_action_steps': 8,
        'n_latency_steps': 0, 
        'past_action_visible': False,
        'obsv_as_cond_type': 'global',
        'cond_predict_scale': True,
        'pred_action_steps_only': False,

        ## dataset
        'split': 'train', 
        'loader': 'dataset.DiscreteGoalDataset',
        'dataset_root_path': 'data',
        'dataset': 'avd/pose_nav',
        'map_res': (40, 40), 
        'ori_res': None, 
        'v_res': 10, 
        'resize': None, 
        'map_bbox': [-15, -15, 15, 15], 
        'v_bbox': (-4, 4), 
        'target': 'coca_cola_glass_bottle', 
        'target_size_ratio': .2, 
        'in_ram': False, 
        'termination_penalty': None,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer'},
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_max_len_padding': False,
        'use_condition_padding': False,
        'max_n_episodes': 10000, 
        'max_path_length': 200,
        'n_episodes': 10000, 
        'min_traj_len': 10, 
        'max_steps': 100, 
        'sample_free': 8, 

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
        'train_batch_size': 8,
        'val_batch_size': 1,
        'learning_rate': 2e-4, 
        'policy_learning_rate': 2e-4,
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
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
        ## loading
        'load_from_ckpt': False,
        'diffusion_loadpath': 'f:diffusion/H{horizon}_Tp{n_policy_diffusion_steps}',
        'diffusion_epoch': 'latest',
        'self_supervise_point_cloud_epoch': None, 
        'self_supervise_flat_map_epoch': None
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## dataset
        'split': 'val', 
        'dataset': 'avd/pose_nav',
        'dataset_root_path': 'data',
        'map_res': (25, 25), 
        'ori_res': None, 
        'resize': None, 
        'map_bbox': [-11, -11, 11, 11], 
        'target': 'coca_cola_glass_bottle', 
        'target_size_ratio': .2, 
        'in_ram': False, 
        'n_episodes': 10000, 
        'min_traj_len': 10, 
        'max_steps': 60, 
        'sample_free': 8, 

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'max_episode_steps': 200,
        'normalizers': {'LimitsNormalizer',},
        'ema_decay': .999, 
        'schedule_sampler': 'lognormal', 

        ## conditional
        'n_obsv_steps': 2,
        'n_action_steps': 8,
        'n_latency_steps': 0, 
        'env_cond_only': False,
        'past_action_visible': False,
        'obsv_as_policy_cond_type': 'global',
        'cond_predict_scale': True,
        'pred_action_steps_only': False,

        ## serialization
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch_diffusion(plan_args_to_watch),
        'suffix': '0',

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

    'segmentation': {
        'data_name': 's3dis',
        'yz_shift': True,
        'classes': 13,
        'fea_dim': 6,
        'voxel_size': 0.04,
        'voxel_max': 80000,
        'loop': 6,
        'vote_num': 12, 
        'save_output': False, 

        # arch
        'arch': 'Swin3D_RGB',
        'fp16_mode': 1,
        'stem_transformer': True,
        'use_xyz': True,
        'use_offset': True,
        'sync_bn': True,  # adopt sync_bn or not
        'rel_query': True,
        'rel_key': True,
        'rel_value': True,
        'quant_size': 4, # pos_bias_table: 2x(4x5)-1 = 39
        'num_layers': 5,
        'patch_size': 1,
        'window_size': [5, 7, 7, 7, 7], 
        'depths': [2, 4, 9, 4, 4], 
        'channels': [48, 96, 192, 384, 384], 
        'num_heads': [6, 6, 12, 24, 24],
        'signal': True,
        'knn_down': True,
        'down_stride': 2,
        'upsample': 'linear_attn',
        'up_k': 3,
        'drop_path_rate': 0.3,
        'concat_xyz': True,

        # training
        'aug': True,
        'transformer_lr_scale': 0.1, 
        'jitter_sigma': 0.005,
        'jitter_clip': 0.02,
        'scheduler_update': 'epoch', 
        'scheduler': 'MultiStep', 
        'warmup': 'linear',
        'warmup_iters': 1500,
        'warmup_ratio': 0.000001,
        'use_amp': True,
        'optimizer': 'AdamW',
        'ignore_label': 255,
        'train_gpu': [0], 
        'workers': 16,  # data loader workers
        'batch_size': 3, # batch size for training
        'batch_size_val': 2, # batch size for validation during training, memory and speed tradeoff
        'base_lr': 0.0006,
        'epochs': 100,
        'start_epoch': 0,
        'step_epoch': 30,
        'multiplier': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.05,
        'drop_rate': 0.5,
        'manual_seed': 123,
        'print_freq': 10,
        'save_freq': 1,
        'save_path': '/data/zhanggengyu/projects/main/nerv/swin3d/Swin3D/SemanticSeg/runs/s3dis_Swin3D_RGB_S',
        'weight':  '/data/zhanggengyu/projects/main/nerv/swin3d/Swin3D/SemanticSeg/checkpoints/Swin3D-S.pth',
        'weight_for_innner_model': False,
        'resume': 'latest',  # path to latest checkpoint (default: none)
        'skip_first_conv': False,
        'evaluate': True,  # evaluate on validation set, extra gpu memory needed and small batch_size_val is recommend
        'eval_freq': 2,
        'dist_url': 'tcp://127.0.0.1:6789',
        'dist_backend': 'nccl',
        'multiprocessing_distributed': True,
        'world_size': 8,
        'rank': 0,
    },
}

base['diffusion']['arch_variation'] = ''
base['plan']['arch_variation'] = ''

#------------------------ overrides ------------------------#

avd_pose_nav = {
    'diffusion': {
        'loader': 'dataset.PartiallyObservableConditionalDiscreteDataset',
        'horizon': 4,
        'n_obsv_steps': 1,
        'n_action_steps': 1,  
        'n_bits': 8,
        'obsv_as_cond_type': 'global',
        'use_max_len_padding': True,
        'use_condition_padding': True,
        'n_policy_diffusion_steps': 16,
        'map_res': (25, 25),
        'ori_res': 12, 
        'v_res': 5, 
        'resize': (135, 240), 
        'map_bbox': [-11, -11, 11, 11], 
        'target': None, 
        'target_size_ratio': .2, 
        'max_steps': 100, 
        'train_batch_size': 8, 
        # 'env_dim': 3,
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',},
        'n_episodes': 5000, 
        'max_path_length': 80,
        'arch_variation': 'edm_training_avd_pose_nav_8_scenes_simplified',
        'learning_rate': 1e-4, 
        'n_steps_per_epoch': 5000,
        'n_train_steps': 6e5,
        'eval_every': 1,
        'val_every': 1,
        'load_from_ckpt': False, 
        'diffusion_epoch': None, 
        'self_supervise_point_cloud_epoch': None, 
        'self_supervise_flat_map_epoch': None
    },
    'plan': {
        'horizon': 4,
        'n_obsv_steps': 1,
        'n_action_steps': 2,
        'obsv_as_cond_type': 'global',
        'n_policy_diffusion_steps': 16,
        'map_res': (25, 25), 
        'ori_res': 12, 
        'v_res': 5, 
        'resize': (135, 240), 
        'map_bbox': [-11, -11, 11, 11], 
        'target': 'coca_cola_glass_bottle', 
        'target_size_ratio': .2, 
        'normalizers': {'BitNormalizer', 'LimitsNormalizer',},
        'n_episodes': 5000, 
        'max_path_length': 80,
        'arch_variation': 'edm_training_avd_pose_nav_8_scenes_simplified',
        'diffusion_loadpath': \
            'f:diffusion/H{horizon}_Tp{n_policy_diffusion_steps}',
        'diffusion_epoch': '276001',
        'self_supervise_point_cloud_epoch': None, 
        'self_supervise_flat_map_epoch': None
    },
}


avd_pos_nav = {
    'diffusion': {
        'dataset': 'avd/pos_nav', 
        'loader': 'dataset.PartiallyObservableConditionalDiscreteDataset',
        'horizon': 4,
        'n_obsv_steps': 4,
        'n_action_steps': 1,  
        'n_bits': 4,
        'obsv_as_cond_type': 'global',
        'use_max_len_padding': True,
        'use_condition_padding': True,
        'n_policy_diffusion_steps': 32,
        'map_res': (100, 100),
        'v_res': 6, 
        'resize': (135, 240), 
        'map_bbox': [-50, -50, 50, 50], 
        'target': None, 
        'target_size_ratio': .6, 
        'num_heads': 4, # attention heads
        'max_steps': 50, 
        'train_batch_size': 32, 
        'val_batch_size': 1, 
        'env_dim': 3,
        'normalizers': {'BitNormalizer',},
        'n_episodes': 2000, 
        'max_path_length': 50,
        'arch_variation': 'bev_cond_pos_nav',
        'learning_rate': 2e-5, 
        'policy_learning_rate': 2e-5, 
        'n_steps_per_epoch': 10000,
        'n_train_steps': 5e6,
        'eval_every': 1,
        'val_every': 1,
        'load_from_ckpt': False, 
        'diffusion_epoch': None, 
    },
    'plan': {
        'dataset': 'avd/pos_nav', 
        'horizon': 4,
        'n_obsv_steps': 4,
        'n_action_steps': 1,
        'obsv_as_cond_type': 'global',
        'n_policy_diffusion_steps': 32,
        'map_res': (100, 100), 
        'env_dim': 3,
        'v_res': 6, 
        'resize': (135, 240),
        'map_bbox': [-50, -50, 50, 50], 
        'target': None, 
        'target_size_ratio': .2, 
        'normalizers': {'BitNormalizer',},
        'n_episodes': 2000, 
        'max_path_length': 50,
        'max_steps': 50, 
        'arch_variation': 'bev_cond_pos_nav',
        'diffusion_loadpath': \
            'f:diffusion/H{horizon}_Tp{n_policy_diffusion_steps}',
    },
}
