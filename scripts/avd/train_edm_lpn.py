"""
Train a diffusion model on images.
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import povdp.utils as utils
from povdp.utils import (
    logger, 
    load_config, 
    create_named_schedule_sampler, 
)
from swin3d.Swin3D.SemanticSeg.util import config
from swin3d.Swin3D.SemanticSeg.model.Swin3D_RGB import Swin3D

from povdp.train.edm_trainer import EDMPolicyTrainer

from calvin.core.domains.gridworld.actions import GridDirs
import copy

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(profile='full')
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'avd_pos_nav'
    config: str = 'configs.avd'

args = Parser().parse_args('diffusion')
seg_args = Parser().parse_args('segmentation')

logger.configure()

action_list = GridDirs.DIRS_8 + [(0, 0)]

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
experiment_params = (
    f"{args.n_episodes}_{'_'.join(map(str, args.map_bbox))}"
    f"__{'_'.join(map(str, args.map_res))}__{args.ori_res or 'pos'}"
    f"_{'_'.join(map(str, args.resize if args.resize is not None else (1080, 1920)))}_{args.target}"
    f"_{args.min_traj_len}_{args.max_steps}_{args.sample_free}")

swin3d = Swin3D(
    depths=seg_args.depths, 
    channels=seg_args.channels, 
    num_heads=seg_args.num_heads, 
    window_sizes=seg_args.window_size, 
    up_k=seg_args.up_k, 
    quant_sizes=seg_args.quant_size, 
    drop_path_rate=seg_args.drop_path_rate, 
    num_classes=seg_args.classes, 
    num_layers=seg_args.num_layers, 
    stem_transformer=seg_args.stem_transformer, 
    upsample=seg_args.upsample, 
    down_stride=seg_args.down_stride, 
    knn_down=seg_args.knn_down, 
    signal=seg_args.signal, 
    in_channels=seg_args.fea_dim, 
    use_offset=seg_args.use_offset, 
    fp16_mode=seg_args.fp16_mode,
)

train_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'train_dataset_config.pkl'),
    map_res=args.map_res, 
    ori_res=args.ori_res, 
    resize=args.resize, 
    target=args.target, 
    target_size_ratio=args.target_size_ratio, 
    in_ram=args.in_ram, 
    horizon=args.horizon,
    n_obsv_steps=args.n_obsv_steps, 
    n_action_steps=args.n_action_steps,
    normalizers=args.normalizers,
    ds_path=os.path.join(
        args.dataset_root_path, 
        args.dataset, 
        experiment_params),
    ds_type='train',
    preprocess_fns=args.preprocess_fns,
    use_max_len_padding=args.use_max_len_padding,
    use_condition_padding=args.use_condition_padding,
    max_path_length=args.max_path_length,
    max_n_episodes=args.max_n_episodes, 
    n_bits=args.n_bits,
    segmentation_args=seg_args
)
val_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'val_dataset_config.pkl'),
    map_res=args.map_res, 
    ori_res=args.ori_res, 
    resize=args.resize, 
    target=args.target, 
    target_size_ratio=args.target_size_ratio, 
    in_ram=args.in_ram, 
    horizon=args.horizon,
    n_obsv_steps=args.n_obsv_steps, 
    n_action_steps=args.n_action_steps,
    normalizers=args.normalizers,
    ds_path=os.path.join(
        args.dataset_root_path, 
        args.dataset, 
        experiment_params),
    ds_type='val',
    preprocess_fns=args.preprocess_fns,
    use_max_len_padding=args.use_max_len_padding,
    use_condition_padding=args.use_condition_padding,
    max_path_length=args.max_path_length,
    max_n_episodes=args.max_n_episodes, 
    n_bits=args.n_bits,
)

train_dataset = train_dataset_config(swin3d)
val_dataset = val_dataset_config(swin3d)
action_dim = train_dataset.action_dim
obsv_dim = train_dataset.obsv_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#
point_cloud_to_2d_projector_config = utils.Config(
    args.point_cloud_to_2d_projector, 
    savepath=(args.savepath, 'point_cloud_to_2d_projector.pkl'), 
    map_bbox=args.map_bbox, 
    map_res=args.map_res, 
    pcn_h=args.pcn_h, 
    pcn_i=args.pcn_i, 
    pcn_f=args.pcn_f, 
    v_bbox=args.v_bbox, 
    v_res=args.v_res, 
    xyz_to_h=args.xyz_to_h, 
    xyz_to_w=args.xyz_to_w, 
    pcn_sample_ratio=args.pcn_sample_ratio, 
    use_group_norm=args.pcn_use_group_norm, 
    use_embeddings=args.pcn_use_embeddings, 
    dot_channels=args.dot_channels, 
    reduce_thresh=args.reduce_thresh, 
    dropout=args.point_cloud_dropout, 
    device=args.device, 
)
point_cloud_to_2d_projector = point_cloud_to_2d_projector_config()

edm_model1d_config = utils.Config(
    args.edm_model1d, 
    savepath=(args.savepath, 'edm_model1d_config.pkl'), 
    inp_dim=args.n_bits,
    env_map_dim=20 + args.v_res, 
    global_cond_dim=args.env_embed_dim,
    time_step_embed_dim=args.diffusion_step_embed_dim, 
    dim_mults=args.policy_dim_mults, 
    dropout=args.dropout, 
    conv_resample=args.conv_resample, 
    use_fp16=args.use_fp16, 
    n_groups=args.n_groups, 
    num_heads=args.num_heads, 
    num_head_dim=args.num_head_dim, 
    cond_predict_scale=args.cond_predict_scale,
    use_scale_shift_norm=args.use_scale_shift_norm,
    is_inp_env_map=False, 
    device=args.device, 
)
edm_model1d = edm_model1d_config()

logger.log("creating model and diffusion...")

if args.use_fp16:
    edm_model1d.convert_to_fp16() 
    # edm_model2d.convert_to_fp16()

edm_policy_config = utils.Config(
    args.edm_policy, 
    savepath=(args.savepath, 'edm_policy_config.pkl'), 
    horizon=args.horizon, 
    action_dim=action_dim, 
    n_action_steps=args.n_action_steps,
    n_obsv_steps=args.n_obsv_steps,
    n_timesteps=args.n_policy_diffusion_steps, 
    loss_norm=args.loss_type, 
    clip_denoised=args.clip_denoised, 
    sigma_data=args.sigma_data, 
    sigma_max=args.sigma_max, 
    sigma_min=args.sigma_min, 
    rho=args.rho, 
    weight_schedule=args.weight_schedule, 
    s_churn=args.s_churn, 
    s_tmax=args.s_tmax, 
    s_tmin=args.s_tmin, 
    s_noise=args.s_noise, 
    pred_action_steps_only=args.pred_action_steps_only,
    bit_scale=args.bit_scale, 
    device=args.device, 
)
edm_policy = edm_policy_config(
    edm_model1d, 
    point_cloud_to_2d_projector, 
)

policy_schedule_sampler = create_named_schedule_sampler(
    name=args.schedule_sampler, 
    diffusion=edm_policy
)

edm_policy_trainer_config = utils.Config(
    EDMPolicyTrainer, 
    savepath=(args.savepath, 'edm_policy_trainer_config.pkl'),
    ema_decay=args.ema_decay, 
    train_batch_size=args.train_batch_size, 
    val_batch_size=args.val_batch_size, 
    train_lr=args.learning_rate, 
    lr_anneal_steps=args.learning_rate_anneal_steps, 
    num_train_steps=args.n_train_steps, 
    log_freq=args.log_freq, 
    sample_freq=args.sample_freq, 
    save_freq=args.save_freq, 
    label_freq=10000, 
    use_fp16=args.use_fp16, 
    fp16_scale_growth=args.fp16_scale_growth, 
    results_folder=args.savepath, 
    load_from_ckpt=args.load_from_ckpt, 
    resume_diffusion_checkpoint=args.diffusion_epoch, 
    resume_point_cloud_checkpoint=args.self_supervise_point_cloud_epoch, 
    resume_flat_map_checkpoint=args.self_supervise_flat_map_epoch 
)
edm_policy_trainer = edm_policy_trainer_config(
    edm_policy, 
    train_dataset, 
    val_dataset, 
    policy_schedule_sampler, 
)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

# utils.report_parameters(diffusion_model1d)

# print('Testing forward...', end=' ', flush=True)
# batch = utils.batchify(train_dataset[0])

# t, weights = edm_policy_trainer.schedule_sampler.sample(
#      1, device='cuda'
# )
# policy_loss, _ = edm_policy.loss(
#     x_start=batch.actions, 
#     sigma=t, 
#     cond=batch.conditions
# )
# policy_loss.backward()
# print('Policy -- ✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

summary_writer = SummaryWriter()

edm_policy_trainer.train(
    summary_writer=summary_writer
)

summary_writer.close()
