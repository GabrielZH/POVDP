"""
Train a diffusion model on images.
"""

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import povdp.utils as utils
from povdp.utils import (
    logger, 
    load_config, 
    create_named_schedule_sampler, 
)

from povdp.train.edm_trainer import EDMPolicyTrainer
import copy

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
np.set_printoptions(threshold=sys.maxsize)

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'avd_pos_nav'
    config: str = 'configs.avd'

args = Parser().parse_args('diffusion')

logger.configure()

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#
experiment_params = (
    f"{args.n_episodes}_{'_'.join(map(str, args.map_bbox))}"
    f"__{'_'.join(map(str, args.map_res))}__{args.ori_res or 'pos'}"
    f"_{'_'.join(map(str, args.resize if args.resize is not None else (1080, 1920)))}_{args.target}"
    f"_{args.min_traj_len}_{args.max_steps}_{args.sample_free}")


train_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'train_dataset_config.pkl'),
    map_res=args.map_res,
    horizon=args.horizon,
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
    n_obsv_steps=args.n_obsv_steps,
    n_bits=args.n_bits,
)
val_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'val_dataset_config.pkl'),
    map_res=args.map_res,
    horizon=args.horizon,
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
    n_obsv_steps=args.n_obsv_steps,
    n_bits=args.n_bits,
)

train_dataset = train_dataset_config()
val_dataset = val_dataset_config()
action_dim = train_dataset.action_dim
obsv_dim = train_dataset.obsv_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

edm_model1d_config = utils.Config(
    args.edm_model1d, 
    savepath=(args.savepath, 'edm_model1d_config.pkl'), 
    inp_dim=args.n_bits,
    global_cond_dim=args.env_embed_dim,
    time_step_embed_dim=args.diffusion_step_embed_dim, 
    dim_mults=args.policy_dim_mults, 
    dropout=args.dropout, 
    conv_resample=args.conv_resample, 
    use_fp16=args.use_fp16, 
    n_groups=args.n_groups, 
    num_heads=args.num_heads, 
    num_head_dim=args.num_head_dim, 
    n_obsv_steps=args.n_obsv_steps,
    cond_predict_scale=args.cond_predict_scale,
    use_scale_shift_norm=args.use_scale_shift_norm, 
    device=args.device, 
)
edm_model1d = edm_model1d_config()

logger.log("creating model and diffusion...")

if args.use_fp16:
    edm_model1d.convert_to_fp16() 
    # edm_model2d.convert_to_fp16()
edm_model1d = DataParallel(edm_model1d).cuda()

edm_policy_config = utils.Config(
    args.edm_policy, 
    savepath=(args.savepath, 'edm_policy_config.pkl'), 
    horizon=args.horizon, 
    action_dim=action_dim, 
    n_action_steps=args.n_action_steps,
    n_obsv_steps = args.n_obsv_steps, 
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
    edm_model1d, None
)

policy_schedule_sampler = create_named_schedule_sampler(
    name=args.policy_schedule_sampler, 
    diffusion=edm_policy
)

edm_policy_trainer_config = utils.Config(
    EDMPolicyTrainer, 
    savepath=(args.savepath, 'edm_policy_trainer_config.pkl'),
    ema_decay=args.ema_decay, 
    train_batch_size=args.train_batch_size, 
    val_batch_size=args.val_batch_size, 
    train_lr=args.policy_learning_rate, 
    lr_anneal_steps=args.policy_learning_rate_anneal_steps, 
    num_train_steps=args.n_train_steps, 
    log_freq=args.log_freq, 
    sample_freq=args.sample_freq, 
    save_freq=args.save_freq, 
    label_freq=10000, 
    use_fp16=args.use_fp16, 
    fp16_scale_growth=args.fp16_scale_growth, 
    results_folder=args.savepath, 
    resume_diffusion_checkpoint=args.diffusion_epoch, 
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

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(train_dataset[10])
t, weights = edm_policy_trainer.schedule_sampler.sample(
     1, device='cuda'
)

policy_loss, _ = edm_policy.loss(
    x_start=batch.actions, 
    sigma=t, 
    cond=batch.conditions
)
policy_loss.backward()
print('Policy -- ✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

# n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
summary_writer = SummaryWriter()

edm_policy_trainer.train(
    summary_writer=summary_writer
)
    
    # policy = trainer.ema_policy
    # env_recon = trainer.env_reconstructor
    # policy.eval()
    # env_recon.eval()
    
    # summary_writer.add_scalar('Epoch_loss/train', epoch_train_loss, i)
    # summary_writer.add_scalar('Epoch_loss/valid', epoch_val_loss, i)
    
    # if args.eval_every and not i % args.eval_every:
    #     with torch.no_grad():
    #         for k in range(5):
    #             batch = utils.batchify(val_dataset[k])
    #             expert_actions, conditions = batch
    #             expert_actions = expert_actions[0].to('cpu').numpy()
    #             plan = policy(conditions)

    #             ground_true_env_map = conditions['env_maps'][:, args.n_env_recon_obsv_steps]
    #             pred_normed_env_map, pred_unnormed_env_map = env_recon(conditions)

    #             if args.pred_action_steps_only:
    #                 pred_actions = plan['action']
    #                 start = args.n_policy_obsv_steps - 1
    #                 end = start + args.n_action_steps
    #                 expert_actions = expert_actions[start:end, :]
    #             else:
    #                 pred_actions = plan['action_pred']

    #             coord_actions = np.array([action_list[a] for a in pred_actions[0]])
    #             normalizer = BitNormalizer(expert_actions, n_bits=args.n_bits)
    #             expert_actions = normalizer.unnormalize(expert_actions)
    #             expert_actions = np.array([action_list[a] for a in expert_actions])
    #             print(f"predicted env map:\n {pred_unnormed_env_map}")
    #             print(f"ground true env map:\n {ground_true_env_map}")
    #             print(f'predicted actions: {coord_actions}')
    #             print(f'expert actions: {expert_actions}')

    #             del batch
    #             del conditions
    #             del expert_actions
    #             del plan
    #             del pred_actions
    #             del coord_actions
    #             del normalizer
    #             del ground_true_env_map
    #             del pred_normed_env_map
    #             del pred_unnormed_env_map

    # policy.train()
    # env_recon.train()

summary_writer.close()
