import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import povdp.utils as utils
from povdp.dataset.normalization import BitNormalizer
from povdp.train.trainer import Trainer
from povdp.networks.helpers import unnormalize_env
from calvin.core.domains.gridworld.actions import GridDirs
import pdb

np.set_printoptions(threshold=sys.maxsize)


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'grid_maze_15x15_vr_2'
    config: str = 'config.grid_maze2d'

args = Parser().parse_args('diffusion')
assert args.horizon > max(args.n_policy_obs_steps, args.n_env_recon_obs_steps)

action_list = GridDirs.DIRS_8 + [(0, 0)]

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

train_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'train_dataset_config.pkl'),
    env_size=args.env_size,
    horizon=args.horizon,
    n_policy_obs_steps=args.n_policy_obs_steps, 
    n_env_recon_obs_steps=args.n_env_recon_obs_steps,
    n_action_steps=args.n_action_steps,
    n_latency_steps=args.n_latency_steps,
    normalizers=args.normalizers,
    ds_path=os.path.join(args.dataset_root_path, args.dataset),
    ds_type='train',
    preprocess_fns=args.preprocess_fns,
    use_max_len_padding=args.use_max_len_padding,
    use_condition_padding=args.use_condition_padding,
    max_path_length=args.max_path_length,
    n_bits=args.n_bits,
)
val_dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'val_dataset_config.pkl'),
    env_size=args.env_size,
    horizon=args.horizon,
    n_policy_obs_steps=args.n_policy_obs_steps, 
    n_env_recon_obs_steps=args.n_env_recon_obs_steps,
    n_action_steps=args.n_action_steps,
    n_latency_steps=args.n_latency_steps,
    normalizers=args.normalizers,
    ds_path=os.path.join(args.dataset_root_path, args.dataset),
    ds_type='val',
    preprocess_fns=args.preprocess_fns,
    use_max_len_padding=args.use_max_len_padding,
    use_condition_padding=args.use_condition_padding,
    max_path_length=args.max_path_length,
    n_bits=args.n_bits,
)

# render_config = utils.Config(
#     args.renderer,
#     savepath=(args.savepath, 'render_config.pkl'),
#     env=args.dataset,
# )

train_dataset = train_dataset_config()
val_dataset = val_dataset_config()
action_dim = train_dataset.action_dim
obs_dim = train_dataset.obs_dim
# observation_dim = dataset.observation_dim
# action_dim = dataset.action_dim
# renderer = render_config()

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# model_config = utils.Config(
#     args.model,
#     savepath=(args.savepath, 'model_config.pkl'),
#     horizon=args.horizon,
#     transition_dim=observation_dim + action_dim,
#     cond_dim=observation_dim,
#     dim_mults=args.dim_mults,
#     device=args.device,
# )

diffusion_model1d_config = utils.Config(
    args.diffusion_model1d, 
    savepath=(args.savepath, 'diffusion_model1d_config.pkl'), 
    inp_dim=args.n_bits, 
    obs_cond_dim=args.multi_step_obs_embed_dim,
    env_cond_dim=args.env_embed_dim,
    diffusion_step_embed_dim=args.diffusion_step_embed_dim, 
    dim_mults=args.policy_dim_mults, 
    cond_predict_scale=args.cond_predict_scale,
    env_cond_only=args.env_cond_only,
    device=args.device, 
)
diffusion_model1d = diffusion_model1d_config()


diffusion_model2d_config = utils.Config(
    args.diffusion_model2d, 
    savepath=(args.savepath, 'diffusion_model2d_config.pkl'), 
    inp_dim=args.env_dim,
    global_cond_dim=obs_dim,
    diffusion_step_embed_dim=args.diffusion_step_embed_dim, 
    dim_mults=args.env_recon_dim_mults, 
    cond_predict_scale=args.cond_predict_scale, 
    device=args.device
)
diffusion_model2d = diffusion_model2d_config()

# multi_step_obs_encoder_config = utils.Config(
#     args.multi_step_obs_encoder, 
#     savepath=(args.savepath, 'multi_step_obs_encoder_config.pkl'), 
#     inp_dim=obs_dim, 
#     out_dim=args.multi_step_obs_embed_dim, 
#     device=args.device,
# )
# multi_step_obs_encoder = multi_step_obs_encoder_config()

vision_env_encoder_config = utils.Config(
    args.env_map_encoder, 
    savepath=(args.savepath, 'vision_env_encoder_config.pkl'), 
    inp_channels=args.env_dim + 1 \
        if args.include_goal_beliefs else args.env_dim, 
    out_channels=args.env_embed_dim, 
    hidden_channels=args.env_encoder_hidden_dims,
    device=args.device,
)
env_map_encoder = vision_env_encoder_config()

diffusion_policy_config = utils.Config(
    args.diffusion_policy,
    savepath=(args.savepath, 'diffusion_policy_config.pkl'),
    horizon=args.horizon,
    action_dim=action_dim, 
    obs_dim=obs_dim, 
    n_action_steps=args.n_action_steps,
    n_policy_obs_steps=args.n_policy_obs_steps,
    n_env_recon_obs_steps=args.n_env_recon_obs_steps,
    env_cond_only=args.env_cond_only,
    obs_as_cond_type=args.obs_as_policy_cond_type,
    n_timesteps=args.n_policy_diffusion_steps, 
    loss_type=args.loss_type, 
    clip_denoised=args.clip_denoised, 
    predict_epsilon=args.predict_epsilon, 
    pred_action_steps_only=args.pred_action_steps_only,
    bit_scale=args.bit_scale, 
    ## loss weighting
    action_weight=args.action_weight, 
    loss_weights=args.loss_weights, 
    loss_discount=args.loss_discount, 
    device=args.device, 
)
diffusion_policy = diffusion_policy_config(
    diffusion_model1d, env_map_encoder, None)

env_recon_config = utils.Config(
    args.env_reconstructor, 
    savepath=(args.savepath, 'env_reconstructor_config.pkl'), 
    horizon=args.horizon, 
    n_policy_obs_steps=args.n_policy_obs_steps,
    n_env_recon_obs_steps=args.n_env_recon_obs_steps,
    obs_as_cond_type=args.obs_as_env_recon_cond_type,
    n_timesteps=args.n_env_recon_diffusion_steps,
    clip_denoised=args.clip_denoised, 
    predict_epsilon=args.predict_epsilon, 
    device=args.device,
)
env_reconstructor = env_recon_config(diffusion_model2d)

trainer_config = utils.Config(
    Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.train_batch_size,
    val_batch_size=args.val_batch_size,
    train_lr=args.learning_rate,
    num_train_steps=args.n_train_steps,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    separate_training=args.separate_training,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=10000,
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)
# trainer = trainer_config(diffusion, dataset, renderer)
trainer = trainer_config(
    diffusion_policy, 
    env_reconstructor, 
    train_dataset, 
    val_dataset
)

epoch = '0'
if args.load_from_ckpt:
    epoch = args.diffusion_epoch
    if epoch == 'latest':
        epoch = utils.get_latest_epoch(
            args.logbase, 
            args.dataset, 
            args.arch_variation, 
            args.diffusion_loadpath
        )
    print(f"epoch: {epoch}")
    trainer.load(epoch)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

# utils.report_parameters(diffusion_model1d)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(train_dataset[0])
policy_loss, _ = diffusion_policy.loss(*batch)
policy_loss.backward()
env_recon_loss = env_reconstructor.loss(
    x=batch.conditions['env_maps'][:, env_reconstructor.n_env_recon_obs_steps], 
    cond=batch.conditions)
env_recon_loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
summary_writer = SummaryWriter()

for i in range(int(epoch) // args.n_steps_per_epoch, n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    if args.separate_training:
        epoch_train_loss, epoch_val_loss = trainer.train_separate(
            n_train_steps=args.n_steps_per_epoch, 
            summary_writer=summary_writer, 
            epoch_i=i)
    else:
        epoch_train_loss, epoch_val_loss = trainer.train_joint(
            n_train_steps=args.n_steps_per_epoch, 
            summary_writer=summary_writer, 
            epoch_i=i)
    policy = trainer.ema_policy
    env_recon = trainer.env_reconstructor
    policy.eval()
    env_recon.eval()
    
    summary_writer.add_scalar('Epoch_loss/train', epoch_train_loss, i)
    summary_writer.add_scalar('Epoch_loss/valid', epoch_val_loss, i)
    
    if args.eval_every and not i % args.eval_every:
        with torch.no_grad():
            for k in range(5):
                batch = utils.batchify(val_dataset[k])
                expert_actions, conditions = batch
                expert_actions = expert_actions[0].to('cpu').numpy()
                plan = policy(conditions)

                ground_true_env_map = conditions['env_maps'][:, args.n_env_recon_obs_steps]
                pred_normed_env_map, pred_unnormed_env_map = env_recon(conditions)

                if args.pred_action_steps_only:
                    pred_actions = plan['action']
                    start = args.n_policy_obs_steps - 1
                    end = start + args.n_action_steps
                    expert_actions = expert_actions[start:end, :]
                else:
                    pred_actions = plan['action_pred']

                coord_actions = np.array([action_list[a] for a in pred_actions[0]])
                normalizer = BitNormalizer(expert_actions, n_bits=args.n_bits)
                expert_actions = normalizer.unnormalize(expert_actions)
                expert_actions = np.array([action_list[a] for a in expert_actions])
                print(f"predicted env map:\n {pred_unnormed_env_map}")
                print(f"ground true env map:\n {ground_true_env_map}")
                print(f'predicted actions: {coord_actions}')
                print(f'expert actions: {expert_actions}')

                del batch
                del conditions
                del expert_actions
                del plan
                del pred_actions
                del coord_actions
                del normalizer
                del ground_true_env_map
                del pred_normed_env_map
                del pred_unnormed_env_map

    policy.train()
    env_recon.train()

summary_writer.close()