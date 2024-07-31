import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple
from povdp.utils.resample import create_named_schedule_sampler


GeneralExperiment = namedtuple(
    'GeneralExperiment', 
    {
        'train_dataset',
        'val_dataset',
    }
)


EDMPolicyTestExperiment = namedtuple(
    'EDMPolicyTestExperiment', 
    (
        'train_dataset '
        'val_dataset '
        'edm_policy '
        'edm_policy_trainer '
        'epoch'
    )
)


DiffusionExperiment = namedtuple(
    'DiffusionExperiment', 
    (
        'train_dataset, '
        'val_dataset, ' 
        'diffusion_model1d, '
        'diffusion_policy, '
        'ema_policy, '
        'env_encoder, '
        'trainer, '
        'epoch'
    )
)

ValueExperiment = namedtuple(
    'ValueExperiment', 
    (
        'train_dataset', 
        'val_dataset', 
        'value_function', 
        'belief_filter', 
        'value_trainer', 
        'epoch'
    )
)

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(*loadpath, ema_decay=None):
    states = glob.glob1(os.path.join(*loadpath), 'policy_ema_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace(f'policy_ema_{ema_decay}_state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config


def load_dataset(
        *loadpath, 
        train_dataset_config='train_dataset_config',
        val_dataset_config='val_dataset_config', 
        device='cuda:0', 
        **kwargs
):
    train_dataset_config, val_dataset_config = [
        load_config(*loadpath, config + '.pkl') \
            if not config.endswith('.pkl') else load_config(*loadpath, config) 
        for config in ( 
        train_dataset_config, 
        val_dataset_config
        )
    ]

    train_dataset = train_dataset_config()
    val_dataset = val_dataset_config()

    return GeneralExperiment(
        train_dataset, 
        val_dataset
    )

def load_diffusion(
        *loadpath, 
        epoch='latest', 
        train_dataset_config='train_dataset_config',
        val_dataset_config='val_dataset_config', 
        model1d_config='diffusion_model1d_config',
        vision_env_encoder_config='vision_env_encoder_config',
        diffusion_policy_config='diffusion_policy_config',
        schedule_sampler='lognormal', 
        trainer_config='trainer_config', 
        device='cuda:0', 
        **kwargs):
    train_dataset_config, val_dataset_config, \
    model1d_config, \
    vision_env_encoder_config, \
    diffusion_policy_config, \
    trainer_config = [
        load_config(*loadpath, config + '.pkl') \
            if not config.endswith('.pkl') else load_config(*loadpath, config) 
        for config in ( 
        train_dataset_config, val_dataset_config, 
        model1d_config, 
        vision_env_encoder_config, 
        diffusion_policy_config, 
        trainer_config)
    ]

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    train_dataset = train_dataset_config()
    val_dataset = val_dataset_config()
    # renderer = render_config()
    diffusion_model1d = model1d_config()
    env_encoder = vision_env_encoder_config()
    diffusion_policy = diffusion_policy_config(
        diffusion_model1d, 
        env_encoder, 
        None
    )

    # trainer = trainer_config(diffusion, dataset, renderer)
    trainer = trainer_config(
        diffusion_policy, 
        train_dataset, 
        val_dataset)

    if epoch == 'latest':
        epoch = get_latest_epoch(*loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(
        train_dataset, 
        val_dataset, 
        diffusion_model1d,
        diffusion_policy, 
        trainer.ema_policy, 
        env_encoder,
        trainer, 
        epoch)


def load_diffusion_edm(
        *loadpath, 
        epoch='latest', 
        point_cloud_encoder_epoch=None, 
        flat_map_encoder_epoch=None, 
        ema_decay=None, 
        schedule_sampler='uniform', 
        train_dataset_config='train_dataset_config',
        val_dataset_config='val_dataset_config', 
        model1d_config='edm_model1d_config',
        point_cloud_to_2d_config=None, 
        policy_config='edm_policy_config', 
        trainer_config='edm_policy_trainer_config', 
        device='cuda:0', 
        **kwargs, 
):
    train_dataset_config, val_dataset_config, \
    model1d_config, point_cloud_to_2d_config, \
    policy_config, trainer_config = [
        load_config(*loadpath, config + '.pkl') \
            if config is not None and not config.endswith('.pkl') 
            else load_config(*loadpath, config) if config is not None else None
        for config in ( 
        train_dataset_config, val_dataset_config, 
        model1d_config, point_cloud_to_2d_config, 
        policy_config, trainer_config)
    ]

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)
    print(f"train_dataset_config: {train_dataset_config}")

    train_dataset = train_dataset_config()
    val_dataset = val_dataset_config()
    # renderer = render_config()
    edm_model1d = model1d_config()
    point_cloud_to_2d_projector = None
    if point_cloud_to_2d_config is not None:
        point_cloud_to_2d_projector = point_cloud_to_2d_config()
    edm_policy = policy_config(
        edm_model1d, 
        point_cloud_to_2d_projector
    )
    schedule_sampler = create_named_schedule_sampler(
        name=schedule_sampler, 
        diffusion=edm_policy
    )
    # trainer = trainer_config(diffusion, dataset, renderer)
    trainer = trainer_config(
        edm_policy, 
        train_dataset, 
        val_dataset, 
        schedule_sampler
    )

    if epoch == 'latest':
        epoch = get_latest_epoch(*loadpath, ema_decay=ema_decay)
        epoch = f'{epoch:04d}'

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    policy_ckpt_loadpath = os.path.join(*loadpath, f'policy_ema_{ema_decay}_state_{epoch}.pt')
    policy_ckpt = torch.load(policy_ckpt_loadpath)
    # new_policy_ckpt = dict()
    # for key, value in policy_ckpt.items():
    #     if key.startswith('module.'):
    #         new_key = key.replace('module.', '')
    #         new_policy_ckpt[new_key] = value
    edm_policy.diffusion_model.load_state_dict(policy_ckpt, strict=False)
    if point_cloud_to_2d_projector is not None:
        if point_cloud_encoder_epoch is None and flat_map_encoder_epoch is None:
            projector_ckpt_loadpath = os.path.join(*loadpath, f'point_to_2d_{epoch}.pt')
            projector_ckpt = torch.load(projector_ckpt_loadpath)
            edm_policy.point_cloud_to_2d_projector.load_state_dict(projector_ckpt)
        elif point_cloud_encoder_epoch is not None and flat_map_encoder_epoch is not None:
            point_cloud_encoder_ckpt_loadpath = os.path.join(
                *loadpath, 
                f'point_cloud_encoder_{point_cloud_encoder_epoch}.pt'
            )
            point_cloud_encoder_ckpt = torch.load(point_cloud_encoder_ckpt_loadpath)
            edm_policy.point_cloud_to_2d_projector.point_conv_net.load_state_dict(
                point_cloud_encoder_ckpt
            )
            flat_map_encoder_ckpt_loadpath = os.path.join(
                *loadpath, 
                f'flat_map_encoder_{flat_map_encoder_epoch}.pt'
            )
            flat_map_encoder_ckpt = torch.load(flat_map_encoder_ckpt_loadpath)
            edm_policy.point_cloud_to_2d_projector.map_conv_net.load_state_dict(
                flat_map_encoder_ckpt
            )
        else:
            raise NotImplementedError

    return EDMPolicyTestExperiment(
        train_dataset, 
        val_dataset, 
        edm_policy,
        trainer, 
        epoch
    )


def load_value_guidance(
        *loadpath, 
        epoch='latest', 
        train_value_dataset_config='train_value_dataset_config',
        val_value_dataset_config='val_value_dataset_config', 
        belief_filter_config='belief_filter_config', 
        value_function_config='value_function_config', 
        value_trainer_config='value_trainer_config', 
        device='cuda:0', 
        **kwargs):
    train_value_dataset_config, val_value_dataset_config, \
    belief_filter_config, value_function_config, \
    value_trainer_config = [
        load_config(*loadpath, config + '.pkl') \
            if not config.endswith('.pkl') else load_config(*loadpath, config) 
        for config in ( 
            train_value_dataset_config, val_value_dataset_config,
            belief_filter_config, value_function_config,  
            value_trainer_config)
    ]

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    value_trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    train_value_dataset = train_value_dataset_config()
    val_value_dataset = val_value_dataset_config()
    belief_filter = belief_filter_config()
    value_function = value_function_config()

    # trainer = trainer_config(diffusion, dataset, renderer)
    value_trainer = value_trainer_config(
        value_function, 
        belief_filter, 
        train_value_dataset, 
        val_value_dataset
    )

    if epoch == 'latest':
        epoch = get_latest_epoch(*loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    value_trainer.load(epoch)

    return ValueExperiment(
        train_value_dataset, 
        val_value_dataset, 
        value_function, 
        belief_filter,
        value_trainer, 
        epoch)
