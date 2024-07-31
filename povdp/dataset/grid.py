import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_dataset(ds_path, dataset_type='train'):
    episode, obsv, pred, trans = [
        get_data(os.path.join(ds_path, dataset_type), data)
        for data in ['episode', 'obsv', 'pred', 'trans']
    ]
    return episode, obsv, pred, trans

def get_data(ds_path, data):
    data = torch.load(
        os.path.join(
            ds_path, 
            data if data.endswith('.pt') else f'{data}.pt'))

    return data

def sequence_dataset(ds_path, preprocess_fn, ds_type='train'):
    episode, obsv, _, trans = get_dataset(ds_path, dataset_type=ds_type)

    if preprocess_fn:
        obsv = preprocess_fn(obsv)
        trans = preprocess_fn(trans)

    assert len(episode) == len(obsv) == len(trans)

    for index in range(len(episode)):
        proc_episode_data = process_gridworld_single_episode(
            index=str(index), 
            obsv=obsv, 
            trans=trans)
        yield proc_episode_data

def process_gridworld_single_episode(
        index: int, 
        obsv: dict, 
        trans: dict
):
    episode = dict()
    episode['actions'] = trans[index]['actions']
    # episode['actions'] = torch.cat(
    #     [episode['actions'], episode['actions'][-1:]], axis=0)
    # episode['next_actions'] = episode['actions'][1:]
    # episode['actions'] = episode['actions'][:-1]
    episode['rewards'] = trans[index]['rewards']

    episode['feature_maps'] = obsv[index]['feature_map'][:-1]
    # episode['obsv_maps'] = pad(obsv[index]['obsv_maps'][:-1])
    episode['states'] = obsv[index]['poses'][:-1]
    # episode['beliefs'] = torch.zeros_like(
    #     obsv[index]['feature_map'][:-1], dtype=torch.float32)[:, 0]
    # indices = torch.arange(episode['beliefs'].shape[0])
    # episode['beliefs'][
    #     indices, episode['states'][:, 0].int(), 
    #     episode['states'][:, 1].int()] = 1.
    # episode['beliefs'] = pad(episode['beliefs'])

    episode['next_feature_maps'] = obsv[index]['feature_map'][1:]
    # episode['next_obsv_maps'] = pad(obsv[index]['obsv_maps'][1:])
    episode['next_states'] = obsv[index]['poses'][1:]
    # episode['next_beliefs'] = torch.zeros_like(
    #     obsv[index]['feature_map'][1:], dtype=torch.float32)[:, 0]
    # episode['next_beliefs'][
    #     indices, episode['next_states'][:, 0].int(), 
    #     episode['next_states'][:, 1].int()] = 1.
    # episode['next_beliefs'] = pad(episode['next_beliefs'])

    return episode


def collate_fn(batch: list):
    actions = list()
    cond = defaultdict(list)
    for i, sample in enumerate(batch):
        actions.append(torch.as_tensor(sample.actions))
        for k, v in sample.conditions.items():
            cond[k].append(torch.as_tensor(v))
    actions = torch.stack(actions)
    conditions = dict()
    for k, v in cond.items():
        conditions[k] = v
    batch = type(batch[0])(actions, conditions)
    return batch


def pad(x: torch.Tensor):
    # if x.shape[-1] % 2:
    #     x = nn.ConstantPad2d((0, 1, 0, 0), 0)(x)
    # if x.shape[-2] % 2:
    #     x = nn.ConstantPad2d((0, 0, 1, 0), 0)(x)
    h, w = x.shape[-2:]
    h_pad = 2 ** np.ceil(np.log2(h)) - h
    w_pad = 2 ** np.ceil(np.log2(w)) - w
    padding = (
        int(w_pad // 2), 
        int(w_pad - w_pad // 2), 
        int(h_pad // 2), 
        int(h_pad - h_pad // 2)
    )
    return F.pad(x, padding, mode='constant')

