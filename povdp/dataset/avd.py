from typing import Union, Tuple, List
import os
from pathlib import Path
from collections import namedtuple, defaultdict
from typing import List

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from tqdm import tqdm

from calvin.core.domains.avd.dataset.data_classes import Scene
from calvin.core.domains.avd.dataset.scene_manager import AVDSceneManager
from calvin.core.domains.avd.navigation.pos_nav.position_map import (
    PositionNode, 
    get_node_image_names
)


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


def sequence_dataset(
        ds_path, 
        preprocess_fn, 
        ds_type='train', 
        # resize=None, 
        # target=None, 
        # target_size_ratio=None, 
        # target_dist_thresh=None, 
        # ori_res=None, 
        # in_ram=False
):
    episode, obsv, _, trans = get_dataset(ds_path, dataset_type=ds_type)
    if preprocess_fn:
        obsv = preprocess_fn(obsv)
        trans = preprocess_fn(trans)

    assert len(episode) == len(obsv) == len(trans)

    # data_path = Path(ds_path)
    # scene_path = os.path.join(
    #     str(data_path.parent.parent), 'src'
    # )
    # scenes = AVDSceneManager(
    #     data_dir=scene_path, 
    #     scene_resize=resize, 
    #     target=target, 
    #     in_ram=in_ram, 
    #     target_size_ratio=target_size_ratio, 
    #     target_dist_thresh=target_dist_thresh, 
    #     avd_workers=4, 
    # )
    for index in range(len(episode)):
        proc_episode_data = process_avd_single_episode(
            index=str(index), 
            obsv=obsv, 
            trans=trans
        )
        yield proc_episode_data


# def process_avd_single_episode(
#         index: int, 
#         episode_info: dict, 
#         obsv: dict, 
#         trans: dict, 
#         scenes: AVDSceneManager, 
#         ori_res: int = None
#     ):
#     scene_name = episode_info[index]['scene_name']
#     scene: Scene = scenes[scene_name]
#     episode = dict()
#     episode['target'] = episode_info[index]['target']
#     episode['target_emb'] = episode_info[index]['target_emb']
#     episode['target_grid'] = episode_info[index]['targets_grid']
#     episode['occupancy'] = episode_info[index]['occupancy']

#     episode['actions'] = trans[index]['actions']
#     episode['rewards'] = trans[index]['rewards']
#     episode['dones'] = trans[index]['dones']

#     # episode['state_info'] = obsv[index]['state_info']
#     episode['poses'] = obsv[index]['poses'][:-1]
#     if ori_res:
#         image_names = obsv[index]['state_info'][:-1]
#     else:
#         image_names = list()
#         for node_name in obsv[index]['state_info'][:-1]:
#             image_names += get_node_image_names(node_name)
    
#     output = get_3d_observations(scene, image_names)
#     rgb = output['rgb']
#     depth = output['depth']
#     emb = output['emb']
#     valids = output['valids']
#     coords = output['coords']
#     flat_rgb = output['flat_rgb']
#     flat_valids = output['flat_valids']
#     flat_points = output['flat_points']

#     if ori_res:
#         # rgb = rearrange(rgb, 'b h w f -> b f h w')
#         coords = rearrange(coords, 'b h w f -> b f h w')
#     else:
#         n_ori = len(get_node_image_names(obsv[index]['state_info'][0]))
#         # rgb = rearrange(rgb, '(b o) h w f -> b f (o h) w', o=n_ori)
#         if emb is not None:
#             emb = rearrange(emb, '(b o) f h w -> b f (o h) w', o=n_ori)
#         valids = rearrange(valids, '(b o) h w -> b (o h) w', o=n_ori)
#         coords = rearrange(coords, '(b o) h w f -> b f (o h) w', o=n_ori)

#     episode['flat_points'] = flat_points

#     # episode['rgb'] = torch.from_numpy(rgb).float() / 255
#     if emb is not None: episode['emb'] = emb
#     # episode['rgb'] = rgb
#     # episode['valid_points'] = torch.from_numpy(valids).bool()
#     # episode['surf_xyz'] = torch.from_numpy(coords).float()

#     return episode


def process_avd_single_episode(
        index: Union[int, str], 
        obsv: dict, 
        trans: dict
):
    episode = dict()
    episode['actions'] = trans[index]['actions']
    episode['rewards'] = trans[index]['rewards']

    episode['feature_maps'] = obsv[index]['bev_map'][:-1]
    episode['states'] = obsv[index]['poses'][:-1]

    episode['next_feature_maps'] = obsv[index]['bev_map'][1:]
    episode['next_states'] = obsv[index]['poses'][1:]

    return episode


def get_3d_observations(
        scene: Scene, 
        image_names: List[str], 
        target_obj_name: str = None
):
    indices = scene.names_to_indices(image_names)
    rgb = scene.rgb_images[indices]
    depth = scene.depth_images[indices] / scene.scale
    emb = scene.embeddings[indices] if scene.embeddings is not None else None
    
    valids = depth != 0
    coords = scene.coords(image_names)

    flat_coords = coords.reshape(-1, coords.shape[-1])
    flat_rgb = rgb.reshape(-1, rgb.shape[-1])
    flat_valids = valids.reshape(np.prod(valids.shape))
    # flat_coords = voxel_downsample(flat_coords, voxel_size=.1)
    # flat_rgb = voxel_downsample(flat_rgb, voxel_size=.1)
    # flat_valids = voxel_downsample
    # Combine main points and target points
    
    target_coords, target_rgb, target_depth = scene.target_coords_rgb_depth(
        image_names=image_names[-6:], 
        target_obj_name=target_obj_name
    )
    if len(target_coords) and len(target_rgb) and len(target_depth):
        target_valids = (target_depth / scene.scale) != 0
        target_valids = target_valids.reshape(np.prod(target_valids.shape))

        # print(f"flat_coords: {flat_coords.shape}")
        # print(f"flat_target_coords: {target_coords.shape}")
        # print(f"flat_rgb: {flat_rgb.shape}")
        # print(f"flat_target_rgb: {target_rgb.shape}")
        # print(f"flat_valids: {flat_valids.shape}")
        # print(f"flat_target_valids: {target_valids.shape}")

        # Downsample main points
        main_coords, main_indices, _ = voxel_downsample_and_trace(
            flat_coords, 
            voxel_size=.025
        )
        main_rgb = flat_rgb[main_indices[:, 0]]
        main_valids = flat_valids[main_indices[:, 0]]

        # Downsample target points
        target_coords, target_indices, _ = voxel_downsample_and_trace(
            target_coords, 
            voxel_size=.025
        )
        target_rgb = target_rgb[target_indices[:, 0]]
        target_valids = target_valids[target_indices[:, 0]]

        main_points = np.concatenate([main_coords, main_rgb], axis=-1)
        main_points = main_points[main_valids]

        target_points = np.concatenate([target_coords, target_rgb], axis=-1)
        target_points = target_points[target_valids]

        xyz_min = np.amin(main_points, axis=0)[:3]
        main_points[:, :3] -= xyz_min
        target_points[:, :3] -= xyz_min

    else:
        flat_coords, indices, _ = voxel_downsample_and_trace(
            flat_coords, 
            voxel_size=.025
        )
        flat_rgb = flat_rgb[indices[:, 0]]
        flat_valids = flat_valids[indices[:, 0]]

        main_points = np.concatenate([flat_coords, flat_rgb], axis=-1)
        main_points = main_points[flat_valids]
        xyz_min = np.amin(main_points, axis=0)[:3]
        main_points[:, :3] -= xyz_min
        target_points = None

    return {
        'rgb': rgb, 
        'depth': depth, 
        'emb': emb, 
        'valids': valids, 
        'coords': coords, 
        'flat_rgb': flat_rgb, 
        'flat_valids': flat_valids, 
        'flat_coords': flat_coords, 
        'flat_points': main_points, 
        'flat_target_points': target_points
    }

def create_bev(
        points: np.ndarray, 
        target_points: np.ndarray,
        target_grid_size: Tuple[int, int], 
        visualize_grid: bool = False
):
    bev_map_1 = np.zeros(target_grid_size, dtype=np.uint8)
    bev_map_2 = np.zeros_like(bev_map_1)
    bev_map_3 = np.zeros_like(bev_map_1)
    points = filter_outliers(points, threshold=2)

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    y_floor = np.percentile(y_coords, 25)
    y_ceiling = np.percentile(y_coords, 80)

    height_filter = (y_coords > y_floor) & (y_coords < y_ceiling)
    x_coords = x_coords[height_filter]
    z_coords = z_coords[height_filter]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_z, max_z = np.min(z_coords), np.max(z_coords)
    norm_x_coords = (x_coords - min_x) / (max_x - min_x) * (target_grid_size[1] - 1)
    norm_z_coords = (z_coords - min_z) / (max_z - min_z) * (target_grid_size[0] - 1)

    for i in range(len(x_coords)):
        grid_x = int(norm_x_coords[i])
        grid_y = int(norm_z_coords[i])
        if 0 <= grid_y < target_grid_size[0] and 0 <= grid_x < target_grid_size[1]:
            bev_map_1[grid_y, grid_x] = 1
    
    contours, _ = cv2.findContours(bev_map_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_points = np.vstack(contours).squeeze()
    convex_hull = cv2.convexHull(all_points)
    cv2.drawContours(bev_map_2, [convex_hull], -1, 1, thickness=cv2.FILLED)
    bev_map_2[bev_map_1 == 1] = 0

    if target_points is not None and len(target_points) > 0:
        target_x_coords = target_points[:, 0]
        target_z_coords = target_points[:, 2]
        target_norm_x_coords = (target_x_coords - min_x) / (max_x - min_x) * (target_grid_size[1] - 1)
        target_norm_z_coords = (target_z_coords - min_z) / (max_z - min_z) * (target_grid_size[0] - 1)
        for i in range(len(target_x_coords)):
            target_grid_x = int(target_norm_x_coords[i])
            target_grid_y = int(target_norm_z_coords[i])
            if 0 <= target_grid_y < target_grid_size[0] and 0 <= target_grid_x < target_grid_size[1]:
                bev_map_3[target_grid_y, target_grid_x] = 1
    bev_map_3 = cleanup_bev(bev_map_3, min_size=3)

    bev_map = np.stack((bev_map_1, bev_map_2, bev_map_3), axis=0)

    if visualize_grid:
        visualize_bev_1 = bev_map_1.copy()
        visualize_bev_2 = bev_map_2.copy()
        visualize_bev_3 = bev_map_3.copy()
        visualize_bev_1[visualize_bev_1 == 0] = 255
        visualize_bev_1[visualize_bev_1 == 1] = 0
        visualize_bev_2[visualize_bev_2 == 0] = 255
        visualize_bev_2[visualize_bev_2 == 1] = 0
        visualize_bev_3[visualize_bev_3 == 0] = 255
        visualize_bev_3[visualize_bev_3 == 1] = 0

        combined_bev = cv2.cvtColor(visualize_bev_1, cv2.COLOR_GRAY2BGR)
        combined_bev[bev_map_3 == 1] = [255, 0, 0]
        
        # contour_visualization = np.zeros((*target_grid_size, 3), dtype=np.uint8)
         # Draw each contour with a different color
        # for contour in contours:
        #     color = tuple(np.random.randint(0, 256, 3).tolist())
        #     cv2.drawContours(contour_visualization, [contour], -1, color, thickness=1)

        fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), dpi=600)
        axes[0].imshow(combined_bev, cmap='gray', origin='lower')
        axes[1].imshow(visualize_bev_2, cmap='gray', origin='lower')
        axes[2].imshow(visualize_bev_3, cmap='gray', origin='lower')
        # axes[2].imshow(contour_visualization, origin='lower')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        plt.tight_layout()
        figs = glob.glob("bev_*.png")
        if not figs:
            index = 0
        else:
            indices = [int(f.split('_')[-1].split('.')[0]) for f in figs]
            index = max(indices) + 1
        plt.savefig(f'bev_{index}.png', bbox_inches='tight')

    return bev_map


def cleanup_bev(bev_map, min_size=1):
    # Get the connected components and their stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bev_map, 
        connectivity=8
    )
    
    # Create an output image to store the cleaned-up result
    cleaned_bev_map = np.zeros_like(bev_map)
    
    # Iterate through the components and keep only the large ones
    for i in range(1, num_labels):  # Skip the background component (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_bev_map[labels == i] = 1
    
    return cleaned_bev_map


def fill_free_space(all_points, target_grid_size, axis):
    free_space_map = np.zeros(target_grid_size, dtype=np.uint8)
    unique_values = np.unique(all_points[:, axis])
    last_min, last_max = None, None

    for val in unique_values:
        indices = np.where(all_points[:, axis] == val)[0]
        other_values = all_points[indices, 1 - axis]
        min_other, max_other = np.min(other_values), np.max(other_values)

        if min_other == max_other:
            if last_min is not None and last_max is not None:
                min_other = min(min_other, last_min)
                max_other = max(max_other, last_max)

        if max_other - min_other == 0 and val + 1 < target_grid_size[axis]:
            next_other_values = all_points[np.where(all_points[:, axis] == val + 1), 1 - axis]
            if next_other_values.size > 0:
                min_other = min(min_other, np.min(next_other_values))
                max_other = max(max_other, np.max(next_other_values))

        last_min, last_max = min_other, max_other

        if axis == 0:
            free_space_map[val, min_other:max_other + 1] = 1
        else:
            free_space_map[min_other:max_other + 1, val] = 1

    return free_space_map


def get_bev_params(
        points: np.ndarray, 
        target_grid_size: Tuple[int, int]
):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    point_cloud_size = max(x_max - x_min, z_max - z_min)
    bound_size = point_cloud_size * 2.
    resolution = bound_size / max(target_grid_size)
    # origin = np.array([(x_min + x_max) / 2, (z_min + z_max) / 2])
    origin = np.array([0., 0.])

    return resolution, origin


def init_bev(grid_size: Tuple[int, int]):
    return np.zeros(grid_size, dtype=np.uint8)


def update_bev(
        bev_map: np.ndarray, 
        points: np.ndarray, 
        origin: np.ndarray, 
        resolution: float, 
        visualize_grid: bool = False
):
    # Filter out outlier points, those too far away from origin
    filtered_points = filter_outliers(points, threshold=2)
    # x_coords = filtered_points[:, 0]
    # z_coords = filtered_points[:, 2]
    # coords = np.vstack((x_coords, z_coords)).T
    coords = filtered_points[:, [0, 2]]
    # Transform point coordinates to grid coordinates
    grid_coords = ((coords - origin) / resolution).astype(int)
    valid_indices = (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < bev_map.shape[0]) & \
                    (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < bev_map.shape[1])
    # Update the BEV map: set cells to '1' to indicate explored/visible areas
    bev_map[grid_coords[valid_indices][:, 1], grid_coords[valid_indices][:, 0]] = 1

    if visualize_grid:
        visualize_bev = bev_map.copy()
        visualize_bev[visualize_bev == 0] = 255
        visualize_bev[visualize_bev == 1] = 0
        fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=600, ncols=1, nrows=1, layout='tight')
        ax.imshow(visualize_bev, cmap='gray', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        # axs[0].add_artist(ScaleBar(8, 'nm', location='lower right', box_color='k', box_alpha=0.5, color='w', width_fraction=0.01))
        figs = glob.glob("bev_*.png")
        if not figs:
            index = 0
        else:
            indices = [int(f.split('_')[-1].split('.')[0]) for f in figs]
            index = max(indices) + 1
        plt.savefig(f'bev_{index}.png', bbox_inches='tight')

    return bev_map


def filter_outliers(
        points: np.ndarray, 
        threshold: float = 3.
):
    # Calculate mean and standard deviation
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)

    filtered_points = points[np.all(np.abs(points - mean) < threshold * std, axis=1)]

    return filtered_points


def voxel_downsample(points, voxel_size=.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downpcd = pcd.voxel_down_sample(voxel_size)
    
    return np.asarray(downpcd.points)


def voxel_downsample_and_trace(points, voxel_size=.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd, voxel_indices, indices = pcd.voxel_down_sample_and_trace(
        voxel_size, 
        pcd.get_min_bound(), 
        pcd.get_max_bound()
    )
    voxel_indices = np.asarray(voxel_indices)
    indices = [np.asarray(index) for index in indices]

    return np.asarray(down_pcd.points), voxel_indices, indices


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
        if k in {
            'target', 
            'target_grid', 
            'target_emb', 
            'occupancy'
        }:
            try: conditions[k] = torch.stack(v)
            except: conditions[k] = v
        else:
            try: conditions[k] = torch.cat(v)
            except: conditions[k] = v
    # elem = next(iter(cond.values()))
    elem = cond['rgb']
    conditions[f"lens"] = torch.tensor(
        [len(val) 
         for i, val in enumerate(elem)], 
         dtype=torch.long
    )
    conditions[f"index"] = torch.cat(
        [torch.full(
            (len(val),), i, dtype=torch.long) 
        for i, val in enumerate(elem)]
    )
    conditions[f"step"] = torch.cat(
        [torch.arange(
            len(val), 
            dtype=torch.long) 
        for i, val in enumerate(elem)]
    )
   
    batch = type(batch[0])(actions, conditions)
    return batch
