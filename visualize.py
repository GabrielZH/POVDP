import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import glob 


def visualize(grid_map, target_map):
    if isinstance(grid_map, torch.Tensor):
        grid_map = grid_map.cpu().numpy()
    if isinstance(target_map, torch.Tensor):
        target_map = target_map.cpu().numpy()
    visualize_bev_1 = grid_map.copy().astype(np.uint8)
    visualize_bev_3 = target_map.copy().astype(np.uint8)
    visualize_bev_1[visualize_bev_1 == 0] = 255
    visualize_bev_1[visualize_bev_1 == 1] = 0
    visualize_bev_3[visualize_bev_3 == 0] = 255
    visualize_bev_3[visualize_bev_3 == 1] = 0

    combined_bev = cv2.cvtColor(visualize_bev_1, cv2.COLOR_GRAY2BGR)
    combined_bev[target_map == 1] = [255, 0, 0]

    fig, axis = plt.subplots(1, 1, figsize=(2.6, 2.6), dpi=600)
    axis.imshow(combined_bev, cmap='gray', origin='lower')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.set_xticklabels([])
    axis.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    plt.tight_layout()
    figs = glob.glob("bev_*.png")
    if not figs:
        index = 0
    else:
        indices = [int(f.split('_')[-1].split('.')[0]) for f in figs]
        index = max(indices) + 1
    plt.savefig(f'bev_{index}.png', bbox_inches='tight') 


obsv = torch.load('data/avd/pos_nav/64_-50_-50_50_50__100_100__pos_135_240_None_10_50_8/train/obsv.pt')


indices = set()
for i in [11, 12]:
    # i = np.random.randint(0, len(obsv))
    if i in indices:
        continue
    indices.add(i)
    bev_maps = obsv[str(i)]['bev_map']
    for j, bev_map in enumerate(bev_maps):
        visualize(bev_map[0], bev_map[2])
