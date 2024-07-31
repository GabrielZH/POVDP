import json
import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from core.model import Model
from core.utils.logger import MetaLogger
from core.utils.tensor_utils import to_numpy
from core.utils.utils import Stats


torch.set_printoptions(threshold=sys.maxsize)


class Trainer:
    def __init__(self, model: Model, optimizer, config=None,
                 checkpoint: str = None, clip: float = None, clear: bool = False, save_interval: int = 300, **kwargs):
        self.config = config
        self.device = model.device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.clip = clip
        self.clear = clear
        self.save_interval = save_interval
        self.start_time = time.time()

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def save_checkpoint(self, checkpoint_path, **data):
        dirpath = os.path.dirname(checkpoint_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        torch.save({
            'arch': type(self.model).__name__,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            **data
        }, checkpoint_path)
        with open(checkpoint_path+".json", "w") as f:
            json.dump(self.config, f)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        with open(checkpoint_path+".json", "r") as f:
            self.config = json.load(f)

    def predict(self, inputs: dict, is_train: bool, **settings):
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        self.model.train() if is_train else self.model.eval()
        with torch.enable_grad() if is_train else torch.no_grad():
            if hasattr(self.model, "preprocess"): self.model.preprocess(inputs, **settings)
            # print(f"inputs: {inputs.keys()}")
            # print(f"curr_poses: {inputs['curr_poses'][0]}")
            # print(f"curr_feature_map:\n{inputs['curr_feature_map'][0, 0]}")
            # print(f"feature_map:\n{inputs['feature_map'][0, 0]}")
            # print(f"curr_actions: {inputs['curr_actions'][0]}")
            # print(f"post_poses: {inputs['post_poses'][0]}")
            # print(f"post_feature_map:\n{inputs['post_feature_map'][0, 0]}")
        
            outputs = self.model(**inputs, **settings)

        return outputs

    def forward_pass(self, inputs: dict, is_train: bool):
        # print(f"inputs: {inputs.keys()}")
        # print(f"feature_map: {inputs['feature_map'].shape}")
        # print(f"curr_feature_map: {inputs['curr_feature_map'].shape}")
        # print(f"post_feature_map: {inputs['post_feature_map'].shape}")
        # print(f"curr_poses: {inputs['curr_poses'], inputs['curr_poses'].shape}")
        # print(f"curr_actions: {inputs['curr_actions'], inputs['curr_actions'].shape}")
        # print(f"post_poses: {inputs['post_poses'], inputs['post_poses'].shape}")
        # print(f"poses: {inputs['poses'], inputs['poses'].shape}")
        # print(f"next_poses: {inputs['next_poses'], inputs['next_poses'].shape}")
        # print(f"actions: {inputs['actions'], inputs['actions'].shape}")
        # print(f"_poses: {inputs['_poses'], inputs['_poses'].shape}")
        # print(f"_next_poses: {inputs['_next_poses'], inputs['_next_poses'].shape}")
        # print(f"_actions: {inputs['_actions'], inputs['_actions'].shape}")
        
        outputs = self.predict(inputs, is_train)

        loss_batch, loss_outputs = self.model.loss(**{**inputs, **outputs})
        stats = {'loss': loss_batch.item()}
        if hasattr(self.model, "metrics"):
            stats = {**stats, **self.model.metrics(**{**inputs, **outputs, **loss_outputs})}

        return {**outputs, **loss_outputs}, loss_batch, stats

    def fit_epoch(self, loader: DataLoader, is_train=True, visualize=False, epochs=1, save_path=None, logger: MetaLogger=None, **settings):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        self.model.train() if is_train else self.model.eval()

        with torch.enable_grad() if is_train else torch.no_grad():
            sum_acc, num_batches, i_batch = 0.0, len(loader) * epochs, 0
            stats_collector = Stats()
            start_time = time.time()

            for _ in range(epochs):
                last_saved = None
                for i, inputs in enumerate(loader):  # Loop over batches of data
                    inputs = {**inputs, **settings}
                    outputs, loss_batch, stats = self.forward_pass(inputs, is_train)
                    
                    if visualize and i and not i % 100:
                        # Grid map figure
                        colors_grid = [(0.7, 0.7, 1), (0, 0, 0.7), (1, 1, 1)]  # Light Blue -> Medium Blue -> White
                        n_bins = 100
                        cmap_name_grid = 'obstacle_free_space'
                        cm_grid = mcolors.LinearSegmentedColormap.from_list(cmap_name_grid, colors_grid, N=n_bins)

                        fig_grid, ax_grid = plt.subplots(figsize=(2.6, 2.6), dpi=600)
                        ax_grid.imshow(1.0 - inputs['curr_feature_map'][0, 0].cpu(), cmap=cm_grid)
                        ax_grid.set_xticks([])
                        ax_grid.set_yticks([])
                        ax_grid.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
                        plt.savefig(f'plot/grid_map_{i}.png', bbox_inches='tight')
                        plt.close(fig_grid)

                        # Value map figure
                        fig_value, ax_value = plt.subplots(figsize=(2.6, 2.6), dpi=600)
                        # ax_value.imshow(1.0 - inputs['curr_feature_map'][0, 0], cmap=cm_grid)  # Overlay grid with semi-transparency
                        ax_value.set_xticks([])
                        ax_value.set_yticks([])
                        ax_value.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
                        colors_value = [(0, 0, 0.7), (1, 1, 0.5)]
                        cmap_name_value = "yellow_blue"
                        cm_value = mcolors.LinearSegmentedColormap.from_list(cmap_name_value, colors_value, N=n_bins)

                        ax_value.imshow(outputs['v'][0, 0].detach().cpu().numpy(), cmap=cm_value)
                        plt.savefig(f'plot/value_func_{i}.png', bbox_inches='tight')

                    stats_collector.add_all(stats)

                    if is_train:
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # backward pass
                        loss_batch.backward()
                        if self.clip:
                            # clip gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    if is_train:
                        self.optimizer.step()

                    if save_path:
                        needs_updating = last_saved is None or time.time() > last_saved + self.save_interval
                        if needs_updating:
                            torch.save(to_numpy({
                                **inputs, **outputs, 'saved_time': datetime.now().strftime('%m%d_%H%M%S_%f')
                            }), save_path)
                            last_saved = time.time()

                    i_batch += 1
                    sys.stdout.write(f"\r--- {i_batch} / {num_batches} batches; avg. loss: {loss_batch}")
                    sys.stdout.flush()

                # if not is_train:
                #     print(f"Ground truth value:\n{inputs['values'][0]}")
                #     print(f"Predicted value:\n{outputs['v'][0, 0]}")

            sys.stdout.write("\r")
            sys.stdout.flush()

            time_duration = time.time() - start_time

            # if is_train: self.scheduler.step()

        return stats_collector.means(), time_duration
