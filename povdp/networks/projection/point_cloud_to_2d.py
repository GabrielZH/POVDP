# from pytorch_memlab import profile, set_target_gpu
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch_scatter import scatter

from calvin.core.models.detector.resnet.resnet_feature_extractor import PretrainedResNetShort
from povdp.networks.attention import (
    AttentionBlock2d
)


def selector_to_index(selector, output_sizes):
    B, V, H, W = output_sizes
    return selector[:, 0] * V * H * W + selector[:, 1] * H * W + selector[:, 2] * W + selector[:, 3]


def get_counts(index, output_sizes):
    feature_counts = torch.bincount(index, minlength=np.prod(output_sizes))
    return feature_counts.view(*output_sizes).detach()


def bin_reduce(data, selector, output_sizes):
    index = selector_to_index(selector, output_sizes)
    output = scatter(data, index, dim=0, dim_size=np.prod(output_sizes), reduce="sum")
    return output.view(*output_sizes, data.size(-1)), get_counts(index, output_sizes)


class PointCloudTo2D(nn.Module):
    def __init__(
            self, 
            map_bbox=None, 
            map_res=None, 
            pcn_h=None, 
            pcn_i=None, 
            pcn_f=None, 
            v_bbox=None, 
            v_res=None,
            dropout=None,
            xyz_to_h=None, 
            xyz_to_w=None, 
            pcn_sample_ratio=None, 
            noise_ratio=0.0, 
            use_group_norm=False,
            use_resnet=False, 
            use_embeddings=None,
            device=None, 
            dot_channels=None, 
            use_attention=False, 
            **kwargs
    ):
        super(PointCloudTo2D, self).__init__()
        self.map_bbox = map_bbox  # (h1, w1, h2, w2)
        self.map_res = map_res  # (map_x, map_z)
        self.i = 128 if use_embeddings else pcn_i
        self.h = h = pcn_h  # size of hidden layers
        self.f = f = pcn_f  # size of hidden layers
        self.xyz_to_h = xyz_to_h  # xyz dim corresponding to h dim
        self.xyz_to_w = xyz_to_w  # xyz dim corresponding to w dim
        axes = {0, 1, 2}
        axes.remove(xyz_to_h)
        axes.remove(xyz_to_w)
        self.xyz_to_v = axes.pop()
        self.pcn_sample_ratio = pcn_sample_ratio  # number of output features

        self.v_bbox = v_bbox
        self.v_res = v_res

        if use_attention: use_group_norm = False

        if use_resnet:
            print("Using ResNet18...")
            self.resnet = PretrainedResNetShort(freeze=True, device=device, cutoff_layers=5)
            self.resnet.eval()

            print(self.resnet.model)

            layers = list()
            if use_group_norm:
                layers.append(nn.GroupNorm(8, self.i))
            if use_attention:
                layers.append(AttentionBlock2d(
                    dim=h, 
                    groups=8, 
                    num_heads=8, 
                    attention_type='legacy'
                ))
            layers.extend([
                nn.Conv2d(
                    in_channels=64, 
                    out_channels=h,
                    kernel_size=(3, 3), 
                    dilation=(2, 2), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False
                ),
                nn.Dropout(dropout),
                nn.Mish(),
            ])
            if use_group_norm:
                layers.append(nn.GroupNorm(8, h))
            if use_attention:
                layers.append(AttentionBlock2d(
                    dim=h, 
                    groups=8, 
                    num_heads=8, 
                    attention_type='legacy'
                ))
            layers.extend([
                nn.Conv2d(
                    in_channels=h, 
                    out_channels=h, 
                    dilation=(2, 2),
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False
                ),
                nn.Dropout(dropout),
                nn.Mish()
            ])
            self.rgb_conv_net = nn.Sequential(*layers)

        elif use_embeddings:
            self.resnet = None

            layers = list()
            if use_group_norm:
                layers.append(nn.GroupNorm(8, self.i))
            if use_attention:
                layers.append(AttentionBlock2d(
                    dim=h, 
                    groups=8, 
                    num_heads=8, 
                    attention_type='legacy'
                ))
            layers.extend([
                nn.Conv2d(
                    in_channels=64, 
                    out_channels=h,
                    kernel_size=(3, 3), 
                    dilation=(2, 2), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False
                ),
                nn.Dropout(dropout),
                nn.Mish(),
            ])
            if use_group_norm:
                layers.append(nn.GroupNorm(8, h))
            if use_attention:
                layers.append(AttentionBlock2d(
                    dim=h, 
                    groups=8, 
                    num_heads=8, 
                    attention_type='legacy'
                ))
            layers.extend([
                nn.Conv2d(
                    in_channels=h, 
                    out_channels=h, 
                    dilation=(2, 2),
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False
                ),
                nn.Dropout(dropout),
                nn.Mish()
            ])
            self.rgb_conv_net = nn.Sequential(*layers)
        else:
            self.resnet = None

            self.rgb_conv_net = nn.Sequential(
                *((nn.GroupNorm(8, self.i),) if use_group_norm else ()),
                AttentionBlock2d(
                    dim=self.i, 
                    groups=8, 
                    num_heads=8, 
                    attention_type='legacy'
                ) if use_attention else (), 
                nn.Conv2d(
                    in_channels=self.i, 
                    out_channels=h, 
                    dilation=(2, 2),
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False
                ),
                nn.Dropout(dropout),
                nn.Mish(),
                nn.MaxPool2d(
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),

                *((nn.GroupNorm(8, h),) if use_group_norm else ()),
                AttentionBlock2d(
                    dim=h, 
                    groups=8, 
                    num_heads=8, 
                    attention_type='legacy'
                ) if use_attention else (), 
                nn.Conv2d(
                    in_channels=h, 
                    out_channels=h, 
                    dilation=(2, 2),
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False
                ),
                nn.Dropout(dropout),
                nn.Mish(),
                nn.MaxPool2d(
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                )
            )

        self.scale = 4
        self.use_group_norm = use_group_norm
        self.use_embeddings = use_embeddings

        self.noise_ratio = noise_ratio

        self.pre_dot_layer = nn.Conv2d(
            in_channels=self.i, 
            out_channels=dot_channels * h,
            kernel_size=(5, 5), 
            stride=(1, 1), 
            bias=True
        )
        self.post_dot_layer = nn.Conv2d(
            in_channels=dot_channels, 
            out_channels=h,
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True
        )
        self.device = device

    def __repr__(self):
        return f"{self.__class__.__name__}_h_{self.h}{'_gn' if self.use_group_norm else ''}"

    # @profile
    def forward(
            self, 
            *, 
            rgb=None, 
            emb=None, 
            surf_xyz=None, 
            index=None, 
            valid_points=None, 
            free_xyz=None,
            target_emb=None, 
            batch_size=None, 
            self_supervise=False, 
            **kwargs
    ):
        """
        N = sum_{batch}{traj_len * orientation}, H = image height, W = image width, 3 = (X, Y, Z)
        :param rgb:            (N, 3, H, W) RGB for each surface point
        :param surf_xyz:        (N, 3, H, W) XYZ for each surface point
        :param free_xyz:        (N, H, W, k, 3) XYZ for each surface point
        :param valid_points:    (N, H, W) boolean for each point
        :param index:           (N, H, W) batch element index for each point
        :return: feature_map:   (B, F, X, Y)
        """
        # print(f"emb shape in cond: {emb.shape}")
        # print(f"surf_xyz shape in cond: {surf_xyz.shape}")
        # print(f"index shape in cond: {index.shape}")
        # print(f"valid_points shape in cond: {valid_points.shape}")
        # print(f"target emb shape in cond: {target_emb.shape}")
        # print(f"batch size: {batch_size}")
        scale = self.scale
        # N, H, W, K, _ = free_xyz.size()
        N, _, H, W = surf_xyz.size()
        if self.use_embeddings:
            surf_rgb = rgb
            surf_features = self.rgb_conv_net(emb)
            _index = index.view(-1, 1, 1).expand(-1, H, W)
            assert False
        else:
            surf_rgb = F.avg_pool2d(rgb, scale)
            surf_xyz = surf_xyz[:, :, scale // 2::scale, scale // 2::scale]
            # free_xyz = free_xyz[:, scale//2::scale, scale//2::scale]
            _index = index.view(-1, 1, 1).expand(-1, ceil(H / scale), ceil(W / scale))

            if self.resnet:
                with torch.no_grad():
                    rgb = self.resnet(rgb)
            surf_features = self.rgb_conv_net(rgb)

        assert surf_xyz.size() == surf_rgb.size()

        if target_emb is not None:
            pre_dot_target_emb = self.pre_dot_layer(target_emb)
            reduced_mean_target_emb = reduce(pre_dot_target_emb, "b f h w -> b f () ()", "mean")
            selected_target_emb = torch.index_select(reduced_mean_target_emb, 0, index)
            rgb_embedding = self.pre_dot_layer(F.pad(emb, pad=(2, 2, 2, 2)))
            dot_channel = reduce(selected_target_emb * rgb_embedding, "n (f c) h w -> n c h w", "sum", f=self.h)
            post_dot_channel = self.post_dot_layer(dot_channel)
            surf_features = surf_features + post_dot_channel

        if self_supervise:
            return { 
                'surf_features': surf_features, 
                'dot_channel': dot_channel if target_emb is not None else None, 
                'post_dot_channel': post_dot_channel, 
                'pre_dot_target_emb': pre_dot_target_emb, 
                'rgb_embedding': rgb_embedding,
            }

        if valid_points is None:
            flat_features = rearrange(surf_features, "n f h w -> (n h w) f")
            flat_xyz = rearrange(surf_xyz, "n f h w -> (n h w) f")
            flat_rgb = rearrange(surf_rgb, "n f h w -> (n h w) f")
            # flat_free = rearrange(free_xyz, "n h w k f -> (n h w) k f")
            flat_index = rearrange(_index, "n h w -> (n h w)")
        else:
            if not self.use_embeddings:
                valid_points = valid_points[:, ::scale, ::scale]
            _surf_features = rearrange(surf_features, "n f h w -> n h w f")
            # print(f"rearranged surf_features shape: {_surf_features.shape}")
            _surf_xyz = rearrange(surf_xyz, "n f h w -> n h w f")
            # print(f"rearranged surf_xyz shape: {_surf_xyz.shape}")
            _surf_rgb = rearrange(surf_rgb, "n f h w -> n h w f")
            flat_features = _surf_features[valid_points]  # (M, F)
            # print(f"flat_features (rearranged surf_features indexed by valid_points) shape: {flat_features.shape}")
            flat_xyz = _surf_xyz[valid_points]  # (M, 3)
            # print(f"flat_xyz (rearranged surf_xyz indexed by valid_points) shape: {flat_xyz.shape}")
            flat_rgb = _surf_rgb[valid_points]  # (M, 3)
            # flat_free = free_xyz[_valid_points]     # (M, k, 3)
            flat_index = _index[valid_points]  # (M,)
            # print(f"flat index (expanded index indexed by valid points) shape: {flat_index.shape}")

        selector, samples = self.coord_to_grid(flat_xyz, flat_index, self.pcn_sample_ratio)
        # print(f"selector from coord_to_grid shape: {selector.shape}")
        # print(f"samples from coord_to_grid shape: {samples.shape}")
        flat_features, flat_rgb, selector = flat_features[samples], flat_rgb[samples], selector[samples]
        # print(f"sampled flat features: {flat_features.shape}")
        # print(f"sampled selector: {selector.shape}")

        output_sizes = (batch_size, self.v_res, *self.map_res)

        feature_map, feature_counts = bin_reduce(flat_features, selector, output_sizes)
        rgb_map, _ = bin_reduce(flat_rgb, selector, output_sizes)

        # _flat_free = rearrange(flat_free, "m k f -> (m k) f")
        # _flat_index = flat_index.unsqueeze(1).repeat(1, K).view(-1)

        # selector, samples = self.coord_to_grid(_flat_free, _flat_index)
        # selector = selector[samples]
        # index = selector_to_index(selector, output_sizes)
        # free_map = get_counts(index, output_sizes)

        return {
            'feature_map': rearrange(feature_map, "b v x y f -> b f v x y"),
            'feature_counts': feature_counts.unsqueeze(1),
            'rgb_map': rearrange(rgb_map, "b v x y f -> b f v x y"),
            # 'free_map': free_map.unsqueeze(1)
        }

    def coord_to_grid(self, xyz, index, pcn_sample_ratio=1.0):
        """
        :param xyz: (M, 3)
        :param index: (M,)
        :return:
        """
        # convert coordinates to map grid indices
        map_h, map_w = self.map_res
        h1, w1, h2, w2 = self.map_bbox
        v1, v2 = self.v_bbox
        hs = ((xyz[:, self.xyz_to_h] - h1) * map_h / (h2 - h1)).long()  # (N',)
        ws = ((xyz[:, self.xyz_to_w] - w1) * map_w / (w2 - w1)).long()  # (N',)
        vs = ((xyz[:, self.xyz_to_v] - v1) * self.v_res / (v2 - v1)).long()  # (N',)
        selector = torch.stack([index, vs, hs, ws], dim=-1)
        samples = (hs >= 0) & (hs < map_h) & (ws >= 0) & (ws < map_w) & (vs >= 0) & (vs < self.v_res)

        if pcn_sample_ratio < 1.0:
            samples = samples & (torch.rand(len(selector), device=self.device) < self.pcn_sample_ratio)

        return selector, samples
