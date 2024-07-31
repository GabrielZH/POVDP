import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


def index_to_selector(index, input_size):
    B, V, H, W = input_size
    b = index // (V * H * W)
    remainder = index % (V * H * W)
    
    v = remainder // (H * W)
    remainder = remainder % (H * W)
    
    h = remainder // W
    w = remainder % W
    
    return torch.stack([b, v, h, w], dim=-1)


def bin_distribute(compacted_data, index, original_shape):
    output = torch.zeros(original_shape, device=compacted_data.device, dtype=compacted_data.dtype)
    flat_output = output.view(-1, compacted_data.size(-1))
    flat_output.index_add_(
        0, index, compacted_data.view(-1, compacted_data.size(-1)))
    return output, index_to_selector(index)


class PointCloudDecoder(nn.Module):
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
            **kwargs
    ):
        super().__init__()
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

        if use_resnet:
            raise NotImplementedError
        elif use_embeddings:
            self.rgb_deconv_net = nn.Sequential(
                nn.Mish(), 
                nn.Dropout(dropout), 
                nn.ConvTranspose2d(
                    in_channels=h, 
                    out_channels=h, 
                    dilation=(2, 2), 
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False
                ), 
                *(
                    (nn.GroupNorm(8, h), ) 
                    if use_group_norm else ()
                ), 
                nn.Mish(), 
                nn.Dropout(dropout), 
                nn.ConvTranspose2d(
                    in_channels=h, 
                    out_channels=self.i, 
                    dilation=(2, 2),
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(2, 2), 
                    bias=False 
                ), 
                *(
                    (nn.GroupNorm(8, self.i), ) 
                    if use_group_norm else ()
                )
            )
        else:
            raise NotImplementedError
        
        self.scale = 4
        self.use_batch_norm = use_group_norm
        self.use_embeddings = use_embeddings

        self.noise_ratio = noise_ratio

        self.pre_inverse_dot_layer = nn.ConvTranspose2d(
            in_channels=dot_channels * h, 
            out_channels=self.i,
            kernel_size=(5, 5), 
            stride=(1, 1), 
            bias=True
        )
        self.post_inverse_dot_layer = nn.ConvTranspose2d(
            in_channels=h, 
            out_channels=dot_channels,
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True
        )
        self.device = device

    def __repr__(self):
        return f"{self.__class__.__name__}_h_{self.h}{'_gn' if self.use_group_norm else ''}"
    
    def forward(
            self, 
            surf_features, 
            post_dot_channel=None, 
            pre_dot_target_emb=None, 
            rgb_embedding=None
        ):
        if pre_dot_target_emb is not None:
            assert post_dot_channel is not None and rgb_embedding is not None
            surf_features = surf_features - post_dot_channel
            dot_channel = self.post_inverse_dot_layer(post_dot_channel)
            emb_from_pre_dot_layer = self.pre_inverse_dot_layer(rgb_embedding)
            target_emb = self.pre_inverse_dot_layer(pre_dot_target_emb)

        emb_from_rgb_conv_net = self.rgb_deconv_net(surf_features)
        return {
            'emb_from_rgb_conv_net': emb_from_rgb_conv_net, 
            'emb_from_pre_dot_layer': emb_from_pre_dot_layer, 
            'dot_channel': dot_channel if pre_dot_target_emb is not None else None,
            'target_emb': target_emb if pre_dot_target_emb is not None else None
        }
    
    # def grid_to_coord(self, selector, pcn_sample_ratio=1.0):
    #     """
    #     :param xyz: (M, 3)
    #     :param index: (M,)
    #     :return:
    #     """
    #     if pcn_sample_ratio < 1.0:
    #         raise NotImplementedError
        
    #     index, vs, hs, ws = selector.split(1, dim=-1)

    #     map_h, map_w = self.map_res
    #     h1, w1, h2, w2 = self.map_bbox
    #     v1, v2 = self.v_bbox

    #     # convert grid indices back to xyz coordinates
    #     h_to_xyz = (hs.float() / map_h * (h2 - h1)) + h1
    #     w_to_xyz = (ws.float() / map_w * (w2 - w1)) + w1
    #     v_to_xyz = (vs.float() / self.v_res * (v2 - v1)) + v1
    #     samples = (hs >= 0) & (hs < map_h) \
    #         & (ws >= 0) & (ws < map_w) \
    #             & (vs >= 0) & (vs < self.v_res)

    #     hwv_to_xyz = [
    #         value for _, value in sorted(
    #             zip([self.xyz_to_h, self.xyz_to_w, self.xyz_to_v], 
    #                 [h_to_xyz, w_to_xyz, v_to_xyz])
    #         )
    #     ]
    #     xyz = torch.cat(hwv_to_xyz, dim=-1)

    #     return xyz, index.squeeze(-1), samples
