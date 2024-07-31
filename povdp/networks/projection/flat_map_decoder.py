import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from povdp.networks.projection.point_cloud_decoder import PointCloudDecoder


class FlatMapDecoder(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.point_deconv_net = self.get_point_deconv_model(**config)

        self.reduce_thresh = config['reduce_thresh']
        self.h = h = config['pcn_h']  # size of hidden layers
        self.f = f = config['pcn_f']  # number of output features
        self.v_res = config['v_res']
        dropout = config['dropout']

        self.map_deconv_net = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=f, 
                out_channels=h, 
                dilation=(1, 2, 2), 
                kernel_size=(1, 3, 3), 
                stride=(1, 1, 1), 
                padding=(0, 2, 2), 
                bias=True
            ), 
            nn.Mish(), 
            nn.Dropout(dropout), 
            nn.ConvTranspose3d(
                in_channels=h, 
                out_channels=h, 
                dilation=(1, 2, 2), 
                kernel_size=(self.v_res, 3, 3), 
                stride=(1, 1, 1), 
                padding=(0, 2, 2), 
                bias=True
            )
        )

    def __repr__(self):
        return f"{super().__repr__()}_{repr(self.point_deconv_net)}_{self.f}"
    
    def get_point_deconv_model(self, **config):
        return PointCloudDecoder(**config)

    def forward(self, feature_map):
        return self.map_deconv_net(feature_map)


