import os

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import pdb

from povdp.networks.attention import (
    AttentionBlock2d
)
from povdp.networks.elements import (
    Conv2dBlock
)


class ResidualBlock2d(nn.Module):
    def __init__(
            self, 
            inp_channels, 
            out_channels, 
            n_groups=8, 
            dropout=0.
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv2dBlock(
                inp_channels=inp_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1, 
                n_groups=n_groups, 
                dropout=dropout
            ), 
            Conv2dBlock(
                inp_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1, 
                n_groups=n_groups, 
                dropout=dropout
            ), 
        )
        # make sure dimensions compatible
        self.residual_conv = nn.Conv2d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        return self.blocks(x) + self.residual_conv(x)


# class MultiGridObsEncoder(nn.Module):
#     def __init__(self, inp_dim, out_dim):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(inp_dim, 32, kernel_size=3, padding=1),
#             nn.Mish(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.Mish(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.Mish(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
        
#         self.time_distributed = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Mish(),
#             nn.Linear(128, 256),
#         )
        
#         self.fc = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.Mish(inplace=True),
#             nn.Linear(256, out_dim),
#         )
        
#     def forward(self, x):
#         B, N, C, H, W = x.shape
#         x = x.reshape(B * N, C, H, W)  # Combine batch and time dimensions
        
#         x = self.features(x)
#         x = self.time_distributed(x)
#         x = x.reshape(B, N, -1)  # Separate batch and time dimensions
#         x = x.mean(dim=1)
#         x = self.fc(x)
    
#         return x


class MultiGridObsvEncoder(nn.Module):
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            n_groups=8, 
            dropout=0., 
            num_heads=1, 
            num_head_dim=-1, 
            multi_obsv_hidden_dims=[64, 128, 256]
    ):
        super().__init__()
        dims = [
            inp_dim, *multi_obsv_hidden_dims
        ]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    inp_dim, 
                    multi_obsv_hidden_dims[0], 
                    kernel_size=3, 
                    padding=1
                )
            )
        ])
        
        for ind, (dim_in, dim_out) in enumerate(in_out[1:]):
            layers = [
                ResidualBlock2d(
                    dim_in, 
                    dim_in, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                ), 
                AttentionBlock2d(
                    dim_in, 
                    groups=n_groups, 
                    num_heads=num_heads, 
                    num_head_dim=num_head_dim, 
                    attention_type='legacy'
                ), 
                ResidualBlock2d(
                    dim_in, 
                    dim_out, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                ), 
                AttentionBlock2d(
                    dim_out, 
                    groups=n_groups, 
                    num_heads=num_heads, 
                    num_head_dim=num_head_dim, 
                    attention_type='legacy'
                ), 
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            self.feature_extractor.append(nn.Sequential(*layers))

        self.feature_extractor.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Flatten(), 
                nn.Mish(), 
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(
                multi_obsv_hidden_dims[-1], 
                multi_obsv_hidden_dims[0]
            ), 
            nn.Mish(), 
            nn.Linear(
                multi_obsv_hidden_dims[0], 
                out_dim
            ), 
        )
        

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.reshape(B * N, C, H, W)  # Combine batch and time dimensions
        
        for module in self.feature_extractor:
            x = module(x)
        x = x.reshape(B, N, -1)  # Separate batch and time dimensions
        x = x.mean(dim=1)
        x = self.fc(x)
    
        return x