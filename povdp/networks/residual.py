from typing import Union, List, Tuple
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import pdb

from povdp.networks.elements import (
    Conv1dBlock, 
    Conv2dBlock, 
    ScaleShiftConv1dBlock, 
    ScaleShiftConv2dBlock, 
)
from povdp.networks.vision import (
    MultiGridObsvEncoder,
    MultiImageObsEncoder,
)
from povdp.networks.helpers import (
    FiLM_scale1d, 
    FiLM_scale2d, 
)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(
            self, 
            x, 
            time_embed=None, 
            non_temporal_embed=None
    ):
        """
        Apply the module to `x` given `embed` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
            self, 
            x, 
            time_embed=None, 
            non_temporal_embed=None
    ):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(
                    x, 
                    time_embed, 
                    non_temporal_embed
                )
            else:
                x = layer(x)
            
        return x


class FiLMResidualBlock1d(TimestepBlock):
    def __init__(
            self, 
            inp_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            dropout=None,
            cond_predict_scale=False
        ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(
                inp_channels, 
                out_channels, 
                kernel_size, 
                padding=kernel_size // 2, 
                n_groups=n_groups, 
                dropout=dropout
            ),
            Conv1dBlock(
                out_channels, 
                out_channels, 
                kernel_size, 
                padding=kernel_size // 2, 
                n_groups=n_groups, 
                dropout=dropout
            ),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    

class FiLMResidualBlock2d(TimestepBlock):
    def __init__(
            self, 
            inp_channels, 
            out_channels, 
            cond_dim, 
            kernel_size=3,
            n_groups=8, 
            dropout=None, 
            cond_predict_scale=False
        ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(
                inp_channels, 
                out_channels, 
                kernel_size, 
                padding=kernel_size // 2, 
                n_groups=n_groups, 
                dropout=dropout),
            Conv2dBlock(
                out_channels, 
                out_channels, 
                kernel_size, 
                padding=kernel_size // 2, 
                n_groups=n_groups, 
                dropout=dropout),
        ])

        # FiLM modulation for 2D input/output
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = MultiGridObsvEncoder(cond_dim, cond_channels)

        # make sure dimensions compatible
        self.residual_conv = nn.Conv2d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x in_height x in_width ]
            cond : [ batch_size x cond_dim x cond_height x cond_width ]
            multi_image_obs_encoder: encodes consecutive [ n_obs_steps ] 
            embodied image observations to low-dim embeddings.
            returns:
            out : [ batch_size x out_channels x out_height x out_width ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    

class TimeConditionalResidualBlock1d(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param inp_dim: the number of input channels.
    :param out_dim: the number of output channels.
    :param time_step_embed_dim: the number of timestep embedding channels.
    :param n_groups: the number of groups for group normalization
    :param dropout: the rate of dropout.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """

    def __init__(
        self,
        inp_channels, 
        out_channels, 
        time_step_embed_dim, 
        n_groups=8, 
        dropout=0.,
        use_conv=False, 
        use_scale_shift_norm=False, 
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.inp_layers = nn.Sequential(
            nn.GroupNorm(n_groups, inp_channels),
            nn.SiLU(),
            nn.Conv1d(inp_channels, out_channels, 3, padding=1),
        )

        self.time_embed_layer = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(
                time_step_embed_dim, 
                2 * out_channels 
                if use_scale_shift_norm else out_channels), 
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(), 
            nn.Dropout(dropout), 
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )

        if out_channels == inp_channels:
            self.residual_conv = nn.Identity()
        elif use_conv:  # conv_kernel_size == 3
            self.residual_conv = nn.Conv1d(
                inp_channels, out_channels, 3, padding=1
            )
        else:  # conv_kernel_size == 1
            self.residual_conv= nn.Conv1d(
                inp_channels, out_channels, 1
            )

    def forward(self, x, embed):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, embed)

    def _forward(self, x, embed):
        out = self.inp_layers(x)
        embed_out = self.time_embed_layer(embed).type(out.dtype)
        while len(embed_out.shape) < len(out.shape):
            embed_out = embed_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(embed_out, 2, dim=1)
            out = out_norm(out) * (1 + scale) + shift
            out = out_rest(out)
        else:
            out = out + embed_out
            out = self.out_layers(out)
        return self.residual_conv(x) + out


class TimeConditionalResidualBlock2d(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param inp_dim: the number of input channels.
    :param out_dim: the number of output channels.
    :param time_step_embed_dim: the number of timestep embedding channels.
    :param n_groups: the number of groups for group normalization
    :param dropout: the rate of dropout.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    """

    def __init__(
        self,
        inp_channels, 
        out_channels, 
        time_step_embed_dim, 
        n_groups=8, 
        dropout=0.,
        use_conv=False, 
        use_scale_shift_norm=False, 
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.inp_layers = nn.Sequential(
            nn.GroupNorm(n_groups, inp_channels),
            nn.SiLU(),
            nn.Conv2d(inp_channels, out_channels, 3, padding=1),
        )

        self.time_embed_layer = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(
                time_step_embed_dim, 
                2 * out_channels 
                if use_scale_shift_norm else out_channels), 
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            nn.SiLU(), 
            nn.Dropout(dropout), 
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if out_channels == inp_channels:
            self.residual_conv = nn.Identity()
        elif use_conv:  # conv_kernel_size == 3
            self.residual_conv = nn.Conv2d(
                inp_channels, out_channels, 3, padding=1
            )
        else:  # conv_kernel_size == 1
            self.residual_conv= nn.Conv2d(
                inp_channels, out_channels, 1
            )

    def forward(self, x, embed):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, embed)

    def _forward(self, x, embed):
        out = self.inp_layers(x)
        embed_out = self.time_embed_layer(embed).type(out.dtype)
        while len(embed_out.shape) < len(out.shape):
            embed_out = embed_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(embed_out, 2, dim=1)
            out = out_norm(out) * (1 + scale) + shift
            out = out_rest(out)
        else:
            out = out + embed_out
            out = self.out_layers(out)
        return self.residual_conv(x) + out
    

class HybridConditionalResBlock1d(TimestepBlock):
    def __init__(
            self,
            inp_channels, 
            out_channels, 
            time_embed_dim, 
            n_groups=8, 
            dropout=0., 
            cond_mode='coupled+cat', 
            time_embed_encoding_mode='embed', 
            non_temp_cond_encoding_mode='embed', 
            cond_predict_scale=False, 
            use_scale_shift_norm=False, 
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(
                inp_channels, 
                out_channels,
                kernel_size=3, 
                padding=1, 
                n_groups=n_groups, 
                dropout=dropout
            ),
            ScaleShiftConv1dBlock(
                out_channels, 
                out_channels, 
                n_groups=n_groups, 
                dropout=dropout
            ) if use_scale_shift_norm 
            else Conv1dBlock(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                n_groups=n_groups, 
                dropout=dropout
            )
        ])

        # FiLM modulation for 2D input/output
        cond_out_channels = out_channels
        if cond_predict_scale or use_scale_shift_norm:
            cond_out_channels = out_channels * 2

        self.out_channels = out_channels
        self.cond_mode = cond_mode
        self.cond_predict_scale = cond_predict_scale
        self.use_scale_shift_norm = use_scale_shift_norm

        if cond_mode == 'coupled+cat':
            cond_dim = time_embed_dim * 2
            if cond_predict_scale and use_scale_shift_norm:
                self.FiLM_scale_cond_encoder = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(cond_dim, cond_out_channels),
                    Rearrange('batch t -> batch t 1'),
                )
                self.scale_shift_cond_encoder = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(cond_dim, cond_out_channels),
                    Rearrange('batch t -> batch t 1'),
                )
            else:
                self.cond_encoder = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(cond_dim, cond_out_channels),
                    Rearrange('batch t -> batch t 1'),
                )
        elif cond_mode == 'coupled+sum':
            cond_dim = time_embed_dim
            if cond_predict_scale and use_scale_shift_norm:
                self.FiLM_scale_cond_encoder = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(cond_dim, cond_out_channels),
                    Rearrange('batch t -> batch t 1'),
                )
                self.scale_shift_cond_encoder = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(cond_dim, cond_out_channels),
                    Rearrange('batch t -> batch t 1'),
                )
            else:
                self.cond_encoder = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(cond_dim, cond_out_channels),
                    Rearrange('batch t -> batch t 1'),
                )
        elif cond_mode != 'decoupled':
            raise ValueError(f"Unknown cond_mode: {cond_mode}")

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()
        
    def forward(
            self, 
            x, 
            time_embed, 
            non_temporal_embed=None
    ):
        if 'coupled+' in self.cond_mode:
            return self._forward_coupled_cond(
                x=x, 
                time_embed=time_embed, 
                non_temporal_embed=non_temporal_embed
            )
        else:
            return self._forward_decoupled_cond(
                x=x, 
                time_embed=time_embed, 
                non_temporal_embed=non_temporal_embed
            )

    def _forward_coupled_cond(
            self, 
            x, 
            time_embed, 
            non_temporal_embed
    ):
        '''
            x : [ batch_size x in_channels x in_height x in_width ]
            cond : [ batch_size x cond_dim x cond_height x cond_width ]
            multi_image_obs_encoder: encodes consecutive [ n_obs_steps ] 
            embodied image observations to low-dim embeddings.
            returns:
            out : [ batch_size x out_channels x out_height x out_width ]
        '''
        out = self.blocks[0](x)
        if self.cond_mode == 'coupled+cat':
            cond = torch.cat([
                time_embed, 
                non_temporal_embed
            ], dim=1)
        else:
            cond = time_embed + non_temporal_embed

        if self.cond_predict_scale:
            if self.use_scale_shift_norm:
                FiLM_scaled_embed = self.FiLM_scale_cond_encoder(cond)
                FiLM_scaled_out = FiLM_scale1d(out, FiLM_scaled_embed)
                scale_shfited_embed = self.scale_shift_cond_encoder(cond)
                scale_shifted_out = self.blocks[1](out, scale_shfited_embed)
                out = FiLM_scaled_out + scale_shifted_out
            else:
                embed = self.cond_encoder(cond)
                out = FiLM_scale1d(out, embed)
                out = self.blocks[1](out)
        elif self.use_scale_shift_norm:
            embed = self.cond_encoder(cond)
            out = self.blocks[1](out, embed)
        else:
            embed = self.cond_encoder(cond)
            out = self.blocks[1](out + embed)
        out = out + self.residual_conv(x)
        return out
    
    def _forward_decoupled_cond(
            self, 
            x, 
            time_embed, 
            non_temporal_embed
    ):
        out = self.blocks[0](x)
        non_temporal_embed = self.non_temporal_cond_encoder(
            non_temporal_embed
        )
        if self.use_scale_shift_norm:
            time_cond_out = self.blocks[1](out, time_embed)
        else:
            time_cond_out = out + time_embed

        if self.cond_predict_scale:
            non_temporal_cond_out = FiLM_scale1d(
                out, 
                non_temporal_embed
            )
        else:
            non_temporal_cond_out = out + non_temporal_embed

        return time_cond_out + non_temporal_cond_out + self.residual_conv(x)


class HybridConditionalResBlock2d(TimestepBlock):
    def __init__(
            self,
            inp_channels, 
            out_channels, 
            time_embed_dim, 
            non_temporal_cond_dim, 
            n_groups=8, 
            dropout=0., 
            cond_mode='coupled+cat', 
            time_embed_encoding_mode='embed', 
            non_temp_cond_encoding_mode='embed', 
            cond_predict_scale=False, 
            use_scale_shift_norm=False, 
    ):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param inp_channels: 
        :param out_channels: 
        :param cond_dim: 
        :param n_groups: 
        :param dropout: 
        :param cond_mode: ['coupled+cat','coupled+sum','decoupled']
        :param cond_predict_scale:
        :param use_scale_shift_norm:
        :return: 
        """
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(
                inp_channels, 
                out_channels,
                kernel_size=3, 
                padding=1, 
                n_groups=n_groups, 
                dropout=dropout
            ),
            ScaleShiftConv2dBlock(
                out_channels, 
                out_channels, 
                n_groups=n_groups, 
                dropout=dropout
            ) if use_scale_shift_norm 
            else Conv2dBlock(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                n_groups=n_groups, 
                dropout=dropout
            )
        ])

        # FiLM modulation for 2D input/output
        cond_out_channels = out_channels
        if cond_predict_scale or use_scale_shift_norm:
            cond_out_channels = out_channels * 2

        self.out_channels = out_channels
        self.cond_mode = cond_mode
        self.cond_predict_scale = cond_predict_scale
        self.use_scale_shift_norm = use_scale_shift_norm

        if cond_mode == 'coupled+cat':
            cond_dim = time_embed_dim + non_temporal_cond_dim
            if cond_predict_scale and use_scale_shift_norm:
                self.FiLM_scale_cond_encoder = MultiGridObsvEncoder(
                    inp_dim=cond_dim, 
                    out_dim=cond_out_channels
                )
                self.scale_shift_cond_encoder = MultiGridObsvEncoder(
                    inp_dim=cond_dim, 
                    out_dim=cond_out_channels
                )
            else:
                self.cond_encoder = MultiGridObsvEncoder(
                    inp_dim=cond_dim, 
                    out_dim=cond_out_channels
                )
        elif cond_mode == 'coupled+sum':
            cond_dim = time_embed_dim
            self.non_temporal_cond_encoder = nn.Conv2d(
                in_channels=non_temporal_cond_dim, 
                out_channels=time_embed_dim, 
                kernel_size=1,  
            ) if non_temp_cond_encoding_mode == 'conv' else nn.Embedding(
                num_embeddings=non_temporal_cond_dim, 
                embedding_dim=time_embed_dim
            )
            if cond_predict_scale and use_scale_shift_norm:
                self.FiLM_scale_cond_encoder = MultiGridObsvEncoder(
                    inp_dim=cond_dim, 
                    out_dim=cond_out_channels
                )
                self.scale_shift_cond_encoder = MultiGridObsvEncoder(
                    inp_dim=cond_dim, 
                    out_dim=cond_out_channels
                )
            else:
                self.cond_encoder = MultiGridObsvEncoder(
                    inp_dim=cond_dim, 
                    out_dim=cond_out_channels
                )
        elif cond_mode == 'decoupled':
            self.time_embed_encoder = nn.Conv2d(
                in_channels=time_embed_dim, 
                out_channels=cond_out_channels, 
                kernel_size=1
            ) if time_embed_encoding_mode == 'conv' else nn.Embedding(
                num_embeddings=time_embed_dim, 
                embedding_dim=cond_out_channels
            )
            self.non_temporal_cond_encoder = MultiGridObsvEncoder(
                inp_dim=non_temporal_cond_dim, 
                out_dim=cond_out_channels
            )
        else:
            raise ValueError(f"Unknown cond_mode: {cond_mode}")

        # make sure dimensions compatible
        self.residual_conv = nn.Conv2d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()
        
    def forward(
            self, 
            x, 
            time_embed, 
            non_temporal_embed=None
    ):
        if 'coupled+' in self.cond_mode:
            return self._forward_coupled_cond(
                x=x, 
                time_embed=time_embed, 
                non_temporal_embed=non_temporal_embed
            )
        else:
            return self._forward_decoupled_cond(
                x=x, 
                time_embed=time_embed, 
                non_temporal_embed=non_temporal_embed
            )

    def _forward_coupled_cond(
            self, 
            x, 
            time_embed, 
            non_temporal_embed
    ):
        '''
            x : [ batch_size x in_channels x in_height x in_width ]
            time_embed: [ batch_size x timestep_dim ]
            cond_embed : [ batch_size x cond_dim]
            returns:
            out : [ batch_size x out_channels x out_height x out_width ]
        '''
        out = self.blocks[0](x)
        if self.cond_mode == 'coupled+cat':
            cond = torch.cat([
                time_embed, 
                non_temporal_embed
            ], dim=2)
        else:
            batch_size = non_temporal_embed.shape[0]
            obsv_steps = non_temporal_embed.shape[1]
            non_temporal_embed = rearrange(
                non_temporal_embed, 
                'b n c h w -> (b n) c h w'
            )
            non_temporal_embed = self.non_temporal_cond_encoder(
                non_temporal_embed
            )
            non_temporal_embed = rearrange(
                non_temporal_embed, 
                '(b n) c h w -> b n c h w', 
                b=batch_size, 
                n=obsv_steps
            )
            cond = time_embed + non_temporal_embed

        if self.cond_predict_scale:
            if self.use_scale_shift_norm:
                FiLM_scaled_embed = self.FiLM_scale_cond_encoder(cond)
                FiLM_scaled_out = FiLM_scale2d(out, FiLM_scaled_embed)
                scale_shfited_embed = self.scale_shift_cond_encoder(cond)
                scale_shifted_out = self.blocks[1](out, scale_shfited_embed)
                out = FiLM_scaled_out + scale_shifted_out
            else:
                embed = self.cond_encoder(cond)
                out = FiLM_scale2d(out, embed)
                out = self.blocks[1](out)
        elif self.use_scale_shift_norm:
            embed = self.cond_encoder(cond)
            out = self.blocks[1](out, embed)
        else:
            embed = self.cond_encoder(cond)
            out = self.blocks[1](out + embed)
        out = out + self.residual_conv(x)
        return out
    
    def _forward_decoupled_cond(
            self, 
            x, 
            time_embed, 
            non_temporal_embed
    ):
        out = self.blocks[0](x)
        non_temporal_embed = self.cond_encoder(
            non_temporal_embed
        )
        if self.use_scale_shift_norm:
            time_cond_out = self.blocks[1](out, time_embed)
        else:
            time_cond_out = out + time_embed

        if self.cond_predict_scale:
            non_temporal_cond_out = FiLM_scale2d(
                out, 
                non_temporal_embed
            )
        else:
            non_temporal_cond_out = out + non_temporal_embed

        return time_cond_out + non_temporal_cond_out + self.residual_conv(x)

