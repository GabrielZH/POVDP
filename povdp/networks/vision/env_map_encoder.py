import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import pdb

from povdp.networks.elements import Conv2dBlock
from povdp.networks.attention import AttentionBlock1d, AttentionBlock2d


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


class EnvMapEncoder(nn.Module):
    """
    Encodes maps in grid_maze2d through CNNs into 1D embeddings as 
    conditions of the diffusion model. 
    """
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            hidden_dims=(64, 128, 256), 
            n_groups=8, 
            dropout=0, 
            num_heads=1, 
            num_head_dim=-1, 
            pooling=True,
    ):
        super().__init__()

        ## 2D Convolutional layers
        channels = [
            inp_dim, *hidden_dims, 
        ]
        in_outs = list(zip(channels[:-1], channels[1:]))

        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    inp_dim, 
                    hidden_dims[0], 
                    kernel_size=3, 
                    padding=1
                )
            )
        ])

        for ind, (dim_in, dim_out) in enumerate(in_outs[1:]):
            layers = [
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
            ]
            if pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
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
                hidden_dims[-1], 
                hidden_dims[0]
            ), 
            nn.Mish(), 
            nn.Linear(
                hidden_dims[0], 
                out_dim
            ), 
        )
        self.feature_extractor.append(self.fc)

    def forward(self, x):
        for module in self.feature_extractor:
            x = module(x)
        return x
    

# class ResidualBlock2d(nn.Module):
#     def __init__(self, in_channels, out_channels, n_groups=8, dropout=0):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.norm1 = nn.GroupNorm(n_groups, out_channels)
#         self.dropout = nn.Dropout(dropout)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.norm2 = nn.GroupNorm(n_groups, out_channels)

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.norm1(self.conv1(x)))
#         out = self.dropout(out)
#         out = self.norm2(self.conv2(out))
#         out += identity
#         return F.relu(out)

# class AttentionBlock2d(nn.Module):
#     def __init__(self, in_channels, groups, num_heads, num_head_dim, attention_type='legacy'):
#         super().__init__()
#         # This is a placeholder for the actual attention implementation
#         # Assuming it is a spatial attention mechanism
#         self.attention = nn.MultiheadAttention(in_channels, num_heads)

#     def forward(self, x):
#         # Here we would need to adapt x to the format expected by nn.MultiheadAttention
#         # This is a simplified example
#         batch, channels, height, width = x.shape
#         x = x.view(batch, channels, height * width)  # Flatten spatial dimensions
#         x = x.permute(2, 0, 1)  # Change to (seq_len, batch, features) for attention
#         attn_output, _ = self.attention(x, x, x)
#         return attn_output.permute(1, 2, 0).view(batch, channels, height, width)  # Back to original shape


class EnvMapSequenceEncoder(nn.Module):
    """
    Encodes sequences of maps in grid_maze2d through CNNs into 1D embeddings as 
    conditions for further processing with a Transformer.
    """
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            hidden_dims=(64, 128, 256), 
            n_groups=8, 
            dropout=0, 
            num_heads=8, 
            num_head_dim=-1,
    ):
        super().__init__()

        # Convolutional layers for feature extraction per map
        self.feature_extractor = nn.ModuleList([
            nn.Conv2d(inp_dim, hidden_dims[0], kernel_size=3, padding=1)
        ])
        channels = [hidden_dims[0], *hidden_dims]

        for dim_in, dim_out in zip(channels[:-1], channels[1:]):
            self.feature_extractor.extend([
                ResidualBlock2d(
                    dim_in, 
                    dim_out, 
                    n_groups=n_groups, 
                    dropout=dropout
                ),
                # AttentionBlock2d(
                #     dim_out, 
                #     groups=n_groups, 
                #     num_heads=num_heads, 
                #     num_head_dim=num_head_dim, 
                #     attention_type='legacy'
                # ), 
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        # Flatten and use a linear layer to reduce dimensionality
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(hidden_dims[-1], out_dim)
        )

        # Transformer for processing sequences of features
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_dim, 
                nhead=num_heads, 
                dim_feedforward=out_dim * 4),
            num_layers=6
        )

        self.output_layer = nn.Linear(out_dim, out_dim)
        self.query_vector = nn.Parameter(torch.randn(out_dim))
        self.attn_layer = AttentionBlock1d(
            out_dim, 
            groups=n_groups, 
            num_heads=num_heads, 
            num_head_dim=num_head_dim, 
            attention_type='legacy'
        )

    def forward(self, x, cond_selection='last'):
        batch_size, seq_len, channels, height, width = x.shape
        # Flatten spatial dimensions and project to embedding dimension
        x = x.view(batch_size * seq_len, channels, height, width)

        # Pass each map through the CNN feature extractor
        for layer in self.feature_extractor:
            x = layer(x)
        x = self.flatten(x)  # Flatten and linear layer

        x = x.view(batch_size, seq_len, -1)  # Reshape back to sequence format for transformer

        # Transformer expects the input as [seq_len, batch_size, features]
        x = x.permute(1, 0, 2)
        encoded = self.transformer_encoder(x)  # No mask is used

        # Taking the output for the last element in the sequence as an example
        # You might want to aggregate or use a specific output differently depending on your case
        encoded = encoded.permute(1, 0, 2)  # Switch back to [batch_size, seq_len, features]
        if cond_selection == 'last':
            output = self.output_layer(encoded[:, -1, :])  # Only taking the output from the last timestep
        elif cond_selection == 'simple_attn':
            attn_scores = torch.matmul(encoded, self.query_vector)  # [batch_size, seq_len]
            attn_weights = F.softmax(attn_scores, dim=1)  # Softmax over seq_len
            attn_output = torch.einsum('bsf,bs->bf', encoded, attn_weights)  # Weighted sum
            output = self.output_layer(attn_output)
        elif cond_selection == 'advanced_attn':
            encoded = encoded.permute(0, 2, 1)  # [batch_size, features, seq_len]
            attn_output = self.attn_layer(encoded)  # apply AttentionBlock1d over the sequence
            attn_output = encoded.permute(0, 2, 1)  # [batch_size, seq_len, features]
            output = attn_output.mean(dim=1)  # Average pooling over the sequence
            output = self.output_layer(output)

        return output


class EnvMapSequenceEncoderDynamicLen(nn.Module):
    """
    Encodes sequences of maps in grid_maze2d through CNNs into 1D embeddings as 
    conditions for further processing with a Transformer.
    """
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            hidden_dims=(64, 128, 256), 
            n_groups=8, 
            dropout=0, 
            num_heads=8, 
            num_head_dim=-1,
            window_size: int = None, 
    ):
        super().__init__()
        self.window_size = window_size

        # Convolutional layers for feature extraction per map
        self.feature_extractor = nn.ModuleList([
            nn.Conv2d(inp_dim, hidden_dims[0], kernel_size=3, padding=1)
        ])
        channels = [hidden_dims[0], *hidden_dims]

        for dim_in, dim_out in zip(channels[:-1], channels[1:]):
            self.feature_extractor.extend([
                ResidualBlock2d(dim_in, dim_out, n_groups=n_groups, dropout=dropout),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        # Flatten and use a linear layer to reduce dimensionality
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(hidden_dims[-1], out_dim)
        )

        # Transformer for processing sequences of features
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_dim, 
                nhead=num_heads, 
                dim_feedforward=out_dim * 4),
            num_layers=6
        )

        self.output_layer = nn.Linear(out_dim, out_dim)

        self.aggregation_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten()
        )
        self.final_fc = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x should have shape [batch_size, seq_len, channels, height, width]
        batch_size, seq_len, channels, height, width = x.shape

        # Flatten spatial dimensions and project to embedding dimension
        x = x.view(batch_size * seq_len, channels, height, width)

        # Pass each map through the CNN feature extractor
        for layer in self.feature_extractor:
            x = layer(x)
        x = self.flatten(x)  # Flatten and linear layer

        x = x.view(batch_size, seq_len, -1)  # Reshape back to sequence format for transformer

        outputs = list()
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            end = i + 1
            
            # Slice the sequence to get the window
            window = x[:, start:end, :]

            # Padding
            if window.size(1) < self.window_size:
                padding_size = self.window_size - window.size(1)
                padding = torch.zeros(batch_size, padding_size, x.size(2)).to(x.device)
                window = torch.cat([padding, window], dim=1)

            # Create mask for the Transformer
            # mask = torch.full(
            #     (self.window_size, self.window_size), float('-inf')
            # ).to(x.device)
            mask = torch.full(
                (self.window_size, self.window_size), -1e10
            ).to(x.device)
            
            mask_start_idx = max(0, self.window_size - i - 1)
            mask[mask_start_idx:, mask_start_idx:] = 0

            window = window.permute(1, 0, 2)
            encoded = self.transformer_encoder(window, mask=mask)
            output = self.output_layer(encoded[-1, :, :])
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.permute(0, 2, 1)
        aggr_output = self.aggregation_layer(outputs)
        final_output = self.final_fc(aggr_output)

        return final_output
    

class FeatureMapEncoder(nn.Module):
    """
    Encodes maps in grid_maze2d through CNNs into 1D embeddings as 
    conditions of the diffusion model. 
    """
    def __init__(
            self, 
            inp_dim, 
            out_dim,
    ):
        super().__init__()

        ## 2D Convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Mish(), 
        )

        self.fc = nn.Sequential(
            nn.Linear(
                inp_dim, 
                out_dim
            ), 
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x


class FullyConvEnvMapEncoder(nn.Module):
    """
    Encodes maps in grid_maze2d through CNNs into 1D embeddings as 
    conditions of the diffusion model. 
    """
    def __init__(
            self, 
            inp_dim, 
            out_dim, 
            hidden_dims=(64, 128, 256), 
            n_groups=8, 
            dropout=0, 
            num_heads=1, 
            num_head_dim=-1, 
            pooling=True,
    ):
        super().__init__()

        ## 2D Convolutional layers
        channels = [
            inp_dim, *hidden_dims, 
        ]
        in_outs = list(zip(channels[:-1], channels[1:]))

        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    inp_dim, 
                    hidden_dims[0], 
                    kernel_size=3, 
                    padding=1
                )
            )
        ])

        for ind, (dim_in, dim_out) in enumerate(in_outs[1:]):
            layers = [
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
            ]
            if pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.feature_extractor.append(nn.Sequential(*layers))

        self.feature_extractor.append(
            nn.Sequential(
                nn.Conv2d(
                    hidden_dims[-1], 
                    out_dim, 
                    kernel_size=1, 
                    padding=0
                )
            )
        )

    def forward(self, x):
        for module in self.feature_extractor:
            x = module(x)
        return x