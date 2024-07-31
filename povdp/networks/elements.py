import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(
            self, 
            dim, 
            out_dim=None, 
            use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.out_dim = out_dim or dim
        if use_conv:
            self.down_sampling = nn.Conv1d(dim, self.out_dim, 3, 2, 1)
        else:
            assert dim == self.out_dim
            self.down_sampling = nn.AvgPool1d(2, 2)

    def forward(self, x):
        return self.down_sampling(x)
    

class Downsample2d(nn.Module):
    def __init__(
            self, 
            dim, 
            out_dim=None, 
            use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.out_dim = out_dim or dim
        if use_conv:
            self.down_sampling = nn.Conv2d(dim, self.out_dim, 3, 2, 1)
        else:
            self.down_sampling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.down_sampling(x)


class Upsample1d(nn.Module):
    def __init__(
            self, 
            dim, 
            out_dim=None, 
            use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.out_dim = out_dim or dim
        if use_conv:
            self.up_sampling = nn.ConvTranspose1d(dim, self.out_dim, 4, 2, 1)

    def forward(self, x):
        if self.use_conv:
            return self.up_sampling(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest_exact')
        return x
    

class Upsample2d(nn.Module):
    def __init__(self, dim, out_dim=None, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.out_dim = out_dim or dim
        if use_conv:
            self.up_sampling = nn.ConvTranspose2d(dim, self.out_dim, 4, 2, 1)

    def forward(self, x):
        if self.use_conv:
            return self.up_sampling(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest_exact')
        return x


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(
            self, 
            inp_channels, 
            out_channels, 
            kernel_size, 
            padding=0, 
            n_groups=8, 
            dropout=0.
        ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, 
                out_channels, 
                kernel_size, 
                padding=padding),
            nn.Dropout(dropout),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Conv2dBlock(nn.Module):
    def __init__(
            self, 
            inp_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            dropout=0., 
            n_groups=8
        ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                inp_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding),
            nn.Dropout(dropout),
            Rearrange('b c h w -> b c 1 h w'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('b c 1 h w -> b c h w'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class ScaleShiftConv1dBlock(nn.Module):
    def __init__(
            self, 
            inp_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            dropout=0., 
            n_groups=8
        ):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(n_groups, inp_channels), 
            nn.Mish(), 
            nn.Dropout(dropout), 
            nn.Conv1d(
                inp_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            ),
        )

    def forward(self, x, embed):
        norm_layer, other_layers = self.block[0], self.block[1:]
        scale, shift = torch.chunk(embed, 2, dim=1)
        x = norm_layer(x) * (1 + scale) + shift
        x = other_layers(x)
        return x


class ScaleShiftConv2dBlock(nn.Module):
    def __init__(
            self, 
            inp_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            dropout=0., 
            n_groups=8
        ):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(n_groups, inp_channels), 
            nn.Mish(), 
            nn.Dropout(dropout), 
            nn.Conv2d(
                inp_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            ),
        )

    def forward(self, x, embed):
        norm_layer, other_layers = self.block[0], self.block[1:]
        scale, shift = torch.chunk(embed, 2, dim=1)
        x = norm_layer(x) * (1 + scale) + shift
        x = other_layers(x)
        return x

class LinearAttention2d(nn.Module):
    def __init__(self, inp_dim, n_heads=4, head_dim=32):
        super().__init__()
        self.heads = n_heads
        out_dim = inp_dim
        hidden_dim = head_dim * n_heads
        self.to_qkv = nn.Conv2d(inp_dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, 
            'b (qkv heads c) h w -> qkv b heads c (h w)', 
            heads=self.heads, 
            qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(
            out, 
            'b heads c (h w) -> b (heads c) h w', 
            heads=self.heads, 
            h=h, 
            w=w
        )
        return self.to_out(out)


class PreNorm2d(nn.Module):
    def __init__(self, inp_dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(inp_dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    