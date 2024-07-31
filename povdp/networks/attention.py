import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from linformer import Linformer

import einops
# from flash_attn.flash_attention import FlashAttention
from .helpers import zero_module


# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv, encoder_kv=None):
#         """
#         Apply QKV attention.

#         :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.chunk(3, dim=1)
#         if encoder_kv is not None:
#             assert encoder_kv.shape[1] == 2 * ch * self.n_heads
#             ek, ev = encoder_kv.chunk(2, dim=1)
#             k = torch.cat([ek, k], dim=-1)
#             v = torch.cat([ev, v], dim=-1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = torch.einsum(
#             "bct,bcs->bts",
#             (q * scale).view(bs * self.n_heads, ch, length),
#             (k * scale).view(bs * self.n_heads, ch, -1),
#         )  # More stable with f16 than dividing afterwards
#         weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
#         return a.reshape(bs, -1, length)

#     @staticmethod
#     def count_flops(model, _x, y):
#         return count_flops_attn(model, _x, y)


# class AttentionPool1d(nn.Module):
#     def __init__(
#         self, 
#         horizon: int, 
#         embed_dim: int, 
#         num_heads_channels: int, 
#         out_dim: int = None,
#     ):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(
#             torch.randn(embed_dim, horizon**2 + 1) / embed_dim**.5
#         ) 
#         self.qkv_proj = nn.Conv1d(embed_dim, 3 * embed_dim, 1)
#         self.c_proj = nn.Conv1d(embed_dim, out_dim or embed_dim, 1)
#         self.num_heads = embed_dim // num_heads_channels
#         self.attention = QKVAttention(self.num_heads)

#     def forward(self, x):
#         b, c, *_ = x.shape
#         x = x.reshape(b, c, -1)  # NC(H)
#         x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(H+1)
#         x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(H+1)
#         x = self.qkv_proj(x)
#         x = self.attention(x)
#         x = self.c_proj(x)
#         return x[:, :, 0]


# class AttentionPool2d(nn.Module):
#     def __init__(
#         self, 
#         spatial_dim: int, 
#         embed_dim: int, 
#         num_heads_channels: int, 
#         out_dim: int = None,
#     ):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(
#             torch.randn(embed_dim, spatial_dim**2 + 1) / embed_dim**.5
#         ) 
#         self.qkv_proj = nn.Conv1d(embed_dim, 3 * embed_dim, 1)
#         self.c_proj = nn.Conv1d(embed_dim, out_dim or embed_dim, 1)
#         self.num_heads = embed_dim // num_heads_channels
#         self.attention = QKVAttention(self.num_heads)

#     def forward(self, x):
#         b, c, *_ = x.shape
#         x = x.reshape(b, c, -1)  # NC(XY)
#         x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(XY+1)
#         x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(XY+1)
#         x = self.qkv_proj(x)
#         x = self.attention(x)
#         x = self.c_proj(x)
#         return x[:, :, 0]


class AttentionBlock1d(nn.Module):

    def __init__(
        self,
        dim, 
        groups=8, 
        num_heads=1,
        num_head_dim=-1,
        attention_type="flash",
        encoder_dim=None,
    ):
        super().__init__()
        if num_head_dim == -1:
            self.num_heads = num_heads
        else:
            assert (
                dim % num_head_dim == 0
            ), f"q,k,v dimensions {dim} is not divisible by num_head_dim {num_head_dim}"
            self.num_heads = dim // num_head_dim
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        if attention_type == "flash":
            self.attention = QKVFlashAttention(dim, self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        if encoder_dim is not None:
            assert attention_type != "flash"
            self.encoder_kv = nn.Conv1d(encoder_dim, dim * 2, 1)
        
        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.proj_out = nn.Conv1d(dim, dim, 1)
        

    def forward(self, x, encoder_out=None):
        if encoder_out is None:
            inputs = (x,)
        else:
            inputs = (x, encoder_out,)

        return self._forward(*inputs)

    def _forward(self, x, encoder_out=None):
        qkv = self.qkv(self.norm(x))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            inputs = (qkv, encoder_out,)
        else:
            inputs = (qkv,)

        h = self.attention(*inputs)
        h = self.proj_out(h)
        return x + h


class AttentionBlock2d(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        dim, 
        groups=8, 
        num_heads=1,
        num_head_dim=-1,
        attention_type="flash",
        encoder_dim=None,
    ):
        super().__init__()
        if num_head_dim == -1:
            self.num_heads = num_heads
        else:
            assert (
                dim % num_head_dim == 0
            ), f"q,k,v dimensions {dim} is not divisible by num_head_dim {num_head_dim}"
            self.num_heads = dim // num_head_dim
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        if attention_type == "flash":
            self.attention = QKVFlashAttention(dim, self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        if encoder_dim is not None:
            assert attention_type != "flash"
            self.encoder_kv = nn.Conv1d(encoder_dim, dim * 2, 1)
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        

    def forward(self, x, encoder_out=None):
        if encoder_out is None:
            inputs = (x,)
        else:
            inputs = (x, encoder_out,)

        return self._forward(*inputs)

    def _forward(self, x, encoder_out=None):
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            inputs = (qkv, encoder_out,)
        else:
            inputs = (qkv,)

        h = self.attention(*inputs)
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h
    

class QKVFlashAttention(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            attention_dropout=0., 
            causal=False, 
            device=None, 
            dtype=None, 
            **kwargs,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        assert not embed_dim % num_heads, \
            "embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads

        self.inner_attn = FlashAttention(
            attention_dropout=attention_dropout
        )
    
    def forward(
            self, 
            qkv, 
            attn_mask=None, 
            key_padding_mask=None, 
            need_weights=False
        ):
        qkv = einops.rearrange(
            qkv, 'b (t h d) s -> b s t h d', t=3, h=self.num_heads)
        qkv, _ = self.inner_attn(
            qkv, 
            key_padding_mask=key_padding_mask, 
            need_weights=need_weights, 
            causal=self.causal
        )
        return einops.rearrange(qkv, 'b s h d -> b (h d) s')
    

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        batch_size, width, length = qkv.shape
        assert not width % (3 * self.n_heads)
        chn = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(
            batch_size * self.n_heads, chn * 3, length
        ).split(chn, dim=1)
        scale = 1 / math.sqrt(math.sqrt(chn))
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight, v)
        return a.reshape(batch_size, -1, length)
    
    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class LinformerAttentionBlock2d(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other using Linformer.
    """
    def __init__(
        self,
        dim, 
        seq_len,  # maximum sequence length
        reduction_factor=4,  # reduction factor for the sequence length
        groups=8, 
        num_heads=1,
        num_head_dim=-1,
        encoder_dim=None,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.seq_len = seq_len
        self.reduction_factor = reduction_factor
        self.attention = Linformer(
            dim=dim,
            seq_len=seq_len,
            depth=1,  # number of attention layers
            heads=num_heads,
            k=seq_len // reduction_factor  # reduction factor for sequence length
        )
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        
        if encoder_dim is not None:
            self.encoder_kv = nn.Conv1d(encoder_dim, dim * 2, 1)

    def forward(self, x, encoder_out=None):
        b, c, h, w = x.shape
        x = self.norm(x)
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, c * 3, h * w).permute(0, 2, 1)  # (b, seq_len, dim * 3)
        
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            encoder_out = encoder_out.permute(0, 2, 1)  # (b, seq_len, dim * 2)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        
        h = h.permute(0, 2, 1).reshape(b, c, h, w)
        h = self.proj_out(h)
        
        return x + h