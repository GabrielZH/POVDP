from typing import Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import pdb

from povdp.networks.residual import (
    FiLMResidualBlock1d, 
    FiLMResidualBlock2d,
    TimeConditionalResidualBlock1d,
    TimeConditionalResidualBlock2d,
    HybridConditionalResBlock1d, 
    HybridConditionalResBlock2d, 
    TimestepEmbedSequential, 
)
from povdp.networks.attention import (
    # AttentionBlock1d, 
    AttentionBlock2d, 
)
from povdp.networks.elements import (
    Conv1dBlock, 
    Conv2dBlock, 
    Downsample1d, 
    Downsample2d, 
    Upsample1d, 
    Upsample2d, 
    LinearAttention2d, 
    PreNorm2d,
)
from povdp.networks.helpers import (
    SinusoidalPosEmb1d, 
    SinusoidalPosEmb2d, 
)
from povdp.utils.fp16_util import (
    convert_module_to_f16, 
    convert_module_to_f32, 
)


class ConditionalConsistencyUnet1d(nn.Module):
    """
    """
    def __init__(
            self, 
            inp_dim, 
            obs_cond_dim, 
            env_cond_dim, 
            local_cond_dim=None, 
            global_cond_dim=None, 
            dropout=0., 
            time_step_embed_dim=256,
            dim_mults=(1, 2, 4, 8), 
            conv_kernel_size=3, 
            n_groups=8, 
            cond_predict_scale=False,
            env_cond_only=True,
    ):
        super().__init__()

        hidden_dims = list(map(lambda m: time_step_embed_dim * m, dim_mults))
        all_dims = [inp_dim, *hidden_dims]
        first_hidden_dim = hidden_dims[0]
 
        time_dim = time_step_embed_dim
        time_step_encoder = nn.Sequential(
            SinusoidalPosEmb1d(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        cond_dim = time_dim
            
        if global_cond_dim is not None:
            if env_cond_only:
                global_cond_dim += env_cond_dim
            else:
                global_cond_dim += (env_cond_dim + obs_cond_dim)
        else:
            if env_cond_only:
                global_cond_dim = env_cond_dim
            else:
                global_cond_dim = env_cond_dim + obs_cond_dim
            
        cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                FiLMResidualBlock1d(
                    dim_in, 
                    dim_out, 
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                FiLMResidualBlock1d(
                    dim_in, 
                    dim_out, 
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            FiLMResidualBlock1d(
                mid_dim, 
                mid_dim, 
                cond_dim=cond_dim,
                kernel_size=conv_kernel_size, 
                n_groups=n_groups, 
                dropout=dropout, 
                cond_predict_scale=cond_predict_scale
            ),
            FiLMResidualBlock1d(
                mid_dim, 
                mid_dim, 
                cond_dim=cond_dim,
                kernel_size=conv_kernel_size, 
                n_groups=n_groups, 
                dropout=dropout, 
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                FiLMResidualBlock1d(
                    dim_in, 
                    dim_out, 
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                FiLMResidualBlock1d(
                    dim_out, 
                    dim_out, 
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                FiLMResidualBlock1d(
                    dim_out*2, 
                    dim_in, 
                    cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                FiLMResidualBlock1d(
                    dim_in, 
                    dim_in, 
                    cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(
                first_hidden_dim, 
                first_hidden_dim, 
                kernel_size=conv_kernel_size, 
                padding=conv_kernel_size // 2),
            nn.Conv1d(first_hidden_dim, inp_dim, 1),
        )

        self.time_step_encoder = time_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
            self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, 
            global_cond=None, 
            **kwargs
        ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.time_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], dim=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = rearrange(x, 'b t h -> b h t')
        return x


class ConditionalConsistencyUnet2d(nn.Module):
    """
    """
    def __init__(
            self,
            inp_dim,
            local_cond_dim=None, 
            global_cond_dim=None, 
            dropout=0., 
            time_step_embed_dim=256, 
            dim_mults=(1, 2, 4, 8), 
            conv_kernel_size=3, 
            n_groups=8, 
            cond_predict_scale=False
    ):
        super().__init__()

        hidden_dims = list(map(lambda m: time_step_embed_dim * m, dim_mults))
        all_dims = [inp_dim, *hidden_dims]
        first_hidden_dim = hidden_dims[0]
 
        time_dim = time_step_embed_dim
        time_step_encoder = nn.Sequential(
            SinusoidalPosEmb1d(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        cond_dim = time_dim
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                FiLMResidualBlock2d(
                    dim_in, 
                    dim_out,  
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                FiLMResidualBlock2d(
                    dim_in, 
                    dim_out,  
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            FiLMResidualBlock2d(
                mid_dim, 
                mid_dim,  
                cond_dim=cond_dim,
                kernel_size=conv_kernel_size, 
                n_groups=n_groups, 
                dropout=dropout, 
                cond_predict_scale=cond_predict_scale
            ),
            FiLMResidualBlock2d(
                mid_dim, 
                mid_dim,  
                cond_dim=cond_dim,
                kernel_size=conv_kernel_size, 
                n_groups=n_groups, 
                dropout=dropout, 
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                FiLMResidualBlock2d(
                    dim_in, 
                    dim_out,  
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                FiLMResidualBlock2d(
                    dim_out, 
                    dim_out, 
                    cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                Downsample2d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                FiLMResidualBlock2d(
                    dim_out*2, 
                    dim_in,  
                    cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                FiLMResidualBlock2d(
                    dim_in, 
                    dim_in,  
                    cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, 
                    n_groups=n_groups, 
                    dropout=dropout, 
                    cond_predict_scale=cond_predict_scale),
                Upsample2d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv2dBlock(
                first_hidden_dim, 
                first_hidden_dim, 
                kernel_size=conv_kernel_size, 
                padding=conv_kernel_size // 2),
            nn.Conv2d(first_hidden_dim, inp_dim, 1),
        )

        self.time_step_encoder = time_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
            self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, 
            global_cond=None, 
            **kwargs
        ):
        """
        sample: (B,env_dim,H,W)
        timestep: (B,) or int, diffusion step
        env_cond: (B,env_dim,H,W)
        obs_cond: (B,(n_env_recon_obs_steps x obs_chn),H,W)
        output: (B,env_dim,H,W)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        time_emb = self.time_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = time_emb[:, None, :, None, None].expand(
                -1, global_cond.shape[1], -1, *global_cond.shape[-2:])
            global_feature = torch.cat((global_feature, global_cond), dim=2)
        else:
            assert local_cond is not None
            global_feature = time_emb[:, None, :, None, None].expand(
                -1, local_cond.shape[1], -1, *local_cond.shape[-2:])
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample.float()
        if x.shape[-1] % 2:
            x = nn.ConstantPad2d((0, 1, 0, 0), 0)(x)
        if x.shape[-2] % 2:
            x = nn.ConstantPad2d((0, 0, 1, 0), 0)(x)
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        return x
    

class AttentionConditionalConsistencyUnet2d(nn.Module):
    def __init__(
        self, 
        inp_dim, 
        time_step_embed_dim=256, 
        local_cond_dim=None, 
        global_cond_dim=None, 
        num_res_blocks=1, 
        dim_mults=(1, 2, 4, 8), 
        # attention_resolutions=(32, 16, 8), 
        dropout=0., 
        conv_resample=True, 
        use_fp16=False, 
        n_groups=8, 
        num_heads=1, 
        num_head_dim=-1, 
        cond_predict_scale=False, 
        use_scale_shift_norm=False, 
    ):
        super().__init__()

        time_dim = time_step_embed_dim
        self.time_step_encoder = nn.Sequential(
            SinusoidalPosEmb1d(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        hidden_dims = list(map(lambda m: time_step_embed_dim * m, dim_mults))
        dim = first_hidden_dim = hidden_dims[0]

        _feature_size = dim
        ds = 1

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            dim_in = local_cond_dim
            dim_out = first_hidden_dim
            self.local_cond_encoder = nn.ModuleList([
                # down encoder
                TimestepEmbedSequential(
                    FiLMResidualBlock2d(
                        dim_in, 
                        dim_out,  
                        cond_dim=global_cond_dim,
                        n_groups=n_groups, 
                        dropout=dropout, 
                        cond_predict_scale=cond_predict_scale
                    ),
                ),
                # up encoder
                TimestepEmbedSequential(
                    FiLMResidualBlock2d(
                        dim_in, 
                        dim_out, 
                        cond_dim=global_cond_dim,  
                        n_groups=n_groups, 
                        dropout=dropout, 
                        cond_predict_scale=cond_predict_scale
                    ),
                )
                
            ])

        down_modules = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(inp_dim, dim, 3, padding=1)
            )
        ])
        down_modules_dims = [dim]
        for idx, hidden_dim in enumerate(hidden_dims):
            for _ in range(num_res_blocks):
                layers = [
                    FiLMResidualBlock2d(
                        dim, 
                        hidden_dim, 
                        cond_dim=global_cond_dim, 
                        n_groups=n_groups, 
                        dropout=dropout, 
                        cond_predict_scale=cond_predict_scale, 
                    )
                ]
                dim = hidden_dim
                # if ds in attention_resolutions:
                layers.append(
                    AttentionBlock2d(
                        dim, 
                        groups=n_groups, 
                        num_heads=num_heads, 
                        num_head_dim=num_head_dim, 
                        attention_type='flash'
                    )
                )
                down_modules.append(TimestepEmbedSequential(*layers))
                _feature_size += dim
                down_modules_dims.append(dim)
            
            if idx < len(hidden_dims) - 1:
                down_modules.append(
                    TimestepEmbedSequential(
                        Downsample2d(
                            dim, 
                            use_conv=conv_resample
                        )
                    )
                )
                down_modules_dims.append(dim)
                ds *= 2
                _feature_size += dim

        mid_modules = TimestepEmbedSequential(
            FiLMResidualBlock2d(
                dim, 
                dim, 
                cond_dim=global_cond_dim, 
                n_groups=n_groups, 
                dropout=dropout, 
                cond_predict_scale=cond_predict_scale
            ), 
            AttentionBlock2d(
                dim, 
                groups=n_groups, 
                num_heads=num_heads, 
                num_head_dim=num_head_dim, 
                attention_type='flash'
            ), 
            FiLMResidualBlock2d(
                dim, 
                dim, 
                cond_dim=global_cond_dim,  
                n_groups=n_groups, 
                dropout=dropout, 
                cond_predict_scale=cond_predict_scale, 
            ), 
        )
        _feature_size += dim

        up_modules = nn.ModuleList([])
        for idx, hidden_dim in enumerate(reversed(hidden_dims)):
            for i in range(num_res_blocks + 1):
                idim = down_modules_dims.pop()
                layers = [
                    FiLMResidualBlock2d(
                        dim + idim, 
                        hidden_dim, 
                        cond_dim=global_cond_dim,  
                        n_groups=n_groups, 
                        dropout=dropout, 
                        cond_predict_scale=cond_predict_scale
                    ), 
                ]
                dim = hidden_dim
                # if ds in attention_resolutions:
                layers.append(
                    AttentionBlock2d(
                        dim, 
                        groups=n_groups, 
                        num_heads=num_heads, 
                        num_head_dim=num_head_dim, 
                        attention_type='flash'
                    )
                )
                if idx and i == num_res_blocks:
                    layers.append(
                        Upsample2d(
                            dim, 
                            use_conv=conv_resample
                        )
                    )
                    ds //= 2
                up_modules.append(TimestepEmbedSequential(*layers))
                _feature_size += dim

        final_conv = nn.Sequential(
            Conv2dBlock(
                first_hidden_dim, 
                first_hidden_dim, 
                kernel_size=3, 
                padding=1), 
            nn.Mish(), 
            nn.Conv2d(first_hidden_dim, inp_dim, 1), 
        )

        self.dtype = torch.float16 if use_fp16 else torch.float32
        self._feature_size = _feature_size
        self.down_modules = down_modules
        self.mid_modules = mid_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.down_modules.apply(convert_module_to_f16)
        self.mid_modules.apply(convert_module_to_f16)
        self.up_modules.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
            self, 
            sample: torch.Tensor, 
            timestep: torch.Tensor, 
            local_cond=None, 
            global_cond=None
        ):
        """
        sample: (B,env_dim,H,W)
        timestep: (B,) or int, diffusion step
        env_cond: (B,env_dim,H,W)
        obs_cond: (B,(n_env_recon_obs_steps x obs_chn),H,W)
        output: (B,env_dim,H,W)
        """
        if not len(timestep.shape):
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        time_emb = self.time_step_encoder(timestep)

        if global_cond is not None:
            global_feature = time_emb[:, None, :, None, None].expand(
                -1, global_cond.shape[1], -1, *global_cond.shape[-2:])
            global_feature = torch.cat((global_feature, global_cond), dim=2)
        else:
            assert local_cond is not None
            global_feature = time_emb[:, None, :, None, None].expand(
                -1, local_cond.shape[1], -1, *local_cond.shape[-2:])

        # encode local features
        h_local = list()
        if local_cond is not None:
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample.type(self.dtype)
        h = list()
        for idx, module in enumerate(self.down_modules):
            x = module(x, global_feature)
            if not idx and len(h_local) > 0:
                x += h_local[0]
            h.append(x)
        x = self.mid_modules(x, global_feature)
        for idx, module in enumerate(self.up_modules):
            x = torch.cat([x, h.pop()], dim=1)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x += h_local[1]
            x = module(x, global_feature)
        x = x.type(sample.dtype)

        return self.final_conv(x)
