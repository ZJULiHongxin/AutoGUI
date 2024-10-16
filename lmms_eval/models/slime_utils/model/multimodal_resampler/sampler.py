# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
import numpy as np
import pdb
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_, normal_

class IdentityMap(nn.Module):
    def __init__(self, hiiden, **kwargs):
        super().__init__()
        

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": 'identity'}

def get_abs_pos(abs_pos, tgt_size):
    src_size = int(math.sqrt(abs_pos.size(0)))
    dtype = abs_pos.dtype

    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            llm_hidden_size=4096,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_post_proj=False
    ):
        super().__init__()
        self.module_type = 'resampler'
        self.num_queries = grid_size ** 2
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).to(torch.float16)
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim), requires_grad=True)
        normal_(self.query, std=.02)
        # self.query = torch.clamp(self.query, min=-2, max=2)
        # trunc_normal_(self.query, std=.02)


        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        
        if use_post_proj:
            self.proj = nn.Linear(embed_dim, llm_hidden_size)
        else:
            self.proj = nn.Identity()


    def forward(self, x, tgt_size=(24,24), text=None, attn_mask=None):
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
            mark=True
        else:
            mark=False
        if x.shape[1] != tgt_size[0] * tgt_size[1]:
            tgt_size = (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))

        pos_embed = get_abs_pos(self.pos_embed.detach(), tgt_size).detach()
        if torch.isnan(self.pos_embed).any():
            # some init error
            self.pos_embed = nn.Parameter(
                torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)).to(torch.float16).to(x.device)
            ).requires_grad_(False)
        pos_embed = get_abs_pos(self.pos_embed.detach(), tgt_size).detach()
        
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)
        
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1).detach(), 
            x + pos_embed.unsqueeze(1),
            x)[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = self.proj(x)
        return x if not mark else x.squeeze()

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
    
class ResamplerWithText(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            llm_hidden_size=4096,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_post_proj=False
    ):
        super().__init__()
        self.module_type = 'instructblip'
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float().to(torch.float16)
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim), requires_grad=True)
        trunc_normal_(self.query, std=.02)

        if llm_hidden_size is not None and llm_hidden_size != embed_dim:
            self.kv_proj = nn.Linear(llm_hidden_size, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        
        if use_post_proj:
            self.proj = nn.Linear(embed_dim, llm_hidden_size)
        else:
            self.proj = nn.Identity()

    def forward(self, x, tgt_size=(24,24), text=None, attn_mask=None):
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
        if len(text.shape) <= 2:
            text = text.unsqueeze(0)
            attn_mask = attn_mask.unsqueeze(0)
        if x.shape[1] != tgt_size[0] * tgt_size[1]:
            tgt_size = (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))
            
        pos_embed = get_abs_pos(self.pos_embed.detach(), tgt_size).detach()
        
        N = x.shape[0]
        
        text = self.kv_proj(text)
        text = self.ln_kv(text)
        
        text, x = text.permute(1, 0, 2), x.permute(1, 0, 2)
        
        query = self._repeat(self.query, N)

        contate_query_text = torch.cat([query, text], dim=0)
        contate_attn_mask = torch.cat([torch.zeros((N, self.num_queries), dtype=attn_mask.dtype, device=attn_mask.device) , ~attn_mask], dim=-1).bool()
        contate_query_text = self.self_attn(
            contate_query_text,
            contate_query_text,
            contate_query_text, 
            key_padding_mask=contate_attn_mask)[0]
        
        query = contate_query_text[:self.query.shape[0]]
        query = self.ln_q(query)
        
        out = self.attn(
            query + self.pos_embed.unsqueeze(1).detach(), 
            x + pos_embed.unsqueeze(1).detach(),
            x)[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = self.proj(x)
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

from einops import rearrange

class MplugDocOwlHReducerModel(nn.Module):
    def __init__(self, embed_dim, conv_shape='1x4'):
        super().__init__()
        self.module_type = 'h-reducer'
        self.ln_q = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.conv_shape = tuple(map(int, conv_shape.split('x')))
        self.conv_patch=self.conv_shape[0]*self.conv_shape[1]
        ## feature interaction with a conv layer
        self.reducer_before = torch.nn.Sequential(
            nn.Conv2d(embed_dim, self.conv_patch*embed_dim, kernel_size=self.conv_shape, stride=self.conv_shape, bias=True),
            nn.GELU()
        )
        self.reducer_before[0]
        ## reduce visual feature length with a conv layer
        self.reducer = nn.Conv2d(embed_dim, embed_dim, kernel_size=self.conv_shape, stride=self.conv_shape, bias=True)    

        # Initialize weights and biases for the convolutional layer
        self.init_weights()

    def init_weights(self):
        # Get the first layer of the reducer_before sequence, which is the Conv2d layer
        conv_layer = self.reducer_before[0]
        
        # Initialize the weights from a normal distribution
        trunc_normal_(conv_layer.weight, std=.02)
        
        # Initialize the biases from a normal distribution, if biases are used
        if conv_layer.bias is not None:
            trunc_normal_(conv_layer.bias, std=.02)

    def forward(
        self,
        encoder_hidden_states
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            batch_size is the number of all images (global+crop) in a batch
            Sequence of hidden-states at the output of the last layer of the encoder.
        """
        B, L, C = encoder_hidden_states.shape # B, 1024=(336/14)^2, 1024

        ## feature interaction with a conv layer
        encoder_hidden_states = rearrange(encoder_hidden_states, 'B (H W) D -> B D H W', H=int(math.sqrt(L)))
        hidden_states = self.reducer_before(encoder_hidden_states) # B 4D H W/4
        ## reduce seq length with a conv layer
        hidden_states = rearrange(hidden_states, 'B (X D) H W -> B D H (W X)', X=self.conv_patch) # B 4D H W/4 -> B D H W
        sequence_output = self.reducer(hidden_states) # B,C,H,W -> B,C,H/conv_shape[0],W/(conv_shape[1])
        sequence_output = sequence_output.flatten(2).transpose(1, 2)  # B,C,H/conv_shape[0],W/(conv_shape[1]) -> B,C,L/conv_patch -> B,L/conv_patch,C
        sequence_output = sequence_output.transpose(0, 1).contiguous() # L/conv_patch, B, C

        return sequence_output

class Merger(nn.Module):
    """
    InternLM
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            llm_hidden_size=4096
    ):
        super().__init__()
        self.module_type = 'merger'
        self.patch_size = int((576 / grid_size ** 2) ** 0.5)
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.llm_hidden_size = llm_hidden_size
        modules = [nn.Linear(embed_dim * self.patch_size * self.patch_size, embed_dim)]
        self.projection = nn.Sequential(*modules)

    def forward(self, x):
        # x: B x 576 (#tokens) x 1024
        if len(x.shape) <= 2:
            x = x.unsqueeze(0)
            mark=True
        else:
            mark=False
        
        B, NUM_PATCHES, DIM = x.shape
        num_patches = int(NUM_PATCHES ** 0.5)
        
        # Refer to https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b/blob/a2c222ebd3a723c3dff00232e4f5cc6429f472d1/build_mlp.py#L102
        x = x.reshape(B,num_patches,num_patches,DIM).reshape(B,num_patches//2,2,num_patches//2,2,DIM).contiguous().permute(0,1,3,2,4,5).reshape(B,num_patches//2,num_patches//2,4*DIM).flatten(1,2).contiguous()
        # reshape(B, num_patches, num_patches, DIM).unfold(1, self.patch_size, 2).unfold(2, self.patch_size, 2).reshape(B, -1, DIM * self.patch_size * self.patch_size) # B x num_patches_H x num_patches_W x DIM x 2 x 2

        x = self.projection(x)

        return x if not mark else x.squeeze()

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

class MultiCompressor(nn.Module):
    def __init__(
            self,
            combination, # ='multi_resampler_h-reducer',
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            llm_hidden_size=4096,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_post_proj=False,
            aggr_type: str = 'avg'
    ):
        super().__init__()
        self.module_type = 'multi_compressor'
        self.projector_list = []
        self.combination = combination.split('_')[1:] # 'multi_resampler_h-reducer'
        for module_type in self.combination:
            module_type = module_type.lower()
            if module_type == 'resampler':
                self.projector_list.append(Resampler(grid_size, embed_dim, num_heads, kv_dim, llm_hidden_size, norm_layer, use_post_proj))
            elif module_type == 'h-reducer':
                self.projector_list.append(MplugDocOwlHReducerModel(embed_dim))
            elif module_type == 'merger':
                self.projector_list.append(Merger(embed_dim))

        self.projector_list = nn.ModuleList(self.projector_list)
        
        self.aggr_type = aggr_type # 'concat'
            
    def aggregate(self, features_all_compressors: list):
        B = len(features_all_compressors[0])

        if self.aggr_type == 'concat':
            aggr_results = [torch.cat([features_all_compressors[comp_idx][batch_idx] for comp_idx in range(len(features_all_compressors))], dim=0) for batch_idx in range(B)]
        elif self.aggr_type == 'avg':
            aggr_results = [torch.mean(torch.stack([features_all_compressors[comp_idx][batch_idx] for comp_idx in range(len(features_all_compressors))]), dim=0) for batch_idx in range(B)]
            
        return aggr_results

def build_pos_embeds(
    num_input_tokens: int, vision_hidden_size: int, use_posemb: bool = True
):
    # pos emb
    if use_posemb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(num_eos_tokens, output_hidden_size: int):
    # think tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=0.02)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(encoder_hidden_size, use_prenorm=False):
    if use_prenorm:
        prenorm = LayerNorm(encoder_hidden_size)
    else:
        prenorm = None
    return prenorm

class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        output_hidden_size: int, # llm_hidden_size
        num_eos_tokens: int = 0,
        use_posemb: bool = True,
        use_prenorm: bool = False,
        vit_num_input_tokens: int=576
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_input_tokens = vit_num_input_tokens

        self.encoder_hidden_size = embed_dim
        self.hidden_size = embed_dim
        self.output_hidden_size = output_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(num_eos_tokens, output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(self.num_input_tokens, embed_dim, use_posemb=use_posemb)

        self.prenorm = build_prenorm(embed_dim, use_prenorm)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder),
                including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x_with_pos_emb = x + self.pos_emb

        x = self._forward(x_with_pos_emb)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)

        return x
    
    # def _load_from_state_dict(self, state_dict, *args, **kwargs):
    #     # update old ckpt compatible with current code
    #     pos_emb = state_dict["abstractor.pos_emb"]
    #     if pos_emb.size(1) == self.pos_emb.size(1) + 1:
    #         # remove obsolete first pos emb (for cls token originally)
    #         state_dict["abstractor.pos_emb"] = pos_emb[:, 1:]

    #     super()._load_from_state_dict(state_dict, *args, **kwargs)

def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)

class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        print('cabs')
        return x

from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d

class CAbstractor(ConvProjector):
    """C-Abstractor based on RegBlock"""
    def build_net(self):
        self.module_type = 'Cabs'
        depth = 1
        mlp_depth = 2

        assert (self.num_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(self.num_queries ** 0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            self.encoder_hidden_size,
            self.hidden_size // 2, # 为了保持和Qformer一样的参数量（4499456）
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            self.hidden_size // 2,
            self.hidden_size,
        )

        if depth:
            self.net = nn.Sequential(s1, sampler, s2)
            self.readout = nn.Identity()#build_mlp(mlp_depth, self.hidden_size, self.output_hidden_size)
        else:
            self.net = sampler
            self.readout = nn.Identity()#build_mlp(mlp_depth, self.encoder_hidden_size, self.output_hidden_size)