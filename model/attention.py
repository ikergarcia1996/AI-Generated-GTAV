"""
Based on https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/attention.py
"""
from typing import Optional
from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from model.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from model.embeddings import TimestepEmbedding, Timesteps, Positions2d

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        is_causal: bool = True,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.time_pos_embedding = (
            nn.Sequential(
                Timesteps(dim),
                TimestepEmbedding(in_channels=dim, time_embed_dim=dim * 4, out_dim=dim),
            )
            if rotary_emb is None
            else None
        )
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        if self.time_pos_embedding is not None:
            time_emb = self.time_pos_embedding(
                torch.arange(T, device=x.device)
            )
            x = x + rearrange(time_emb, "t d -> 1 t 1 1 d")

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
            k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(
            query=q, key=k, value=v, is_causal=self.is_causal
        )

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x

class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.space_pos_embedding = (
            nn.Sequential(
                Positions2d(dim),
                TimestepEmbedding(in_channels=dim, time_embed_dim=dim * 4, out_dim=dim),
            )
            if rotary_emb is None
            else None
        )

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        if self.space_pos_embedding is not None:
            h_steps = torch.arange(H, device=x.device)
            w_steps = torch.arange(W, device=x.device)
            grid = torch.meshgrid(h_steps, w_steps, indexing="ij")
            space_emb = self.space_pos_embedding(grid)
            x = x + rearrange(space_emb, "h w d -> 1 1 h w d")

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        if self.rotary_emb is not None:
            freqs = self.rotary_emb.get_axial_freqs(H, W)
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(
            query=q, key=k, value=v, is_causal=False
        )

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x

