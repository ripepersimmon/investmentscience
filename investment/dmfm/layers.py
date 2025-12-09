"""Core layers for the Deep Multi-Factor Model (DMFM).

Implements a lightweight masked graph attention module for industry and
universe influence blocks described in the paper.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply softmax with additive mask where masked positions are -inf.

    Args:
        scores: (B, H, N, N)
        mask: (B, N, N) boolean; True = keep, False = mask out.
    """
    mask_expanded = mask.unsqueeze(1)  # (B, 1, N, N)
    scores = scores.masked_fill(~mask_expanded, float("-inf"))
    return torch.softmax(scores, dim=-1)


class MaskedGraphAttention(nn.Module):
    """Multi-head scaled dot-product attention restricted by an adjacency mask.

    This is a simplified GAT variant: no learnable edge weights, but it
    respects the time-varying industry/universe masks the paper requires.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, negative_slope: float = 0.1) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked attention.

        Args:
            x: (B, N, D) stock context features.
            mask: (B, N, N) boolean adjacency (1 keeps edge).
        Returns:
            (B, N, D) aggregated messages.
        """
        B, N, _ = x.shape
        qkv = self.qkv(x)  # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, H, N, dk)
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.heads, self.dk).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        # scaled dot-product attention with mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B, H, N, N)
        attn = masked_softmax(scores, mask)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, N, dk)
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        out = self.out(out)
        return self.act(out)


def build_full_mask(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return an all-ones adjacency (universe) mask with self-loops."""
    m = torch.ones(n, n, dtype=torch.bool, device=device)
    return m


__all__ = [
    "MaskedGraphAttention",
    "build_full_mask",
    "masked_softmax",
]
