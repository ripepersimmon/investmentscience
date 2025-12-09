"""Deep Multi-Factor Model (DMFM) as described in the paper.

Implements:
- Stock context encoder (BatchNorm + MLP)
- Industry graph attention + neutrality
- Universe graph attention + neutrality
- Multi-head deep factor learners for multiple forward horizons
- Factor attention module for interpretability
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn

from .config import ModelConfig
from .layers import MaskedGraphAttention, build_full_mask


class StockContextEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, F = x.shape
        x_flat = x.reshape(B * N, F)
        x_norm = self.bn(x_flat)
        hidden = self.mlp(x_norm)
        return hidden.view(B, N, -1)


class FactorAttention(nn.Module):
    """Factor Attention Module as described in the paper (Eq. 9-12).

    Paper equations:
        U_k^t = LeakyReLU(W_{a,k}^T * F_t)  -- Eq. (9)
        A_k^t = softmax(U_k^t)              -- Eq. (10)
        a_bar_k^t = (1/n) * sum_i(a_{ik}^t) -- Eq. (11)
        f_hat_k^t = F_t^T * a_bar_k^t       -- Eq. (12)
    
    The attention weight matrix A_k^t (N x m) shows how much each original
    feature contributes to the deep factor for each stock.
    """

    def __init__(self, feature_dim: int, num_stocks: int = None, negative_slope: float = 0.1) -> None:
        super().__init__()
        # W_{a,k} projects features to attention logits
        # Paper uses W_{a,k}^T in R^{n x n}, but we adapt to variable stock counts
        self.proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, features: torch.Tensor, factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and factor estimate per paper Eq. (9)-(12).

        Args:
            features: (B, N, m) original input factors F_t
            factor: (B, N) deep factor f_k^t (used for reference, not in attention calc)
        Returns:
            f_hat: (B, N) reconstructed factor via attention (Eq. 12)
            attn_weights: (B, N, m) attention weights A_k^t (Eq. 10)
        """
        B, N, m = features.shape
        
        # Eq. (9): U_k^t = LeakyReLU(W_{a,k}^T * F_t)
        U = self.act(self.proj(features))  # (B, N, m)
        
        # Eq. (10): A_k^t = softmax(U_k^t) - softmax over feature dimension
        A = torch.softmax(U, dim=-1)  # (B, N, m)
        
        # Eq. (11): a_bar_k^t = (1/n) * sum_i(a_{ik}^t) - average over stocks
        a_bar = A.mean(dim=1)  # (B, m)
        
        # Eq. (12): f_hat_k^t = F_t^T * a_bar_k^t
        # This gives (B, N) - weighted combination of features for each stock
        f_hat = torch.einsum('bnm,bm->bn', features, a_bar)  # (B, N)
        
        return f_hat, A


class DeepFactorHead(nn.Module):
    def __init__(self, input_dim: int, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        out = self.linear(x).squeeze(-1)  # (B, N)
        return self.act(out)


class DMFM(nn.Module):
    """Deep Multi-Factor Model from the paper."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = StockContextEncoder(cfg.feature_dim, cfg.hidden_dim)
        self.industry_gat = MaskedGraphAttention(cfg.hidden_dim, heads=cfg.gat_heads, dropout=cfg.gat_dropout, negative_slope=cfg.leaky_relu_slope)
        self.universe_gat = MaskedGraphAttention(cfg.hidden_dim, heads=cfg.gat_heads, dropout=cfg.gat_dropout, negative_slope=cfg.leaky_relu_slope)

        deep_dim = cfg.hidden_dim * 3  # Ct || C_bar_I || C_bar_U
        self.heads = nn.ModuleDict({str(h): DeepFactorHead(deep_dim, negative_slope=cfg.leaky_relu_slope) for h in cfg.horizons})
        self.factor_attn = nn.ModuleDict({str(h): FactorAttention(cfg.feature_dim, negative_slope=cfg.leaky_relu_slope) for h in cfg.horizons})

    def forward(
        self,
        features: torch.Tensor,
        industry_mask: torch.Tensor,
        universe_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the DMFM forward pass.

        Args:
            features: (B, N, F) raw input factors.
            industry_mask: (B, N, N) industry adjacency (bool).
            universe_mask: (B, N, N) universe adjacency (bool). If None, uses full mask.
        Returns:
            dict with keys: contexts, factors, attn, recon
        """
        B, N, _ = features.shape
        if universe_mask is None:
            universe_mask = build_full_mask(N, features.device).unsqueeze(0).expand(B, -1, -1)

        # Stock context
        C = self.encoder(features)  # (B, N, m1)

        # Industry influence and neutrality
        H_I = self.industry_gat(C, industry_mask)
        C_bar_I = C - H_I

        # Universe influence and neutrality
        H_U = self.universe_gat(C_bar_I, universe_mask)
        C_bar_U = C_bar_I - H_U

        # Deep factors per horizon
        concat_ctx = torch.cat([C, C_bar_I, C_bar_U], dim=-1)
        factors = {}
        recon = {}
        attn = {}
        for h, head in self.heads.items():
            f = head(concat_ctx)  # (B, N)
            f_hat, w = self.factor_attn[h](features, f)
            factors[h] = f
            recon[h] = f_hat
            attn[h] = w

        return {
            "C": C,
            "C_bar_I": C_bar_I,
            "C_bar_U": C_bar_U,
            "H_I": H_I,
            "H_U": H_U,
            "factors": factors,
            "recon": recon,
            "attn": attn,
        }


__all__ = ["DMFM", "StockContextEncoder", "FactorAttention", "DeepFactorHead"]
