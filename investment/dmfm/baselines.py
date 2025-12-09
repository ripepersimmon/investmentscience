"""Baseline models for comparison with DMFM.

Implements several standard factor-based and ML-based models:
1. Linear Factor Model - OLS regression on raw factors
2. MLP Factor Model - Simple feedforward network
3. MLP + GAT - MLP with graph attention (no neutralization)
4. Equal Weight - Simple 1/N portfolio (benchmark)
5. Momentum - Classic momentum factor strategy
6. Mean Reversion - Short-term reversal strategy
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


class LinearFactorModel(nn.Module):
    """Simple linear model: f = W @ features + b
    
    Equivalent to cross-sectional OLS regression.
    """
    
    def __init__(self, feature_dim: int, horizons: Tuple[int, ...] = (3, 5, 10, 15, 20)) -> None:
        super().__init__()
        self.horizons = horizons
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(feature_dim, 1) for h in horizons
        })
    
    def forward(
        self,
        features: torch.Tensor,
        industry_mask: Optional[torch.Tensor] = None,
        universe_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, N, F)
        Returns:
            dict with 'factors' key containing predictions per horizon
        """
        factors = {}
        for h, head in self.heads.items():
            f = head(features).squeeze(-1)  # (B, N)
            factors[h] = f
        
        return {
            "factors": factors,
            "recon": factors,  # dummy for loss compatibility
            "attn": {h: torch.zeros_like(features) for h in self.heads.keys()},
        }


class MLPFactorModel(nn.Module):
    """Multi-layer perceptron factor model.
    
    Features -> MLP -> factor scores per horizon.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizons: Tuple[int, ...] = (3, 5, 10, 15, 20),
    ) -> None:
        super().__init__()
        self.horizons = horizons
        
        layers = [nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*layers)
        
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim, 1) for h in horizons
        })
    
    def forward(
        self,
        features: torch.Tensor,
        industry_mask: Optional[torch.Tensor] = None,
        universe_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, F = features.shape
        x = features.reshape(B * N, F)
        hidden = self.encoder(x).reshape(B, N, -1)
        
        factors = {}
        for h, head in self.heads.items():
            f = head(hidden).squeeze(-1)
            factors[h] = f
        
        return {
            "factors": factors,
            "recon": factors,
            "attn": {h: torch.zeros_like(features) for h in self.heads.keys()},
        }


class MLPGATModel(nn.Module):
    """MLP + GAT model without neutralization.
    
    Uses graph attention but doesn't subtract the influence (no neutralization).
    This tests whether DMFM's neutralization is important.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
        horizons: Tuple[int, ...] = (3, 5, 10, 15, 20),
    ) -> None:
        super().__init__()
        self.horizons = horizons
        
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Simple attention layer
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim * 2, 1) for h in horizons  # concat original + attended
        })
    
    def forward(
        self,
        features: torch.Tensor,
        industry_mask: Optional[torch.Tensor] = None,
        universe_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, F = features.shape
        x = features.reshape(B * N, F)
        hidden = self.encoder(x).reshape(B, N, -1)  # (B, N, D)
        
        # Self-attention
        Q = self.attn_q(hidden)
        K = self.attn_k(hidden)
        V = self.attn_v(hidden)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden.size(-1) ** 0.5)
        if industry_mask is not None:
            scores = scores.masked_fill(~industry_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        attended = torch.matmul(attn, V)  # (B, N, D)
        
        # Concatenate (no subtraction - key difference from DMFM)
        combined = torch.cat([hidden, attended], dim=-1)
        
        factors = {}
        for h, head in self.heads.items():
            f = head(combined).squeeze(-1)
            factors[h] = f
        
        return {
            "factors": factors,
            "recon": factors,
            "attn": {h: torch.zeros_like(features) for h in self.heads.keys()},
        }


class MomentumModel(nn.Module):
    """Classic momentum factor model.
    
    Uses past returns as the factor score. Doesn't require training.
    """
    
    def __init__(
        self,
        feature_dim: int,
        momentum_col: int = 0,  # index of momentum feature (e.g., return_20d)
        horizons: Tuple[int, ...] = (3, 5, 10, 15, 20),
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.momentum_col = momentum_col
        # Dummy parameter for optimizer compatibility
        self.dummy = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        features: torch.Tensor,
        industry_mask: Optional[torch.Tensor] = None,
        universe_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Use momentum feature directly as factor score
        momentum = features[:, :, self.momentum_col]  # (B, N)
        
        factors = {str(h): momentum for h in self.horizons}
        
        return {
            "factors": factors,
            "recon": factors,
            "attn": {str(h): torch.zeros_like(features) for h in self.horizons},
        }


class MeanReversionModel(nn.Module):
    """Mean reversion (short-term reversal) model.
    
    Uses negative of short-term returns as factor score.
    """
    
    def __init__(
        self,
        feature_dim: int,
        return_col: int = 0,  # index of short-term return feature
        horizons: Tuple[int, ...] = (3, 5, 10, 15, 20),
    ) -> None:
        super().__init__()
        self.horizons = horizons
        self.return_col = return_col
        self.dummy = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        features: torch.Tensor,
        industry_mask: Optional[torch.Tensor] = None,
        universe_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Negative return = buy losers, sell winners
        reversal = -features[:, :, self.return_col]  # (B, N)
        
        factors = {str(h): reversal for h in self.horizons}
        
        return {
            "factors": factors,
            "recon": factors,
            "attn": {str(h): torch.zeros_like(features) for h in self.horizons},
        }


def get_baseline_model(
    name: str,
    feature_dim: int,
    hidden_dim: int = 64,
    horizons: Tuple[int, ...] = (3, 5, 10, 15, 20),
    **kwargs,
) -> nn.Module:
    """Factory function to create baseline models."""
    
    models = {
        "linear": LinearFactorModel,
        "mlp": MLPFactorModel,
        "mlp_gat": MLPGATModel,
        "momentum": MomentumModel,
        "reversal": MeanReversionModel,
    }
    
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    if name == "linear":
        return LinearFactorModel(feature_dim, horizons)
    elif name == "mlp":
        return MLPFactorModel(feature_dim, hidden_dim, horizons=horizons, **kwargs)
    elif name == "mlp_gat":
        return MLPGATModel(feature_dim, hidden_dim, horizons=horizons, **kwargs)
    elif name == "momentum":
        return MomentumModel(feature_dim, horizons=horizons, **kwargs)
    elif name == "reversal":
        return MeanReversionModel(feature_dim, horizons=horizons, **kwargs)


__all__ = [
    "LinearFactorModel",
    "MLPFactorModel", 
    "MLPGATModel",
    "MomentumModel",
    "MeanReversionModel",
    "get_baseline_model",
]
