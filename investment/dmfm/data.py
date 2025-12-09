"""Minimal dataset utilities for DMFM.

This intentionally avoids any dependency on existing code in the workspace.
Provide tensors directly to build a dataset suitable for the model.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset

from .layers import build_full_mask


class DMFMDataset(Dataset):
    """Holds per-timestep cross-sectional panels for DMFM training.
    
    Memory-efficient version: masks are stored once (N, N) and expanded per-sample.
    """

    def __init__(
        self,
        features: torch.Tensor,                 # (T, N, F)
        industry_mask: torch.Tensor,            # (N, N) - shared across time
        forward_returns: Dict[int, torch.Tensor],  # horizon -> (T, N)
        universe_mask: torch.Tensor | None = None,  # optional (N, N)
    ) -> None:
        assert features.dim() == 3, "features must be (T, N, F)"
        self.features = features.float()
        
        # Store masks as (N, N) to save memory - will expand per batch
        if industry_mask.dim() == 3:
            # If passed as (T, N, N), take first slice
            self.industry_mask = industry_mask[0].bool()
        else:
            self.industry_mask = industry_mask.bool()
        
        self.forward_returns = {str(k): v.float() for k, v in forward_returns.items()}
        
        T, N, _ = features.shape
        if universe_mask is None:
            self.universe_mask = build_full_mask(N, device=features.device).bool()
        elif universe_mask.dim() == 3:
            self.universe_mask = universe_mask[0].bool()
        else:
            self.universe_mask = universe_mask.bool()
        
        self.length = features.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "industry_mask": self.industry_mask,  # Same for all timesteps
            "universe_mask": self.universe_mask,  # Same for all timesteps
            "forward_returns": {k: v[idx] for k, v in self.forward_returns.items()},
        }


__all__ = ["DMFMDataset"]
