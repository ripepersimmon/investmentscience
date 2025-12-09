"""Losses and metrics for the Deep Multi-Factor Model."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from .config import LossConfig, ModelConfig


def compute_ic(factor: torch.Tensor, returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Information Coefficient (Pearson correlation) per sample.
    
    Paper definition: IC measures the cross-sectional correlation between
    the deep factor f_k^t and the future return r_{t+k} at each time t.

    Args:
        factor: (B, N) deep factor values
        returns: (B, N) forward returns
    Returns:
        ic: (B,) correlation over the cross-section of stocks per time step.
    """
    # mask NaNs if provided
    mask = torch.isfinite(factor) & torch.isfinite(returns)
    factor = torch.where(mask, factor, torch.zeros_like(factor))
    returns = torch.where(mask, returns, torch.zeros_like(returns))
    mask_count = mask.sum(dim=-1).clamp(min=1).float()

    # Cross-sectional mean (per time step)
    f_mean = (factor * mask).sum(dim=-1, keepdim=True) / mask_count.unsqueeze(-1)
    r_mean = (returns * mask).sum(dim=-1, keepdim=True) / mask_count.unsqueeze(-1)
    
    f_center = (factor - f_mean) * mask
    r_center = (returns - r_mean) * mask
    
    num = (f_center * r_center).sum(dim=-1)
    f_std = torch.sqrt((f_center.pow(2)).sum(dim=-1) + eps)
    r_std = torch.sqrt((r_center.pow(2)).sum(dim=-1) + eps)
    denom = f_std * r_std + eps
    
    ic = num / denom
    return ic


def compute_icir(ic: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Information Coefficient Information Ratio (ICIR).
    
    Paper definition: ICIR = mean(IC) / std(IC)
    Measures the stability of the predictive power of the deep factor.
    Higher ICIR indicates more stable factor performance.
    
    Args:
        ic: (B,) or (T,) IC values across time steps
    Returns:
        icir: scalar ICIR value
    """
    if ic.numel() < 2:
        # Need at least 2 samples for std calculation
        return ic.mean()
    mean_ic = ic.mean()
    std_ic = ic.std(unbiased=True) + eps
    return mean_ic / std_ic


def factor_return(factor: torch.Tensor, returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cross-sectional regression coefficient b_k^t for each time step.
    
    Paper definition: The factor return b_k^t is computed via OLS:
        r_{t+k} = b_k^t * f_k^t + epsilon
        b_k^t = (f^T * r) / (f^T * f)
    
    This represents the return attributable to the deep factor at each time t.
    
    Args:
        factor: (B, N) deep factor values
        returns: (B, N) forward returns
    Returns:
        b: (B,) factor return coefficients per time step
    """
    mask = torch.isfinite(factor) & torch.isfinite(returns)
    factor = torch.where(mask, factor, torch.zeros_like(factor))
    returns = torch.where(mask, returns, torch.zeros_like(returns))
    
    # Cross-sectional regression: b = (f^T r) / (f^T f)
    num = (factor * returns * mask).sum(dim=-1)
    denom = (factor.pow(2) * mask).sum(dim=-1) + eps
    return num / denom


class DMFMLoss(nn.Module):
    """Deep Multi-Factor Model Loss Function.
    
    Paper Loss Function (Eq. 13):
        L = (1/|K||T|) * sum_{k in K} sum_{t in T} (d_k^t - b_k^t - c_k)
    
    Where:
        - d_k^t = ||f_k^t - f_hat_k^t||_2  (attention reconstruction error)
        - b_k^t = factor return (cross-sectional regression coefficient)
        - c_k = ICIR (information coefficient information ratio)
    
    The loss minimizes reconstruction error while maximizing factor return and ICIR.
    """
    
    def __init__(self, model_cfg: ModelConfig, loss_cfg: LossConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.mse = nn.MSELoss(reduction='mean')

    def forward(
        self,
        outputs: Mapping[str, Dict[str, torch.Tensor]],
        forward_returns: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute the DMFM loss.
        
        Args:
            outputs: Model outputs containing 'factors', 'recon', 'attn'
            forward_returns: Dict mapping horizon to forward returns tensor
            
        Returns:
            total_loss: Scalar loss tensor
            metrics: Dictionary of metric values for logging
        """
        total = torch.zeros(1, device=next(iter(outputs["factors"].values())).device)
        metrics: Dict[str, float] = {}

        for h in self.model_cfg.horizons:
            key = str(h)
            f = outputs["factors"][key]       # (B, N) deep factor
            f_hat = outputs["recon"][key]     # (B, N) attention estimate of deep factor
            r = forward_returns[key]          # (B, N) forward returns

            # d_k^t: L2 norm of (f - f_hat), paper uses ||f_k^t - f_hat_k^t||_2
            attn_mse = self.mse(f, f_hat)
            
            # b_k^t: Factor return (cross-sectional regression coefficient)
            b = factor_return(f, r, eps=self.loss_cfg.eps)  # (B,)
            b_mean = b.mean()  # Mean over batch (time steps)
            
            # c_k: ICIR (IC / std(IC)) - stability measure
            ic = compute_ic(f, r, eps=self.loss_cfg.eps)  # (B,)
            icir = compute_icir(ic, eps=self.loss_cfg.eps)  # scalar

            # Paper Eq. (13): L = d_k - b_k - c_k
            # We want to minimize d_k (reconstruction error)
            # We want to maximize b_k (factor return) -> subtract
            # We want to maximize c_k (ICIR/stability) -> subtract
            loss_h = (
                self.loss_cfg.attn_mse_weight * attn_mse
                - self.loss_cfg.factor_return_weight * b_mean
                - self.loss_cfg.icir_weight * icir
            )
            total = total + loss_h

            metrics[f"loss_{key}"] = loss_h.item()
            metrics[f"attn_mse_{key}"] = attn_mse.item()
            metrics[f"b_{key}"] = b_mean.item()
            metrics[f"ic_{key}"] = ic.mean().item()
            metrics[f"icir_{key}"] = icir.item()

        total = total / len(self.model_cfg.horizons)
        metrics["loss"] = total.item()
        return total, metrics


__all__ = ["DMFMLoss", "compute_ic", "compute_icir", "factor_return"]
