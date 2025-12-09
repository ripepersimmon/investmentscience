"""Configuration objects for Deep Multi-Factor Model (DMFM).

Derived from the paper "Factor Investing with a Deep Multi-Factor Model"
(arXiv:2210.12462). Defaults aim to mirror the described architecture while
remaining lightweight to train with custom data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass
class ModelConfig:
    feature_dim: int                  # m: number of input features per stock
    hidden_dim: int = 128             # m1: hidden context dimension
    gat_heads: int = 4                # number of attention heads in GAT blocks
    gat_dropout: float = 0.1
    leaky_relu_slope: float = 0.1
    horizons: Sequence[int] = field(default_factory=lambda: (3, 5, 10, 15, 20))


@dataclass
class LossConfig:
    attn_mse_weight: float = 1.0      # weight on ||f_k - f_hat_k||_2
    factor_return_weight: float = 1.0 # encourages higher factor return b_k
    icir_weight: float = 1.0          # encourages higher IC / IC volatility
    eps: float = 1e-6


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float | None = 1.0


@dataclass
class TrainingConfig:
    model: ModelConfig
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    batch_size: int = 16
    max_epochs: int = 50
    device: str = "cuda"
    log_interval: int = 20
    ckpt_dir: Path | None = None
    seed: int = 42


__all__ = [
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "TrainingConfig",
]
