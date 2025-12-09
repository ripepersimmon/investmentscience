"""Deep Multi-Factor Model (DMFM) package."""

from .config import ModelConfig, LossConfig, OptimConfig, TrainingConfig
from .data import DMFMDataset
from .losses import DMFMLoss, compute_ic, compute_icir, factor_return
from .model import DMFM

__all__ = [
    "DMFM",
    "DMFMDataset",
    "DMFMLoss",
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "TrainingConfig",
    "compute_ic",
    "compute_icir",
    "factor_return",
]
