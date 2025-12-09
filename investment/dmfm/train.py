"""Minimal training loop for the Deep Multi-Factor Model (DMFM).

This script purposefully avoids external dependencies and any workspace
coupling beyond PyTorch. Plug your own tensors into `DMFMDataset`.

Usage (sanity check with synthetic data):

```bash
python -m dmfm.train --device cpu
```

Replace the synthetic dataset generation with real tensors that match
what the paper describes: cross-sectional features per date, an
industry adjacency per date, and forward returns for horizons.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import LossConfig, ModelConfig, OptimConfig, TrainingConfig
from .data import DMFMDataset
from .losses import DMFMLoss
from .model import DMFM


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_synthetic_dataset(T: int = 32, N: int = 50, F: int = 24, horizons=(3, 5, 10)) -> DMFMDataset:
    """Creates a toy dataset to validate the pipeline end-to-end."""
    features = torch.randn(T, N, F)
    # Random block-diagonal industries
    industry_masks = torch.zeros(T, N, N, dtype=torch.bool)
    for t in range(T):
        # 5 industries of roughly equal size
        size = max(1, N // 5)
        for g in range(5):
            start = g * size
            end = min(N, start + size)
            industry_masks[t, start:end, start:end] = True
        # ensure self-loops
        industry_masks[t].fill_diagonal_(True)
    forward_returns = {h: torch.randn(T, N) * 0.01 for h in horizons}
    return DMFMDataset(features, industry_masks, forward_returns)


def collate(batch) -> Dict[str, torch.Tensor]:
    features = torch.stack([b["features"] for b in batch], dim=0)
    industry_mask = torch.stack([b["industry_mask"] for b in batch], dim=0)
    universe_mask = torch.stack([b["universe_mask"] for b in batch], dim=0)
    forward_returns: Dict[str, torch.Tensor] = {}
    for key in batch[0]["forward_returns"]:
        forward_returns[key] = torch.stack([b["forward_returns"][key] for b in batch], dim=0)
    return {
        "features": features,
        "industry_mask": industry_mask,
        "universe_mask": universe_mask,
        "forward_returns": forward_returns,
    }


def train_one_epoch(
    model: DMFM,
    loss_fn: DMFMLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    running = 0.0
    metrics_sum: Dict[str, float] = {}
    steps = 0

    for batch in loader:
        steps += 1
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else {kk: vv.to(device) for kk, vv in v.items()}) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch["features"], batch["industry_mask"], batch["universe_mask"])
        loss, metrics = loss_fn(outputs, batch["forward_returns"])
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss.item()
        for k, v in metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v

    avg_loss = running / max(1, steps)
    avg_metrics = {k: v / max(1, steps) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


def main(args: argparse.Namespace) -> None:
    seed_all(args.seed)

    horizons = tuple(args.horizons)
    dataset = build_synthetic_dataset(T=args.steps, N=args.stocks, F=args.features, horizons=horizons)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model_cfg = ModelConfig(feature_dim=args.features, hidden_dim=args.hidden_dim, horizons=horizons)
    loss_cfg = LossConfig(attn_mse_weight=args.attn_mse_weight, factor_return_weight=args.factor_return_weight, icir_weight=args.icir_weight)
    optim_cfg = OptimConfig(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), grad_clip=args.grad_clip)
    train_cfg = TrainingConfig(model=model_cfg, loss=loss_cfg, optim=optim_cfg, batch_size=args.batch_size, max_epochs=args.epochs, device=args.device)

    device = torch.device(train_cfg.device)
    model = DMFM(model_cfg).to(device)
    loss_fn = DMFMLoss(model_cfg, loss_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, betas=optim_cfg.betas)

    for epoch in range(train_cfg.max_epochs):
        avg_loss, metrics = train_one_epoch(model, loss_fn, loader, optimizer, device, optim_cfg.grad_clip)
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{train_cfg.max_epochs} | loss={avg_loss:.4f} | b={metrics.get('b_3', 0):.4f} | ic={metrics.get('ic_3', 0):.4f}")

    if args.save_path:
        torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.__dict__}, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Deep Multi-Factor Model (DMFM)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--features", type=int, default=24, help="input feature dimension m")
    parser.add_argument("--stocks", type=int, default=50, help="number of stocks N in synthetic data")
    parser.add_argument("--steps", type=int, default=32, help="time steps in synthetic data")
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 10, 15, 20])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--attn-mse-weight", type=float, default=1.0)
    parser.add_argument("--factor-return-weight", type=float, default=1.0)
    parser.add_argument("--icir-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="")
    args = parser.parse_args()
    main(args)
