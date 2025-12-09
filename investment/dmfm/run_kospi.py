"""Quick DMFM experiment on KOSPI CSVs located in ./investment/kospi.

Assumptions
- Each CSV is named <symbol>.csv and contains columns: date, open, high, low, close, volume.
- No industry classification provided; we use a fully connected universe mask (and self-loops).
- Forward returns are computed from close prices: r_{t→t+h} = P_{t+h}/P_t - 1.

This script is a minimal, end-to-end demo using the DMFM components.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from factor_builder import compute_factor_panel
from dmfm.config import LossConfig, ModelConfig, OptimConfig, TrainingConfig
from dmfm.data import DMFMDataset
from dmfm.losses import DMFMLoss
from dmfm.model import DMFM
from dmfm.train import collate, seed_all, train_one_epoch


def load_symbol_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # normalize column names to lower-case
    df.columns = [c.lower() for c in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["symbol"] = path.stem
    return df


def build_panel(data_dir: Path, limit_symbols: int, horizons: List[int], max_dates: int | None, industry_csv: Path | None = None) -> tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray, List[str]]:
    paths = sorted([p for p in data_dir.glob("*.csv")])
    if limit_symbols:
        paths = paths[:limit_symbols]
    if not paths:
        raise RuntimeError("No CSV files found in data_dir")

    dfs = [load_symbol_df(p) for p in paths]
    # intersect dates across symbols to ensure aligned panel
    common_dates = set(dfs[0]["date"].unique())
    for d in dfs[1:]:
        common_dates &= set(d["date"].unique())
    common_dates = sorted(common_dates)
    if max_dates:
        common_dates = common_dates[-max_dates:]
    if len(common_dates) == 0:
        raise RuntimeError("No common dates across symbols")

    aligned = []
    for df in dfs:
        df = df[df["date"].isin(common_dates)].copy()
        aligned.append(df)

    stacked = pd.concat(aligned, ignore_index=True)
    stacked = compute_factor_panel(stacked)

    # Fill missing numeric values: per symbol forward/backward, then per date median
    num_cols = stacked.select_dtypes(include=[np.number]).columns
    stacked[num_cols] = stacked.groupby("symbol")[num_cols].transform(lambda x: x.ffill().bfill())
    stacked[num_cols] = stacked.groupby("date")[num_cols].transform(lambda x: x.fillna(x.median()))

    # Forward returns per horizon
    forward_returns: Dict[int, np.ndarray] = {}
    symbols = sorted({s for s in stacked.symbol.unique()})
    dates = common_dates
    # pivot by date/symbol to ensure consistent ordering
    stacked = stacked.set_index(["date", "symbol"]).sort_index()

    # build feature matrix
    feature_cols = [c for c in stacked.columns if c not in ["open", "high", "low", "close", "volume", "shares_outstanding"]]
    X_list = []
    close_list = []
    for date in dates:
        feat = stacked.xs(date).loc[symbols, feature_cols].to_numpy(dtype=np.float32)
        cls = stacked.xs(date).loc[symbols, "close"].to_numpy(dtype=np.float32)
        X_list.append(feat)
        close_list.append(cls)

    X = np.stack(X_list, axis=0)  # (T, N, F)
    close_panel = np.stack(close_list, axis=0)  # (T, N)

    for h in horizons:
        # forward return: P_{t+h}/P_t - 1; last h rows will be nan and removed
        fwd = close_panel[h:, :] / close_panel[:-h, :] - 1.0
        forward_returns[h] = fwd

    # trim features to align with forward horizon (drop last max_h rows)
    max_h = max(horizons)
    X = X[:-max_h, :, :]
    T_final = X.shape[0]
    # ensure all forward arrays same T
    for h in horizons:
        forward_returns[h] = forward_returns[h][: T_final, :]

    # Build industry mask from CSV if provided
    N = len(symbols)
    if industry_csv and industry_csv.exists():
        ind_df = pd.read_csv(industry_csv)
        ind_df = ind_df.set_index("symbol")
        # Map symbols to industry codes
        symbol_to_industry = {}
        for sym in symbols:
            if sym in ind_df.index:
                symbol_to_industry[sym] = ind_df.loc[sym, "industry"]
            else:
                symbol_to_industry[sym] = "Unknown"
        
        # Build static industry mask (same industry -> True)
        industry_mask_static = np.zeros((N, N), dtype=bool)
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if symbol_to_industry[sym_i] == symbol_to_industry[sym_j]:
                    industry_mask_static[i, j] = True
        
        # Expand to time dimension
        industry_masks = np.tile(industry_mask_static[None, :, :], (T_final, 1, 1))
        
        # Count industries
        industries = list(set(symbol_to_industry.values()))
        print(f"Using {len(industries)} industries from {industry_csv}")
    else:
        # Fallback: fully connected (no industry distinction)
        industry_masks = np.ones((T_final, N, N), dtype=bool)
        print("No industry CSV provided, using fully connected industry mask")

    # final NaN/inf guard
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    for h in horizons:
        forward_returns[h] = np.nan_to_num(forward_returns[h], nan=0.0, posinf=0.0, neginf=0.0)

    return X, forward_returns, industry_masks, symbols


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    seed_all(args.seed)

    horizons = args.horizons
    industry_csv = Path(args.industry_csv) if args.industry_csv else None
    features_np, forward_returns_np, industry_masks_np, symbols = build_panel(
        Path(args.data_dir), args.limit_symbols, horizons, args.max_dates, industry_csv
    )
    T, N, F = features_np.shape
    print(f"Loaded panel: T={T}, N={N}, F={F} (symbols limited to {args.limit_symbols})")

    features = torch.from_numpy(features_np)
    forward_returns = {k: torch.from_numpy(v) for k, v in forward_returns_np.items()}

    # Industry mask from CSV, universe mask is fully connected
    industry_masks = torch.from_numpy(industry_masks_np)
    universe_masks = torch.ones(T, N, N, dtype=torch.bool)

    dataset = DMFMDataset(features, industry_masks, forward_returns, universe_masks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model_cfg = ModelConfig(feature_dim=F, hidden_dim=args.hidden_dim, horizons=horizons)
    loss_cfg = LossConfig(attn_mse_weight=args.attn_mse_weight, factor_return_weight=args.factor_return_weight, icir_weight=args.icir_weight)
    optim_cfg = OptimConfig(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), grad_clip=args.grad_clip)
    train_cfg = TrainingConfig(model=model_cfg, loss=loss_cfg, optim=optim_cfg, batch_size=args.batch_size, max_epochs=args.epochs, device=args.device)

    model = DMFM(model_cfg).to(device)
    loss_fn = DMFMLoss(model_cfg, loss_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, betas=optim_cfg.betas)

    for epoch in range(train_cfg.max_epochs):
        avg_loss, metrics = train_one_epoch(model, loss_fn, loader, optimizer, device, optim_cfg.grad_clip)
        print(
            f"Epoch {epoch+1}/{train_cfg.max_epochs} | loss={avg_loss:.4f} | "
            f"b3={metrics.get('b_3', 0):.4f} | ic3={metrics.get('ic_3', 0):.4f}"
        )

    if args.save_path:
        torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.__dict__}, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMFM on KOSPI OHLCV CSVs")
    parser.add_argument(
        "--data-dir",
        default="./kospi",
        help="Directory containing <symbol>.csv files (relative paths are resolved from current working directory)",
    )
    parser.add_argument(
        "--industry-csv",
        default="./kospi_industry.csv",
        help="CSV file with industry classification (columns: symbol, industry)",
    )
    parser.add_argument("--limit-symbols", type=int, default=50, help="Number of symbols to use (to keep demo fast)")
    parser.add_argument("--max-dates", type=int, default=500, help="Use most recent N dates to limit size")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 10, 15, 20])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--attn-mse-weight", type=float, default=1.0)
    parser.add_argument("--factor-return-weight", type=float, default=1.0)
    parser.add_argument("--icir-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="")
    args = parser.parse_args()
    main(args)
