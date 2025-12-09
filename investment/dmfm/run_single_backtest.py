"""Single period backtest without splitting into early/middle/recent.

Train on first 60%, validate on next 20%, test on final 20%.

Usage:
    python -m dmfm.run_single_backtest --data-npz ./dmfm_data_clean.npz --output-dir ./results_single
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from dmfm.backtest import (
    BacktestConfig,
    BacktestResult,
    compute_metrics,
    compute_portfolio_returns,
    evaluate_model,
    train_model,
)
from dmfm.baselines import get_baseline_model
from dmfm.config import ModelConfig
from dmfm.data import DMFMDataset
from dmfm.model import DMFM
from dmfm.train import collate, seed_all


MODELS_TO_TEST = [
    "dmfm",
    "linear",
    "mlp",
    "mlp_gat",
    "momentum",
    "reversal",
]


def load_data(npz_path: Path, device: torch.device):
    """Load preprocessed data."""
    
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    data = np.load(npz_path, allow_pickle=True)
    
    features_np = data["features"]
    industry_mask_np = data["industry_mask"]
    universe_mask_np = data["universe_mask"]
    
    returns_np = {
        3: data["forward_returns_3"],
        5: data["forward_returns_5"],
        10: data["forward_returns_10"],
        15: data["forward_returns_15"],
        20: data["forward_returns_20"],
    }
    
    dates = data["dates"]
    T, N, F = features_np.shape
    
    print(f"  Time steps: {T}")
    print(f"  Stocks: {N}")
    print(f"  Features: {F}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    features = torch.from_numpy(features_np).to(device)
    forward_returns = {h: torch.from_numpy(returns_np[h]).to(device) for h in [3, 5, 10, 15, 20]}
    industry_mask = torch.from_numpy(industry_mask_np)
    universe_mask = torch.from_numpy(universe_mask_np)
    
    return features, forward_returns, industry_mask, universe_mask, dates


def run_single_backtest(
    features: torch.Tensor,
    forward_returns: Dict[int, torch.Tensor],
    industry_mask: torch.Tensor,
    universe_mask: torch.Tensor,
    dates: np.ndarray,
    config: BacktestConfig,
    device: torch.device,
) -> Dict[str, BacktestResult]:
    """Run single train/test backtest."""
    
    T = features.shape[0]
    
    # Split: 60% train, 20% val, 20% test
    train_end = int(T * 0.6)
    val_end = int(T * 0.8)
    
    train_start = 0
    val_start = train_end
    test_start = val_end
    test_end = T
    
    print(f"\n{'='*60}")
    print("Single Period Backtest")
    print(f"{'='*60}")
    print(f"  Training:   day {train_start:4d} - {train_end-1:4d} ({train_end - train_start} days)")
    print(f"             {dates[train_start]} ~ {dates[train_end-1]}")
    print(f"  Validation: day {val_start:4d} - {val_end-1:4d} ({val_end - val_start} days)")
    print(f"             {dates[val_start]} ~ {dates[val_end-1]}")
    print(f"  Testing:    day {test_start:4d} - {test_end-1:4d} ({test_end - test_start} days)")
    print(f"             {dates[test_start]} ~ {dates[test_end-1]}")
    
    # Create dataset
    full_dataset = DMFMDataset(features, industry_mask, forward_returns, universe_mask)
    
    train_indices = list(range(train_start, train_end))
    val_indices = list(range(val_start, val_end))
    test_indices = list(range(test_start, test_end))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    F = features.shape[2]
    horizons = (3, 5, 10, 15, 20)
    model_cfg = ModelConfig(feature_dim=F, hidden_dim=config.hidden_dim, horizons=horizons)
    
    results: Dict[str, BacktestResult] = {}
    
    for model_name in MODELS_TO_TEST:
        print(f"\n  Training: {model_name}...", flush=True)
        
        seed_all(42)
        
        if model_name == "dmfm":
            model = DMFM(model_cfg).to(device)
        else:
            model = get_baseline_model(
                model_name,
                feature_dim=F,
                hidden_dim=config.hidden_dim,
                horizons=horizons,
            ).to(device)
        
        if model_name not in ["momentum", "reversal"]:
            model = train_model(model, train_loader, val_loader, config, model_cfg, device, model_name)
        
        factor_scores, fwd_returns, metrics = evaluate_model(model, test_loader, config, device)
        
        holding_period = config.eval_horizon
        portfolio_returns = compute_portfolio_returns(
            factor_scores, fwd_returns,
            config.top_k, config.bottom_k,
            holding_period=holding_period
        )
        perf_metrics = compute_metrics(portfolio_returns, holding_period=holding_period)
        
        result = BacktestResult(
            model_name=model_name,
            test_ic=metrics["test_ic"],
            test_icir=metrics["test_icir"],
            test_factor_return=metrics["test_factor_return"],
            total_return=perf_metrics["total_return"],
            annual_return=perf_metrics["annual_return"],
            annual_volatility=perf_metrics["annual_volatility"],
            sharpe_ratio=perf_metrics["sharpe_ratio"],
            max_drawdown=perf_metrics["max_drawdown"],
            win_rate=perf_metrics["win_rate"],
            daily_returns=portfolio_returns.tolist(),
            cumulative_returns=np.cumprod(1 + portfolio_returns).tolist(),
        )
        results[model_name] = result
        
        print(f"    -> IC={result.test_ic:.4f}, ICIR={result.test_icir:.4f}, "
              f"Sharpe={result.sharpe_ratio:.2f}, Return={result.total_return*100:.1f}%")
    
    return results, {
        "train": (train_start, train_end, str(dates[train_start]), str(dates[train_end-1])),
        "val": (val_start, val_end, str(dates[val_start]), str(dates[val_end-1])),
        "test": (test_start, test_end, str(dates[test_start]), str(dates[test_end-1])),
    }


def save_results(results: Dict[str, BacktestResult], split_info: Dict, config: BacktestConfig, output_dir: Path):
    """Save results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "experiment_timestamp": datetime.now().isoformat(),
        "experiment_type": "single_period_backtest",
        "config": {
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "top_k": config.top_k,
            "bottom_k": config.bottom_k,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "hidden_dim": config.hidden_dim,
            "eval_horizon": config.eval_horizon,
        },
        "split_info": split_info,
        "models": MODELS_TO_TEST,
        "results": {model: asdict(result) for model, result in results.items()},
    }
    
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    # Save cumulative returns CSV
    cum_df_data = {}
    max_len = max(len(r.cumulative_returns) for r in results.values())
    
    for model_name, result in results.items():
        cum_returns = result.cumulative_returns
        padded = cum_returns + [cum_returns[-1]] * (max_len - len(cum_returns))
        cum_df_data[model_name] = padded
    
    cum_df = pd.DataFrame(cum_df_data)
    cum_df.to_csv(output_dir / "cumulative_returns.csv", index=False)
    
    # Save summary table
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            "Model": model_name.upper(),
            "IC": result.test_ic,
            "ICIR": result.test_icir,
            "Sharpe": result.sharpe_ratio,
            "Return": result.total_return,
            "Annual Return": result.annual_return,
            "Annual Vol": result.annual_volatility,
            "MDD": result.max_drawdown,
            "Win Rate": result.win_rate,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    summary_df["Rank"] = range(1, len(summary_df) + 1)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    
    print(f"\n  Results saved to {output_dir}")


def print_summary(results: Dict[str, BacktestResult]):
    """Print summary table."""
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Rank':<6} {'Model':<12} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'Return':>10} {'MDD':>8} {'Win%':>8}")
    print("-" * 75)
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1].sharpe_ratio)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        print(f"{rank:<6} {model_name.upper():<12} "
              f"{result.test_ic:>8.4f} "
              f"{result.test_icir:>8.4f} "
              f"{result.sharpe_ratio:>8.2f} "
              f"{result.total_return*100:>9.1f}% "
              f"{result.max_drawdown*100:>7.1f}% "
              f"{result.win_rate*100:>7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", type=str, default="./dmfm_data_clean.npz")
    parser.add_argument("--output-dir", type=str, default="./results_single")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    features, forward_returns, industry_mask, universe_mask, dates = load_data(
        Path(args.data_npz), device
    )
    
    config = BacktestConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )
    
    results, split_info = run_single_backtest(
        features, forward_returns, industry_mask, universe_mask, dates,
        config, device
    )
    
    save_results(results, split_info, config, Path(args.output_dir))
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
