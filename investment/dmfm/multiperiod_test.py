"""Multi-period robustness test for DMFM and baselines.

Tests models across different time periods to check consistency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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


def run_period_test(
    features: torch.Tensor,
    forward_returns: Dict[int, torch.Tensor],
    industry_masks: torch.Tensor,
    universe_masks: torch.Tensor,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    config: BacktestConfig,
    models_to_test: List[str],
    device: torch.device,
    period_name: str,
) -> Dict[str, BacktestResult]:
    """Run backtest for a specific period."""
    
    T, N, F = features.shape
    
    # Validation is last 20% of training
    val_size = int((train_end - train_start) * 0.2)
    val_start = train_end - val_size
    
    print(f"\n{'='*60}")
    print(f"Period: {period_name}")
    print(f"{'='*60}")
    print(f"  Training:   {train_start} - {val_start-1} ({val_start - train_start} days)")
    print(f"  Validation: {val_start} - {train_end-1} ({train_end - val_start} days)")
    print(f"  Testing:    {test_start} - {test_end-1} ({test_end - test_start} days)")
    
    full_dataset = DMFMDataset(features, industry_masks, forward_returns, universe_masks)
    
    train_indices = list(range(train_start, val_start))
    val_indices = list(range(val_start, train_end))
    test_indices = list(range(test_start, test_end))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    horizons = [3, 5, 10, 15, 20]
    model_cfg = ModelConfig(feature_dim=F, hidden_dim=config.hidden_dim, horizons=tuple(horizons))
    
    results: Dict[str, BacktestResult] = {}
    
    for model_name in models_to_test:
        print(f"\n  Training: {model_name}...", end=" ", flush=True)
        
        seed_all(42)
        
        if model_name == "dmfm":
            model = DMFM(model_cfg).to(device)
        else:
            model = get_baseline_model(
                model_name,
                feature_dim=F,
                hidden_dim=config.hidden_dim,
                horizons=tuple(horizons)
            ).to(device)
        
        if model_name not in ["momentum", "reversal"]:
            model = train_model(model, train_loader, val_loader, config, model_cfg, device, model_name)
        
        factor_scores, fwd_returns, metrics = evaluate_model(model, test_loader, config, device)
        portfolio_returns = compute_portfolio_returns(factor_scores, fwd_returns, config.top_k, config.bottom_k)
        perf_metrics = compute_metrics(portfolio_returns)
        
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
        
        print(f"IC={result.test_ic:.4f}, Sharpe={result.sharpe_ratio:.2f}, Return={result.total_return*100:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-period robustness test")
    parser.add_argument("--data-dir", type=str, default="./kospi")
    parser.add_argument("--industry-csv", type=str, default="./kospi_industry.csv")
    parser.add_argument("--output-dir", type=str, default="./backtest_results")
    parser.add_argument("--limit-symbols", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dmfm", "mlp_gat", "mlp", "linear", "momentum", "reversal"],
    )
    args = parser.parse_args()
    
    from dmfm.run_kospi import build_panel
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load all data
    horizons = [3, 5, 10, 15, 20]
    features_np, forward_returns_np, industry_masks_np, symbols = build_panel(
        Path(args.data_dir), args.limit_symbols, horizons, max_dates=None,
        industry_csv=Path(args.industry_csv) if args.industry_csv else None
    )
    
    T, N, F = features_np.shape
    print(f"\nTotal data: {T} periods, {N} stocks, {F} features")
    
    features = torch.from_numpy(features_np)
    forward_returns = {k: torch.from_numpy(v) for k, v in forward_returns_np.items()}
    industry_masks = torch.from_numpy(industry_masks_np)
    universe_masks = torch.ones(T, N, N, dtype=torch.bool)
    
    config = BacktestConfig(epochs=args.epochs)
    
    # Define multiple test periods (rolling windows)
    # Period 1: First 60% train, next 20% test
    # Period 2: 20-80% train, last 20% test  
    # Period 3: 40-80% train, 80-100% test (same as original but shifted)
    
    periods = [
        {
            "name": "Period 1 (Early)",
            "train_start": 0,
            "train_end": int(T * 0.5),
            "test_start": int(T * 0.5),
            "test_end": int(T * 0.7),
        },
        {
            "name": "Period 2 (Middle)",
            "train_start": int(T * 0.2),
            "train_end": int(T * 0.6),
            "test_start": int(T * 0.6),
            "test_end": int(T * 0.8),
        },
        {
            "name": "Period 3 (Recent)",
            "train_start": int(T * 0.4),
            "train_end": int(T * 0.8),
            "test_start": int(T * 0.8),
            "test_end": T,
        },
    ]
    
    all_results = {}
    
    for period in periods:
        results = run_period_test(
            features, forward_returns, industry_masks, universe_masks,
            period["train_start"], period["train_end"],
            period["test_start"], period["test_end"],
            config, args.models, device, period["name"]
        )
        all_results[period["name"]] = results
    
    # Print summary across all periods
    print(f"\n{'='*100}")
    print("MULTI-PERIOD ROBUSTNESS TEST SUMMARY")
    print(f"{'='*100}")
    
    # Header
    print(f"\n{'Model':<12}", end="")
    for period in periods:
        name = period["name"].split("(")[1].replace(")", "")
        print(f" | {name:^25}", end="")
    print(" | {'Average':^25}")
    
    print("-" * 100)
    
    # Metrics to show
    for metric_name, metric_key in [("IC", "test_ic"), ("Sharpe", "sharpe_ratio"), ("Return%", "total_return")]:
        print(f"\n{metric_name}")
        for model in args.models:
            print(f"  {model:<10}", end="")
            values = []
            for period in periods:
                r = all_results[period["name"]][model]
                val = getattr(r, metric_key)
                if metric_key == "total_return":
                    val *= 100
                values.append(val)
                print(f" | {val:>10.2f}", end="              ")
            avg = np.mean(values)
            print(f" | {avg:>10.2f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "periods": [p["name"] for p in periods],
        "models": args.models,
        "results": {
            period["name"]: {
                model: {
                    "ic": all_results[period["name"]][model].test_ic,
                    "icir": all_results[period["name"]][model].test_icir,
                    "sharpe": all_results[period["name"]][model].sharpe_ratio,
                    "return": all_results[period["name"]][model].total_return,
                    "max_dd": all_results[period["name"]][model].max_drawdown,
                    "win_rate": all_results[period["name"]][model].win_rate,
                }
                for model in args.models
            }
            for period in periods
        },
        "average": {
            model: {
                "ic": np.mean([all_results[p["name"]][model].test_ic for p in periods]),
                "sharpe": np.mean([all_results[p["name"]][model].sharpe_ratio for p in periods]),
                "return": np.mean([all_results[p["name"]][model].total_return for p in periods]),
            }
            for model in args.models
        }
    }
    
    with open(output_dir / "multiperiod_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final ranking
    print(f"\n{'='*60}")
    print("FINAL RANKING (by average Sharpe ratio)")
    print(f"{'='*60}")
    
    avg_sharpes = [(m, summary["average"][m]["sharpe"]) for m in args.models]
    avg_sharpes.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (model, sharpe) in enumerate(avg_sharpes, 1):
        avg_ic = summary["average"][model]["ic"]
        avg_ret = summary["average"][model]["return"] * 100
        print(f"  {rank}. {model:<12} Sharpe={sharpe:>6.2f}  IC={avg_ic:>6.4f}  Return={avg_ret:>6.1f}%")
    
    print(f"\nResults saved to {output_dir / 'multiperiod_results.json'}")


if __name__ == "__main__":
    main()
