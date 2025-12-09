"""Complete experiment runner with result saving and plotting.

This script:
1. Runs multi-period backtest across 3 time periods
2. Saves detailed results to JSON
3. Generates comprehensive plots

Usage:
    python -m dmfm.run_experiment --data-dir ./kospi --output-dir ./results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
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


# ============================================================================
# Experiment Configuration
# ============================================================================

PERIODS = {
    "early": {
        "train_start_pct": 0.0,
        "train_end_pct": 0.5,
        "test_start_pct": 0.5,
        "test_end_pct": 0.7,
    },
    "middle": {
        "train_start_pct": 0.2,
        "train_end_pct": 0.6,
        "test_start_pct": 0.6,
        "test_end_pct": 0.8,
    },
    "recent": {
        "train_start_pct": 0.4,
        "train_end_pct": 0.8,
        "test_start_pct": 0.8,
        "test_end_pct": 1.0,
    },
}

MODELS_TO_TEST = [
    "dmfm",      # Our proposed model
    "linear",    # Linear baseline
    "mlp",       # MLP baseline
    "mlp_gat",   # MLP + attention baseline
    "momentum",  # Classic momentum factor
    "reversal",  # Classic reversal factor
]


# ============================================================================
# Helper Functions
# ============================================================================

def load_data(npz_path: Path, device: torch.device):
    """Load preprocessed KOSPI data from NPZ file.
    
    Note: Masks are kept on CPU and expanded per-batch to save GPU memory.
    """
    
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Load arrays
    features_np = data["features"]  # (T, N, F)
    industry_mask_np = data["industry_mask"]  # (N, N)
    
    # Universe mask is optional
    if "universe_mask" in data.files:
        universe_mask_np = data["universe_mask"]  # (N, N)
    else:
        # Create fully connected universe mask
        N = features_np.shape[1]
        universe_mask_np = np.ones((N, N), dtype=bool)
    
    # Forward returns - support both naming conventions
    if "forward_returns_3" in data.files:
        returns_np = {
            3: data["forward_returns_3"],
            5: data["forward_returns_5"],
            10: data["forward_returns_10"],
            15: data["forward_returns_15"],
            20: data["forward_returns_20"],
        }
    else:
        returns_np = {
            3: data["forward_return_3d"],
            5: data["forward_return_5d"],
            10: data["forward_return_10d"],
            15: data["forward_return_15d"],
            20: data["forward_return_20d"],
        }
    
    dates = data["dates"]
    
    # These may not exist in extended data
    if "symbols" in data.files:
        symbols = data["symbols"]
    else:
        symbols = np.array([f"stock_{i}" for i in range(features_np.shape[1])])
    
    if "feature_cols" in data.files:
        feature_cols = data["feature_cols"]
    else:
        feature_cols = np.array([f"feature_{i}" for i in range(features_np.shape[2])])
    
    T, N, F = features_np.shape
    
    print(f"  Time steps: {T}")
    print(f"  Stocks: {N}")
    print(f"  Features: {F}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Feature columns: {list(feature_cols)[:5]}... ({len(feature_cols)} total)")
    print()
    
    # Keep features and forward returns on GPU
    features = torch.from_numpy(features_np).to(device)
    forward_returns = {h: torch.from_numpy(returns_np[h]).to(device) for h in [3, 5, 10, 15, 20]}
    
    # Keep masks on CPU - they'll be expanded per-batch in the dataset
    industry_mask = torch.from_numpy(industry_mask_np)  # (N, N) on CPU
    universe_mask = torch.from_numpy(universe_mask_np)  # (N, N) on CPU
    
    return features, forward_returns, industry_mask, universe_mask, dates


def run_period_backtest(
    features: torch.Tensor,
    forward_returns: Dict[int, torch.Tensor],
    industry_masks: torch.Tensor,
    universe_masks: torch.Tensor,
    period_name: str,
    period_cfg: Dict,
    config: BacktestConfig,
    device: torch.device,
) -> Dict[str, BacktestResult]:
    """Run backtest for a single period."""
    
    T = features.shape[0]
    train_start = int(T * period_cfg["train_start_pct"])
    train_end = int(T * period_cfg["train_end_pct"])
    test_start = int(T * period_cfg["test_start_pct"])
    test_end = int(T * period_cfg["test_end_pct"])
    
    # Validation is last 20% of training
    val_size = int((train_end - train_start) * 0.2)
    val_start = train_end - val_size
    
    print(f"\n{'='*60}")
    print(f"Period: {period_name.upper()}")
    print(f"{'='*60}")
    print(f"  Training:   day {train_start:4d} - {val_start-1:4d} ({val_start - train_start} days)")
    print(f"  Validation: day {val_start:4d} - {train_end-1:4d} ({train_end - val_start} days)")
    print(f"  Testing:    day {test_start:4d} - {test_end-1:4d} ({test_end - test_start} days)")
    
    # Create dataset
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
    
    F = features.shape[2]
    horizons = (3, 5, 10, 15, 20)
    model_cfg = ModelConfig(feature_dim=F, hidden_dim=config.hidden_dim, horizons=horizons)
    
    results: Dict[str, BacktestResult] = {}
    
    for model_name in MODELS_TO_TEST:
        print(f"\n  Training: {model_name}...", flush=True)
        
        # Fixed seed for reproducibility
        seed_all(42)
        
        # Create model
        if model_name == "dmfm":
            model = DMFM(model_cfg).to(device)
        else:
            model = get_baseline_model(
                model_name,
                feature_dim=F,
                hidden_dim=config.hidden_dim,
                horizons=horizons,
            ).to(device)
        
        # Train (skip for rule-based models)
        if model_name not in ["momentum", "reversal"]:
            model = train_model(model, train_loader, val_loader, config, model_cfg, device, model_name)
        
        # Evaluate
        factor_scores, fwd_returns, metrics = evaluate_model(model, test_loader, config, device)
        
        # Portfolio returns with proper holding period
        holding_period = config.eval_horizon  # 5 days by default
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
    
    return results


def save_results(
    all_results: Dict[str, Dict[str, BacktestResult]],
    config: BacktestConfig,
    output_dir: Path,
):
    """Save all results to JSON."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    results_dict = {
        "experiment_timestamp": datetime.now().isoformat(),
        "config": {
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "top_k": config.top_k,
            "bottom_k": config.bottom_k,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "hidden_dim": config.hidden_dim,
            "eval_horizon": config.eval_horizon,
        },
        "periods": PERIODS,
        "models": MODELS_TO_TEST,
        "results": {},
    }
    
    for period_name, period_results in all_results.items():
        results_dict["results"][period_name] = {}
        for model_name, result in period_results.items():
            results_dict["results"][period_name][model_name] = asdict(result)
    
    # Save main results
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    # Save cumulative returns CSV for each period
    for period_name, period_results in all_results.items():
        cum_df_data = {}
        max_len = max(len(r.cumulative_returns) for r in period_results.values())
        
        for model_name, result in period_results.items():
            cum_returns = result.cumulative_returns
            # Pad to max length
            padded = cum_returns + [cum_returns[-1]] * (max_len - len(cum_returns))
            cum_df_data[model_name] = padded
        
        cum_df = pd.DataFrame(cum_df_data)
        cum_df.to_csv(output_dir / f"cumulative_returns_{period_name}.csv", index=False)
    
    print(f"\n  Results saved to {output_dir}")


def compute_summary_table(all_results: Dict[str, Dict[str, BacktestResult]]) -> pd.DataFrame:
    """Compute summary table averaging across periods."""
    
    summary_data = []
    
    for model_name in MODELS_TO_TEST:
        model_results = [
            all_results[period][model_name]
            for period in all_results.keys()
        ]
        
        avg_ic = np.mean([r.test_ic for r in model_results])
        avg_icir = np.mean([r.test_icir for r in model_results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in model_results])
        avg_return = np.mean([r.total_return for r in model_results])
        avg_mdd = np.mean([r.max_drawdown for r in model_results])
        avg_winrate = np.mean([r.win_rate for r in model_results])
        
        std_ic = np.std([r.test_ic for r in model_results])
        std_sharpe = np.std([r.sharpe_ratio for r in model_results])
        
        summary_data.append({
            "Model": model_name.upper(),
            "IC (avg)": avg_ic,
            "IC (std)": std_ic,
            "ICIR (avg)": avg_icir,
            "Sharpe (avg)": avg_sharpe,
            "Sharpe (std)": std_sharpe,
            "Return (avg)": avg_return,
            "MDD (avg)": avg_mdd,
            "Win Rate (avg)": avg_winrate,
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values("Sharpe (avg)", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run DMFM experiment")
    parser.add_argument("--data-npz", type=str, default="./dmfm_data.npz", help="Preprocessed data file")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--top-k", type=int, default=10, help="Top K for long")
    parser.add_argument("--bottom-k", type=int, default=10, help="Bottom K for short")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_npz = Path(args.data_npz)
    output_dir = Path(args.output_dir)
    
    features, forward_returns, industry_masks, universe_masks, dates = load_data(
        data_npz, device
    )
    
    # Configuration
    config = BacktestConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )
    
    # Run experiments
    all_results: Dict[str, Dict[str, BacktestResult]] = {}
    
    for period_name, period_cfg in PERIODS.items():
        results = run_period_backtest(
            features, forward_returns, industry_masks, universe_masks,
            period_name, period_cfg, config, device
        )
        all_results[period_name] = results
    
    # Save results
    save_results(all_results, config, output_dir)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY (averaged across all periods)")
    print("=" * 80)
    
    summary_df = compute_summary_table(all_results)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
