"""Run DMFM experiment with realistic transaction costs.

This script runs the full experiment pipeline with transaction costs
applied during backtesting.

Usage:
    python -m dmfm.run_realistic_experiment --data-npz ./dmfm_data_extended.npz --output-dir ./results_realistic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import models from dmfm package
from dmfm.model import DMFM
from dmfm.baselines import LinearFactorModel, MLPFactorModel, MLPGATModel, MomentumModel, MeanReversionModel


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Data split
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Portfolio
    top_k: int = 10
    bottom_k: int = 10
    holding_period: int = 5  # days
    
    # Training
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-4
    hidden_dim: int = 128
    
    # Evaluation
    eval_horizon: int = 5  # Use 5-day forward returns
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% one-way
    slippage_rate: float = 0.002    # 0.2% market impact
    short_borrow_rate: float = 0.02 # 2% annual
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def compute_backtest_with_costs(
    predictions: np.ndarray,
    actuals: np.ndarray,
    config: ExperimentConfig,
) -> Dict:
    """Compute backtest metrics with transaction costs.
    
    Args:
        predictions: (T, N) predicted scores
        actuals: (T, N) actual forward returns
        config: Experiment config with cost parameters
        
    Returns:
        Dictionary of metrics
    """
    T, N = predictions.shape
    n_periods = T // config.holding_period
    
    period_returns_gross = []
    period_returns_net = []
    total_tx_cost = 0.0
    total_short_cost = 0.0
    
    prev_long = set()
    prev_short = set()
    
    for p in range(n_periods):
        t = p * config.holding_period
        
        # Get predictions and actuals at rebalancing point
        pred = predictions[t]
        actual = actuals[t]
        
        # Valid mask
        valid_mask = np.isfinite(pred) & np.isfinite(actual)
        if valid_mask.sum() < config.top_k + config.bottom_k:
            continue
        
        valid_idx = np.where(valid_mask)[0]
        valid_pred = pred[valid_mask]
        valid_actual = actual[valid_mask]
        
        # Rank and select
        sorted_idx = np.argsort(valid_pred)[::-1]
        top_k_idx = sorted_idx[:config.top_k]
        bottom_k_idx = sorted_idx[-config.bottom_k:]
        
        # Portfolio returns
        long_ret = valid_actual[top_k_idx].mean()
        short_ret = valid_actual[bottom_k_idx].mean()
        
        # Gross L/S return
        gross_ret = long_ret - short_ret
        period_returns_gross.append(gross_ret)
        
        # Calculate turnover
        new_long = set(valid_idx[top_k_idx])
        new_short = set(valid_idx[bottom_k_idx])
        
        long_turnover = len(new_long - prev_long) + len(prev_long - new_long)
        short_turnover = len(new_short - prev_short) + len(prev_short - new_short)
        total_turnover = long_turnover + short_turnover
        
        # Transaction cost (proportional to turnover)
        # turnover / (2 * n_positions) = fraction of portfolio changed
        n_positions = config.top_k + config.bottom_k
        turnover_frac = total_turnover / (2 * n_positions)
        tx_cost = turnover_frac * 2 * (config.commission_rate + config.slippage_rate)
        
        # Short borrow cost (pro-rated)
        short_cost = config.short_borrow_rate * (config.holding_period / 252)
        
        # Net return
        net_ret = gross_ret - tx_cost - short_cost
        period_returns_net.append(net_ret)
        
        total_tx_cost += tx_cost
        total_short_cost += short_cost
        
        prev_long = new_long
        prev_short = new_short
    
    # Convert to arrays
    gross_rets = np.array(period_returns_gross)
    net_rets = np.array(period_returns_net)
    
    if len(gross_rets) == 0:
        return {
            "sharpe_gross": 0.0,
            "sharpe_net": 0.0,
            "cumulative_return_gross": 0.0,
            "cumulative_return_net": 0.0,
            "annual_return_gross": 0.0,
            "annual_return_net": 0.0,
            "annual_volatility": 0.0,
            "max_drawdown": 0.0,
            "total_tx_cost": 0.0,
            "total_short_cost": 0.0,
            "n_periods": 0,
            "period_returns_gross": [],
            "period_returns_net": [],
        }
    
    # Metrics
    periods_per_year = 252 / config.holding_period
    
    # Gross metrics
    sharpe_gross = gross_rets.mean() / gross_rets.std() * np.sqrt(periods_per_year) if gross_rets.std() > 0 else 0
    cum_gross = np.prod(1 + gross_rets) - 1
    
    # Net metrics
    sharpe_net = net_rets.mean() / net_rets.std() * np.sqrt(periods_per_year) if net_rets.std() > 0 else 0
    cum_net = np.prod(1 + net_rets) - 1
    
    n_years = len(net_rets) / periods_per_year
    annual_return_gross = (1 + cum_gross) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_return_net = (1 + cum_net) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = net_rets.std() * np.sqrt(periods_per_year)
    
    # Max drawdown
    cum_returns = np.cumprod(1 + net_rets)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "sharpe_gross": float(sharpe_gross),
        "sharpe_net": float(sharpe_net),
        "cumulative_return_gross": float(cum_gross),
        "cumulative_return_net": float(cum_net),
        "annual_return_gross": float(annual_return_gross),
        "annual_return_net": float(annual_return_net),
        "annual_volatility": float(annual_vol),
        "max_drawdown": float(max_dd),
        "total_tx_cost": float(total_tx_cost),
        "total_short_cost": float(total_short_cost),
        "n_periods": len(net_rets),
        "period_returns_gross": gross_rets.tolist(),
        "period_returns_net": net_rets.tolist(),
    }


def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> Tuple[float, float]:
    """Compute Information Coefficient."""
    ics = []
    T = predictions.shape[0]
    
    for t in range(T):
        pred = predictions[t]
        actual = actuals[t]
        mask = np.isfinite(pred) & np.isfinite(actual)
        
        if mask.sum() > 10:
            ic = np.corrcoef(pred[mask], actual[mask])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    
    ics = np.array(ics)
    mean_ic = ics.mean() if len(ics) > 0 else 0
    icir = mean_ic / ics.std() if ics.std() > 0 else 0
    
    return float(mean_ic), float(icir)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    industry_mask: Optional[torch.Tensor] = None,
    universe_mask: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Train a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            features, targets = batch
            features = features.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, DMFM):
                outputs = model(features, industry_mask, universe_mask)
            elif isinstance(model, MLPGATModel):
                outputs = model(features, universe_mask)
            else:
                outputs = model(features)
            
            # Loss on valid targets only
            mask = ~torch.isnan(targets)
            if mask.sum() > 0:
                loss = criterion(outputs[mask], targets[mask])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features, targets = batch
                features = features.to(config.device)
                targets = targets.to(config.device)
                
                if isinstance(model, DMFM):
                    outputs = model(features, industry_mask, universe_mask)
                elif isinstance(model, MLPGATModel):
                    outputs = model(features, universe_mask)
                else:
                    outputs = model(features)
                
                mask = ~torch.isnan(targets)
                if mask.sum() > 0:
                    loss = criterion(outputs[mask], targets[mask])
                    val_loss += loss.item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    industry_mask: Optional[torch.Tensor] = None,
    universe_mask: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features, targets = batch
            features = features.to(config.device)
            
            if isinstance(model, DMFM):
                outputs = model(features, industry_mask, universe_mask)
            elif isinstance(model, MLPGATModel):
                outputs = model(features, universe_mask)
            else:
                outputs = model(features)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_targets, axis=0)
    
    return predictions, actuals


def run_experiment(
    data_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> Dict:
    """Run the full experiment with transaction costs."""
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    features_np = data['features']
    T, N, F = features_np.shape
    
    # Get forward returns for evaluation horizon
    horizon_key = f'forward_returns_{config.eval_horizon}'
    if horizon_key in data:
        targets_np = data[horizon_key]
    else:
        # Fallback: use first horizon
        targets_np = data['forward_returns_5']
    
    print(f"Data shape: T={T}, N={N}, F={F}")
    
    # Load masks
    if 'industry_mask' in data:
        industry_mask_np = data['industry_mask']
    else:
        industry_mask_np = np.eye(N, dtype=bool)
    
    if 'universe_mask' in data:
        universe_mask_np = data['universe_mask']
    else:
        universe_mask_np = np.ones((N, N), dtype=bool)
    
    # Convert to tensors
    features = torch.FloatTensor(features_np)
    targets = torch.FloatTensor(targets_np)
    industry_mask = torch.BoolTensor(industry_mask_np).to(config.device)
    universe_mask = torch.BoolTensor(universe_mask_np).to(config.device)
    
    # Define periods
    periods = {
        "early": {"train": (0.0, 0.5), "test": (0.5, 0.7)},
        "middle": {"train": (0.2, 0.6), "test": (0.6, 0.8)},
        "recent": {"train": (0.4, 0.8), "test": (0.8, 1.0)},
    }
    
    # Models to evaluate
    model_configs = {
        "dmfm": lambda: DMFM(
            input_dim=F,
            hidden_dim=config.hidden_dim,
            n_industries=N,
            n_stocks=N,
            n_factors=10,
        ).to(config.device),
        "linear": lambda: LinearModel(input_dim=F).to(config.device),
        "mlp": lambda: MLPModel(input_dim=F, hidden_dim=config.hidden_dim).to(config.device),
        "mlp_gat": lambda: MLPGATModel(
            input_dim=F, hidden_dim=config.hidden_dim, n_stocks=N
        ).to(config.device),
        "momentum": lambda: MomentumModel(input_dim=F).to(config.device),
        "reversal": lambda: ReversalModel(input_dim=F).to(config.device),
    }
    
    results = {"periods": {}, "config": config.__dict__.copy()}
    results["config"]["device"] = str(config.device)
    
    for period_name, period_range in periods.items():
        print(f"\n{'='*60}")
        print(f"Processing period: {period_name}")
        print(f"{'='*60}")
        
        # Split data
        train_start = int(T * period_range["train"][0])
        train_end = int(T * period_range["train"][1])
        test_start = int(T * period_range["test"][0])
        test_end = int(T * period_range["test"][1])
        
        # Validation is last 20% of training
        val_split = int(train_end - (train_end - train_start) * 0.2)
        
        train_features = features[train_start:val_split]
        train_targets = targets[train_start:val_split]
        val_features = features[val_split:train_end]
        val_targets = targets[val_split:train_end]
        test_features = features[test_start:test_end]
        test_targets = targets[test_start:test_end]
        
        print(f"Train: {train_start}-{val_split}, Val: {val_split}-{train_end}, Test: {test_start}-{test_end}")
        
        # Create data loaders
        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)
        test_dataset = TensorDataset(test_features, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
        
        period_results = {}
        
        for model_name, model_fn in model_configs.items():
            print(f"\n  Training {model_name}...")
            
            # Create and train model
            model = model_fn()
            
            if model_name in ["momentum", "reversal"]:
                # No training needed for these
                pass
            else:
                model = train_model(
                    model, train_loader, val_loader, config,
                    industry_mask, universe_mask
                )
            
            # Evaluate
            predictions, actuals = evaluate_model(
                model, test_loader, config,
                industry_mask, universe_mask
            )
            
            # Compute IC
            mean_ic, icir = compute_ic(predictions, actuals)
            
            # Compute backtest with costs
            backtest = compute_backtest_with_costs(predictions, actuals, config)
            
            period_results[model_name] = {
                "mean_ic": mean_ic,
                "icir": icir,
                "backtest": backtest,
            }
            
            print(f"    IC: {mean_ic:.4f}, ICIR: {icir:.4f}")
            print(f"    Sharpe (Gross): {backtest['sharpe_gross']:.2f}, Sharpe (Net): {backtest['sharpe_net']:.2f}")
            print(f"    Return (Gross): {backtest['cumulative_return_gross']*100:.1f}%, Return (Net): {backtest['cumulative_return_net']*100:.1f}%")
        
        results["periods"][period_name] = period_results
    
    return results


def print_summary(results: Dict):
    """Print experiment summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY (WITH TRANSACTION COSTS)")
    print("=" * 80)
    
    cfg = results["config"]
    print(f"\nTransaction Cost Assumptions:")
    print(f"  Commission: {cfg['commission_rate']*100:.2f}%")
    print(f"  Slippage: {cfg['slippage_rate']*100:.2f}%")
    print(f"  Short Borrow Rate: {cfg['short_borrow_rate']*100:.1f}% p.a.")
    print(f"  Holding Period: {cfg['holding_period']} days")
    
    # Summary table
    models = list(list(results["periods"].values())[0].keys())
    
    print(f"\n{'Model':<12} ", end="")
    for period in results["periods"].keys():
        print(f"| {period:^20} ", end="")
    print("| {'Avg':^20}")
    
    print("-" * 90)
    
    for model in models:
        print(f"{model:<12} ", end="")
        sharpes_net = []
        for period_name, period_data in results["periods"].items():
            if model in period_data:
                bt = period_data[model]["backtest"]
                sharpe_net = bt["sharpe_net"]
                sharpes_net.append(sharpe_net)
                print(f"| {sharpe_net:>6.2f} (Net SR)     ", end="")
        avg_sharpe = np.mean(sharpes_net) if sharpes_net else 0
        print(f"| {avg_sharpe:>6.2f}             ")
    
    # Best model
    print("\n" + "-" * 90)
    avg_sharpes = {}
    for model in models:
        sharpes = []
        for period_data in results["periods"].values():
            if model in period_data:
                sharpes.append(period_data[model]["backtest"]["sharpe_net"])
        avg_sharpes[model] = np.mean(sharpes) if sharpes else 0
    
    best_model = max(avg_sharpes, key=avg_sharpes.get)
    print(f"\n🏆 Best Model: {best_model.upper()} (Avg Net Sharpe: {avg_sharpes[best_model]:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", type=str, default="./dmfm_data_extended.npz")
    parser.add_argument("--output-dir", type=str, default="./results_realistic")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.002)
    parser.add_argument("--short-rate", type=float, default=0.02)
    parser.add_argument("--holding-period", type=int, default=5)
    args = parser.parse_args()
    
    config = ExperimentConfig(
        epochs=args.epochs,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        short_borrow_rate=args.short_rate,
        holding_period=args.holding_period,
    )
    
    data_path = Path(args.data_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running realistic experiment...")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Commission: {config.commission_rate*100:.2f}%")
    print(f"  Slippage: {config.slippage_rate*100:.2f}%")
    print(f"  Short Borrow Rate: {config.short_borrow_rate*100:.1f}% p.a.")
    
    # Run experiment
    results = run_experiment(data_path, output_dir, config)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = output_dir / "experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
