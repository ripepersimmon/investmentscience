"""Backtest framework for DMFM and baseline models.

Implements:
- Train/validation/test split with no data leakage
- Portfolio construction from factor scores
- Performance metrics (Sharpe, IC, ICIR, Max Drawdown, etc.)
- Comparison across multiple models

Usage:
    python -m dmfm.backtest --data-dir ./kospi --industry-csv ./kospi_industry.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dmfm.baselines import get_baseline_model
from dmfm.config import LossConfig, ModelConfig
from dmfm.data import DMFMDataset
from dmfm.losses import DMFMLoss, compute_ic, compute_icir, factor_return
from dmfm.model import DMFM
from dmfm.train import collate, seed_all


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    train_ratio: float = 0.6       # 60% for training
    val_ratio: float = 0.2         # 20% for validation
    test_ratio: float = 0.2        # 20% for testing (out-of-sample)
    
    # Portfolio construction
    top_k: int = 10                # Long top K stocks
    bottom_k: int = 10             # Short bottom K stocks
    rebalance_freq: int = 5        # Rebalance every N days
    
    # Training
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    hidden_dim: int = 128
    
    # Horizons to evaluate
    eval_horizon: int = 5          # Primary horizon for portfolio evaluation


@dataclass
class BacktestResult:
    """Results from backtesting a single model."""
    model_name: str
    
    # In-sample metrics (training period)
    train_ic: float = 0.0
    train_icir: float = 0.0
    
    # Out-of-sample metrics (test period)
    test_ic: float = 0.0
    test_icir: float = 0.0
    test_factor_return: float = 0.0
    
    # Portfolio metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # Daily returns for plotting
    daily_returns: List[float] = field(default_factory=list)
    cumulative_returns: List[float] = field(default_factory=list)


def compute_portfolio_returns(
    factor_scores: np.ndarray,
    forward_returns: np.ndarray,
    top_k: int = 10,
    bottom_k: int = 10,
    holding_period: int = 5,  # Match the forward return horizon
) -> np.ndarray:
    """Compute long-short portfolio returns with proper non-overlapping periods.
    
    Args:
        factor_scores: (T, N) factor scores
        forward_returns: (T, N) forward returns for holding_period days
        top_k: Number of stocks to go long
        bottom_k: Number of stocks to go short
        holding_period: Days to hold each position (should match forward return horizon)
        
    Returns:
        (T // holding_period,) portfolio returns per rebalancing period
    """
    T, N = factor_scores.shape
    n_periods = T // holding_period
    portfolio_returns = []
    
    for p in range(n_periods):
        t = p * holding_period  # Rebalance point
        
        scores = factor_scores[t]
        returns = forward_returns[t]  # This is the return from t to t+holding_period
        
        # Handle NaN scores
        valid_mask = np.isfinite(scores) & np.isfinite(returns)
        if valid_mask.sum() < top_k + bottom_k:
            portfolio_returns.append(0.0)
            continue
        
        # Rank stocks by factor score
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        sorted_idx = np.argsort(valid_scores)[::-1]  # Descending
        
        # Long top K, short bottom K
        long_idx = valid_indices[sorted_idx[:top_k]]
        short_idx = valid_indices[sorted_idx[-bottom_k:]]
        
        # Equal weight within long/short legs
        long_return = returns[long_idx].mean()
        short_return = returns[short_idx].mean()
        
        # Long-short portfolio return
        portfolio_returns.append((long_return - short_return) / 2)
    
    return np.array(portfolio_returns)


def compute_metrics(
    returns: np.ndarray, 
    holding_period: int = 5,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """Compute portfolio performance metrics.
    
    Args:
        returns: Array of period returns (one per rebalancing period)
        holding_period: Days per holding period
        trading_days_per_year: Trading days per year
        
    Returns:
        Dictionary of metrics
    """
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
    
    # Cumulative returns
    cum_returns = np.cumprod(1 + returns)
    total_return = cum_returns[-1] - 1
    
    # Number of periods per year
    periods_per_year = trading_days_per_year / holding_period
    
    # Annualized metrics
    n_periods = len(returns)
    n_years = n_periods / periods_per_year
    
    if n_years > 0:
        annual_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annual_return = 0.0
    
    # Annualized volatility
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio (assuming 0 risk-free rate)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BacktestConfig,
    model_cfg: ModelConfig,
    device: torch.device,
    model_name: str = "model",
) -> nn.Module:
    """Train a model with early stopping based on validation IC."""
    
    loss_cfg = LossConfig()
    loss_fn = DMFMLoss(model_cfg, loss_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    
    best_val_ic = -float('inf')
    best_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else {kk: vv.to(device) for kk, vv in v.items()}) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch["features"], batch["industry_mask"], batch["universe_mask"])
            loss, _ = loss_fn(outputs, batch["forward_returns"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_ics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else {kk: vv.to(device) for kk, vv in v.items()}) for k, v in batch.items()}
                outputs = model(batch["features"], batch["industry_mask"], batch["universe_mask"])
                
                h = str(config.eval_horizon)
                if h in outputs["factors"]:
                    f = outputs["factors"][h]
                    r = batch["forward_returns"][h]
                    ic = compute_ic(f, r).mean().item()
                    val_ics.append(ic)
        
        avg_val_ic = np.mean(val_ics) if val_ics else 0
        
        # Early stopping
        if avg_val_ic > best_val_ic:
            best_val_ic = avg_val_ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{config.epochs} | Val IC: {avg_val_ic:.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: BacktestConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate model on test set.
    
    Returns:
        factor_scores: (T, N) predicted factor scores
        forward_returns: (T, N) actual forward returns
        metrics: Dictionary of IC/ICIR metrics
    """
    model.eval()
    
    all_factors = []
    all_returns = []
    all_ics = []
    
    h = str(config.eval_horizon)
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else {kk: vv.to(device) for kk, vv in v.items()}) for k, v in batch.items()}
            outputs = model(batch["features"], batch["industry_mask"], batch["universe_mask"])
            
            if h in outputs["factors"]:
                f = outputs["factors"][h]
                r = batch["forward_returns"][h]
                
                all_factors.append(f.cpu().numpy())
                all_returns.append(r.cpu().numpy())
                
                ic = compute_ic(f, r)
                all_ics.extend(ic.cpu().numpy().tolist())
    
    factor_scores = np.concatenate(all_factors, axis=0)
    forward_returns = np.concatenate(all_returns, axis=0)
    
    # Compute metrics
    ic_array = np.array(all_ics)
    mean_ic = np.nanmean(ic_array)
    icir = mean_ic / (np.nanstd(ic_array) + 1e-6)
    
    # Factor return
    fr = []
    for t in range(len(factor_scores)):
        f = factor_scores[t]
        r = forward_returns[t]
        mask = np.isfinite(f) & np.isfinite(r)
        if mask.sum() > 0:
            b = np.dot(f[mask], r[mask]) / (np.dot(f[mask], f[mask]) + 1e-6)
            fr.append(b)
    mean_factor_return = np.mean(fr) if fr else 0
    
    metrics = {
        "test_ic": float(mean_ic),
        "test_icir": float(icir),
        "test_factor_return": float(mean_factor_return),
    }
    
    return factor_scores, forward_returns, metrics


def run_backtest(
    data_dir: Path,
    industry_csv: Optional[Path],
    config: BacktestConfig,
    models_to_test: List[str],
    device: torch.device,
    limit_symbols: int = 100,
    output_dir: Optional[Path] = None,
) -> Dict[str, BacktestResult]:
    """Run full backtest comparing multiple models.
    
    Args:
        data_dir: Directory with stock CSVs
        industry_csv: Path to industry classification CSV
        config: Backtest configuration
        models_to_test: List of model names to evaluate
        device: PyTorch device
        limit_symbols: Number of symbols to use
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping model name to BacktestResult
    """
    from dmfm.run_kospi import build_panel
    
    seed_all(42)
    
    # Load data
    horizons = [3, 5, 10, 15, 20]
    features_np, forward_returns_np, industry_masks_np, symbols = build_panel(
        data_dir, limit_symbols, horizons, max_dates=None, industry_csv=industry_csv
    )
    
    T, N, F = features_np.shape
    print(f"\n{'='*60}")
    print(f"BACKTEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"Total periods: {T}")
    print(f"Number of stocks: {N}")
    print(f"Number of features: {F}")
    print(f"Primary horizon: {config.eval_horizon} days")
    
    # Split data: Train / Validation / Test
    train_end = int(T * config.train_ratio)
    val_end = int(T * (config.train_ratio + config.val_ratio))
    
    print(f"\nData split:")
    print(f"  Training:   periods 0-{train_end-1} ({train_end} days)")
    print(f"  Validation: periods {train_end}-{val_end-1} ({val_end - train_end} days)")
    print(f"  Testing:    periods {val_end}-{T-1} ({T - val_end} days)")
    print(f"{'='*60}\n")
    
    # Convert to tensors
    features = torch.from_numpy(features_np)
    forward_returns = {k: torch.from_numpy(v) for k, v in forward_returns_np.items()}
    industry_masks = torch.from_numpy(industry_masks_np)
    universe_masks = torch.ones(T, N, N, dtype=torch.bool)
    
    # Create datasets
    full_dataset = DMFMDataset(features, industry_masks, forward_returns, universe_masks)
    
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, T))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    # Model config
    model_cfg = ModelConfig(feature_dim=F, hidden_dim=config.hidden_dim, horizons=tuple(horizons))
    
    results: Dict[str, BacktestResult] = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*40}")
        print(f"Training and evaluating: {model_name.upper()}")
        print(f"{'='*40}")
        
        # Create model
        if model_name == "dmfm":
            model = DMFM(model_cfg).to(device)
        else:
            model = get_baseline_model(
                model_name, 
                feature_dim=F, 
                hidden_dim=config.hidden_dim,
                horizons=tuple(horizons)
            ).to(device)
        
        # Train model (skip for non-trainable models)
        if model_name not in ["momentum", "reversal", "equal_weight"]:
            model = train_model(model, train_loader, val_loader, config, model_cfg, device, model_name)
        
        # Evaluate on test set
        factor_scores, fwd_returns, metrics = evaluate_model(model, test_loader, config, device)
        
        # Compute portfolio returns
        portfolio_returns = compute_portfolio_returns(
            factor_scores, fwd_returns,
            top_k=config.top_k, bottom_k=config.bottom_k
        )
        
        # Compute performance metrics
        perf_metrics = compute_metrics(portfolio_returns)
        
        # Store results
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
        
        print(f"\n  Results for {model_name}:")
        print(f"    IC: {result.test_ic:.4f}")
        print(f"    ICIR: {result.test_icir:.4f}")
        print(f"    Sharpe: {result.sharpe_ratio:.4f}")
        print(f"    Total Return: {result.total_return*100:.2f}%")
        print(f"    Max Drawdown: {result.max_drawdown*100:.2f}%")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("BACKTEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'WinRate':>8}")
    print("-" * 80)
    
    for name, r in sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True):
        print(f"{name:<15} {r.test_ic:>8.4f} {r.test_icir:>8.4f} {r.sharpe_ratio:>8.4f} "
              f"{r.total_return*100:>9.2f}% {r.max_drawdown*100:>9.2f}% {r.win_rate*100:>7.1f}%")
    
    print(f"{'='*80}")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            "config": {
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "eval_horizon": config.eval_horizon,
                "top_k": config.top_k,
                "bottom_k": config.bottom_k,
                "total_periods": T,
                "num_stocks": N,
                "num_features": F,
            },
            "results": {
                name: {
                    "test_ic": r.test_ic,
                    "test_icir": r.test_icir,
                    "sharpe_ratio": r.sharpe_ratio,
                    "total_return": r.total_return,
                    "annual_return": r.annual_return,
                    "max_drawdown": r.max_drawdown,
                    "win_rate": r.win_rate,
                }
                for name, r in results.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_dir / "backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save cumulative returns for plotting
        cum_returns_df = pd.DataFrame({
            name: r.cumulative_returns for name, r in results.items()
        })
        cum_returns_df.to_csv(output_dir / "cumulative_returns.csv", index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest DMFM and baseline models")
    parser.add_argument("--data-dir", type=str, default="./kospi")
    parser.add_argument("--industry-csv", type=str, default="./kospi_industry.csv")
    parser.add_argument("--output-dir", type=str, default="./backtest_results")
    parser.add_argument("--limit-symbols", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-horizon", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["dmfm", "mlp_gat", "mlp", "linear", "momentum", "reversal"],
        help="Models to evaluate"
    )
    args = parser.parse_args()
    
    config = BacktestConfig(
        epochs=args.epochs,
        eval_horizon=args.eval_horizon,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    results = run_backtest(
        data_dir=Path(args.data_dir),
        industry_csv=Path(args.industry_csv) if args.industry_csv else None,
        config=config,
        models_to_test=args.models,
        device=device,
        limit_symbols=args.limit_symbols,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
