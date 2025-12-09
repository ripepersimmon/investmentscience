"""Realistic backtest with transaction costs and constraints.

Implements:
- Transaction costs (commission + slippage)
- Short selling costs (borrow fee)
- Turnover penalty
- Liquidity constraints
- Rebalancing frequency

Usage:
    python -m dmfm.realistic_backtest --results-dir ./results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RealisticConfig:
    """Realistic trading constraints."""
    
    # Transaction costs (one-way)
    commission_rate: float = 0.001      # 0.1% commission per trade
    slippage_rate: float = 0.001        # 0.1% slippage (market impact)
    
    # Short selling costs (annual rate, applied per holding period)
    short_borrow_rate: float = 0.03     # 3% annual borrow fee
    
    # Portfolio constraints
    top_k: int = 10                     # Long positions
    bottom_k: int = 10                  # Short positions (0 for long-only)
    
    # Rebalancing
    rebalance_freq: int = 5             # Rebalance every N days (1=daily)
    
    # Liquidity filter
    min_avg_volume: float = 0.0         # Minimum average daily volume (KRW)
    
    # Long-only mode
    long_only: bool = False


def compute_realistic_returns(
    factor_scores: np.ndarray,
    forward_returns: np.ndarray,
    config: RealisticConfig,
    prev_long: Optional[set] = None,
    prev_short: Optional[set] = None,
) -> tuple[np.ndarray, float, set, set]:
    """Compute portfolio returns with realistic costs.
    
    Args:
        factor_scores: (T, N) factor scores
        forward_returns: (T, N) forward returns (already for the holding period)
        config: Trading constraints
        prev_long: Previous long positions (for turnover calc)
        prev_short: Previous short positions
        
    Returns:
        returns: (T//rebalance_freq,) portfolio returns per rebalancing period
        total_cost: Total transaction costs incurred
        final_long: Final long positions
        final_short: Final short positions
    """
    T, N = factor_scores.shape
    rebal_freq = config.rebalance_freq
    
    # Number of rebalancing periods
    n_periods = T // rebal_freq
    portfolio_returns = []
    total_costs = 0.0
    
    current_long = prev_long or set()
    current_short = prev_short or set()
    
    for p in range(n_periods):
        t_start = p * rebal_freq
        t_end = min((p + 1) * rebal_freq, T)
        
        # Use factor score at rebalancing point
        scores = factor_scores[t_start]
        
        # Aggregate returns over holding period
        period_returns = forward_returns[t_start:t_end].mean(axis=0)
        
        # Valid stocks
        valid_mask = np.isfinite(scores) & np.isfinite(period_returns)
        if valid_mask.sum() < config.top_k + config.bottom_k:
            portfolio_returns.append(0.0)
            continue
        
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]
        sorted_idx = np.argsort(valid_scores)[::-1]
        
        # New positions
        new_long = set(valid_indices[sorted_idx[:config.top_k]])
        if config.long_only:
            new_short = set()
        else:
            new_short = set(valid_indices[sorted_idx[-config.bottom_k:]])
        
        # Calculate turnover
        long_turnover = len(new_long - current_long) + len(current_long - new_long)
        short_turnover = len(new_short - current_short) + len(current_short - new_short)
        total_turnover = long_turnover + short_turnover
        
        # Transaction cost = (turnover / 2) * 2 * (commission + slippage)
        # turnover/2 because we count both entry and exit
        # *2 because round-trip
        tx_cost_rate = (total_turnover / 2) * 2 * (config.commission_rate + config.slippage_rate)
        # Normalize by number of positions
        n_positions = config.top_k + (0 if config.long_only else config.bottom_k)
        tx_cost = tx_cost_rate / n_positions if n_positions > 0 else 0
        
        # Portfolio return
        long_indices = list(new_long)
        short_indices = list(new_short)
        
        long_return = period_returns[long_indices].mean() if long_indices else 0
        
        if config.long_only:
            gross_return = long_return
        else:
            short_return = period_returns[short_indices].mean() if short_indices else 0
            
            # Short borrow cost (prorated for holding period)
            holding_days = t_end - t_start
            short_cost = config.short_borrow_rate * (holding_days / 252)
            
            # Long-short return with short costs
            gross_return = (long_return - short_return) / 2 - short_cost / 2
        
        # Net return after transaction costs
        net_return = gross_return - tx_cost
        
        portfolio_returns.append(net_return)
        total_costs += tx_cost
        
        current_long = new_long
        current_short = new_short
    
    return np.array(portfolio_returns), total_costs, current_long, current_short


def run_realistic_backtest(
    results_dir: Path,
    config: RealisticConfig,
) -> Dict:
    """Run realistic backtest on saved results."""
    
    # Load original results
    with open(results_dir / "experiment_results.json", "r") as f:
        orig_results = json.load(f)
    
    # Load cumulative return data to reconstruct daily returns
    realistic_results = {
        "config": {
            "commission_rate": config.commission_rate,
            "slippage_rate": config.slippage_rate,
            "short_borrow_rate": config.short_borrow_rate,
            "top_k": config.top_k,
            "bottom_k": config.bottom_k,
            "rebalance_freq": config.rebalance_freq,
            "long_only": config.long_only,
        },
        "results": {},
    }
    
    for period in orig_results["results"].keys():
        realistic_results["results"][period] = {}
        
        for model, data in orig_results["results"][period].items():
            daily_returns = np.array(data["daily_returns"])
            
            if len(daily_returns) == 0:
                continue
            
            # Reconstruct factor-implied returns with costs
            # Since we don't have the original factor scores, we'll apply costs to daily returns
            
            T = len(daily_returns)
            n_periods = T // config.rebalance_freq
            
            # Aggregate returns per rebalancing period
            period_returns = []
            for p in range(n_periods):
                t_start = p * config.rebalance_freq
                t_end = min((p + 1) * config.rebalance_freq, T)
                # Compound returns over holding period
                period_ret = np.prod(1 + daily_returns[t_start:t_end]) - 1
                period_returns.append(period_ret)
            
            period_returns = np.array(period_returns)
            
            # Apply costs
            # Transaction cost per rebalance (assume 50% turnover on average)
            avg_turnover = 0.5  # 50% portfolio changes each rebalance
            tx_cost_per_period = avg_turnover * 2 * (config.commission_rate + config.slippage_rate)
            
            # Short cost per period
            if not config.long_only:
                holding_days = config.rebalance_freq
                short_cost_per_period = config.short_borrow_rate * (holding_days / 252) / 2
            else:
                short_cost_per_period = 0
                # For long-only, only use half the return (no shorting)
                period_returns = period_returns / 2  # Rough approximation
            
            # Net returns
            net_returns = period_returns - tx_cost_per_period - short_cost_per_period
            
            # Compute metrics
            cum_returns = np.cumprod(1 + net_returns)
            total_return = cum_returns[-1] - 1 if len(cum_returns) > 0 else 0
            
            # Annualized metrics
            n_years = len(net_returns) * config.rebalance_freq / 252
            annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
            annual_vol = net_returns.std() * np.sqrt(252 / config.rebalance_freq)
            
            # Correct Sharpe: mean / std * sqrt(periods_per_year)
            periods_per_year = 252 / config.rebalance_freq
            sharpe = (net_returns.mean() / net_returns.std() * np.sqrt(periods_per_year)) if net_returns.std() > 0 else 0
            
            # Max drawdown
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / peak
            max_dd = drawdown.min() if len(drawdown) > 0 else 0
            
            realistic_results["results"][period][model] = {
                "gross_return": float(data["total_return"]),
                "net_return": float(total_return),
                "gross_sharpe": float(data["sharpe_ratio"]),
                "net_sharpe": float(sharpe),
                "annual_return": float(annual_return),
                "annual_volatility": float(annual_vol),
                "max_drawdown": float(max_dd),
                "total_tx_cost": float(tx_cost_per_period * n_periods),
                "total_short_cost": float(short_cost_per_period * n_periods),
            }
    
    return realistic_results


def print_comparison(orig_results: Dict, realistic_results: Dict):
    """Print comparison of gross vs net performance."""
    
    print("\n" + "=" * 80)
    print("REALISTIC BACKTEST COMPARISON")
    print("=" * 80)
    
    print("\nTransaction Cost Assumptions:")
    cfg = realistic_results["config"]
    print(f"  Commission: {cfg['commission_rate']*100:.2f}%")
    print(f"  Slippage: {cfg['slippage_rate']*100:.2f}%")
    print(f"  Short Borrow Rate: {cfg['short_borrow_rate']*100:.1f}% p.a.")
    print(f"  Rebalance Frequency: Every {cfg['rebalance_freq']} days")
    print(f"  Long-Only: {cfg['long_only']}")
    
    for period in realistic_results["results"].keys():
        print(f"\n--- {period.upper()} Period ---")
        print(f"{'Model':<12} {'Gross Ret':>12} {'Net Ret':>12} {'Gross SR':>10} {'Net SR':>10} {'Cost':>10}")
        print("-" * 70)
        
        period_data = realistic_results["results"][period]
        sorted_models = sorted(period_data.items(), key=lambda x: -x[1]["net_sharpe"])
        
        for model, data in sorted_models:
            print(f"{model:<12} "
                  f"{data['gross_return']*100:>11.1f}% "
                  f"{data['net_return']*100:>11.1f}% "
                  f"{data['gross_sharpe']:>10.2f} "
                  f"{data['net_sharpe']:>10.2f} "
                  f"{(data['total_tx_cost']+data['total_short_cost'])*100:>9.1f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (Average Across Periods)")
    print("=" * 80)
    
    models = list(list(realistic_results["results"].values())[0].keys())
    summary = []
    
    for model in models:
        gross_sharpes = []
        net_sharpes = []
        net_returns = []
        
        for period in realistic_results["results"].keys():
            if model in realistic_results["results"][period]:
                data = realistic_results["results"][period][model]
                gross_sharpes.append(data["gross_sharpe"])
                net_sharpes.append(data["net_sharpe"])
                net_returns.append(data["net_return"])
        
        summary.append({
            "model": model,
            "avg_gross_sharpe": np.mean(gross_sharpes),
            "avg_net_sharpe": np.mean(net_sharpes),
            "avg_net_return": np.mean(net_returns),
            "sharpe_decay": np.mean(gross_sharpes) - np.mean(net_sharpes),
        })
    
    summary = sorted(summary, key=lambda x: -x["avg_net_sharpe"])
    
    print(f"\n{'Rank':<6} {'Model':<12} {'Gross SR':>10} {'Net SR':>10} {'Decay':>10} {'Avg Ret':>12}")
    print("-" * 65)
    
    for i, s in enumerate(summary):
        print(f"{i+1:<6} {s['model']:<12} "
              f"{s['avg_gross_sharpe']:>10.2f} "
              f"{s['avg_net_sharpe']:>10.2f} "
              f"{s['sharpe_decay']:>10.2f} "
              f"{s['avg_net_return']*100:>11.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    parser.add_argument("--slippage", type=float, default=0.001, help="Slippage rate")
    parser.add_argument("--short-rate", type=float, default=0.03, help="Short borrow rate (annual)")
    parser.add_argument("--rebalance-freq", type=int, default=5, help="Rebalance frequency (days)")
    parser.add_argument("--long-only", action="store_true", help="Long-only strategy")
    args = parser.parse_args()
    
    config = RealisticConfig(
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        short_borrow_rate=args.short_rate,
        rebalance_freq=args.rebalance_freq,
        long_only=args.long_only,
    )
    
    results_dir = Path(args.results_dir)
    
    # Load original results
    with open(results_dir / "experiment_results.json", "r") as f:
        orig_results = json.load(f)
    
    # Run realistic backtest
    realistic_results = run_realistic_backtest(results_dir, config)
    
    # Print comparison
    print_comparison(orig_results, realistic_results)
    
    # Save results
    output_file = results_dir / "realistic_backtest_results.json"
    with open(output_file, "w") as f:
        json.dump(realistic_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
