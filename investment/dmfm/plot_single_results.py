#!/usr/bin/env python
"""
Plot results for single-period backtest
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Color scheme
COLORS = {
    'dmfm': '#1f77b4',      # blue
    'linear': '#ff7f0e',     # orange
    'mlp': '#2ca02c',        # green
    'mlp_gat': '#d62728',    # red
    'momentum': '#9467bd',   # purple
    'reversal': '#8c564b',   # brown
}


def load_results(result_dir: Path):
    """Load experiment results"""
    with open(result_dir / 'experiment_results.json', 'r') as f:
        results = json.load(f)
    
    cumulative_df = pd.read_csv(result_dir / 'cumulative_returns.csv', index_col=0, parse_dates=True)
    
    return results, cumulative_df


def plot_cumulative_returns(results, cumulative_df, output_dir: Path):
    """Plot cumulative returns"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model in cumulative_df.columns:
        color = COLORS.get(model.lower(), '#333333')
        ax.plot(cumulative_df.index, cumulative_df[model] * 100, 
                label=model.upper(), color=color, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Single Period Backtest: Cumulative Returns (Test Period: 2023-05 ~ 2025-11)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cumulative_returns.png")


def plot_metrics_comparison(results, output_dir: Path):
    """Plot metrics comparison bar chart"""
    models = results['models']
    metrics = ['test_ic', 'test_icir', 'sharpe_ratio', 'total_return']
    metric_labels = ['IC', 'ICIR', 'Sharpe Ratio', 'Total Return (%)']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = []
        colors = []
        
        for model in models:
            val = results['results'][model][metric]
            if metric == 'total_return':
                val *= 100  # Convert to percentage
            values.append(val)
            colors.append(COLORS.get(model.lower(), '#333333'))
        
        bars = ax.bar([m.upper() for m in models], values, color=colors)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}' if metric != 'total_return' else f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Single Period Backtest: Model Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_comparison.png")


def plot_risk_return(results, output_dir: Path):
    """Plot risk-return scatter"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in results['models']:
        res = results['results'][model]
        vol = res['annual_volatility'] * 100
        ret = res['annual_return'] * 100
        sharpe = res['sharpe_ratio']
        
        color = COLORS.get(model.lower(), '#333333')
        ax.scatter(vol, ret, s=200, c=color, label=f"{model.upper()} (SR={sharpe:.2f})", 
                  edgecolors='white', linewidth=2)
        ax.annotate(model.upper(), (vol, ret), textcoords="offset points",
                   xytext=(10, 5), ha='left', fontsize=10)
    
    ax.set_xlabel('Annual Volatility (%)')
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Risk-Return Profile (Test Period: 2023-05 ~ 2025-11)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_return.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: risk_return.png")


def plot_drawdown(results, cumulative_df, output_dir: Path):
    """Plot drawdown comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model in cumulative_df.columns:
        cum_ret = cumulative_df[model] + 1
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max * 100
        
        color = COLORS.get(model.lower(), '#333333')
        ax.plot(cumulative_df.index, drawdown, label=model.upper(), color=color, linewidth=1.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Single Period Backtest: Drawdown (Test Period: 2023-05 ~ 2025-11)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.fill_between(cumulative_df.index, ax.get_ylim()[0], 0, alpha=0.1, color='red')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drawdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: drawdown.png")


def generate_report(results, output_dir: Path):
    """Generate markdown report"""
    report = []
    report.append("# Single Period Backtest Report\n")
    report.append(f"**Generated:** {results['experiment_timestamp'][:10]}\n")
    
    # Experiment Configuration
    report.append("## Experiment Configuration\n")
    config = results['config']
    report.append(f"- **Train/Val/Test Split:** {int(config['train_ratio']*100)}% / {int(config['val_ratio']*100)}% / {int(config['test_ratio']*100)}%")
    report.append(f"- **Portfolio:** Long-Short (Top {config['top_k']} / Bottom {config['bottom_k']})")
    report.append(f"- **Holding Period:** {config['eval_horizon']} days")
    report.append(f"- **Training:** {config['epochs']} epochs, LR={config['lr']}, Hidden={config['hidden_dim']}")
    report.append("")
    
    # Period Information
    report.append("## Data Split\n")
    split = results['split_info']
    report.append(f"| Period | Days | Date Range |")
    report.append(f"|--------|------|------------|")
    report.append(f"| Train | {split['train'][1] - split['train'][0]} | {split['train'][2]} ~ {split['train'][3]} |")
    report.append(f"| Validation | {split['val'][1] - split['val'][0]} | {split['val'][2]} ~ {split['val'][3]} |")
    report.append(f"| Test | {split['test'][1] - split['test'][0]} | {split['test'][2]} ~ {split['test'][3]} |")
    report.append("")
    
    # Results Table
    report.append("## Results Summary\n")
    report.append("| Rank | Model | IC | ICIR | Sharpe | Return | MDD | Win Rate |")
    report.append("|------|-------|-----|------|--------|--------|-----|----------|")
    
    # Sort by Sharpe
    model_sharpes = [(m, results['results'][m]['sharpe_ratio']) for m in results['models']]
    sorted_models = sorted(model_sharpes, key=lambda x: x[1], reverse=True)
    
    for rank, (model, _) in enumerate(sorted_models, 1):
        res = results['results'][model]
        report.append(f"| {rank} | {model.upper()} | {res['test_ic']:.4f} | {res['test_icir']:.4f} | "
                     f"{res['sharpe_ratio']:.2f} | {res['total_return']*100:.1f}% | "
                     f"{res['max_drawdown']*100:.1f}% | {res['win_rate']*100:.1f}% |")
    report.append("")
    
    # Key Findings
    report.append("## Key Findings\n")
    best_sharpe = sorted_models[0]
    best_ic = max(results['results'].items(), key=lambda x: x[1]['test_ic'])
    best_icir = max(results['results'].items(), key=lambda x: x[1]['test_icir'])
    best_return = max(results['results'].items(), key=lambda x: x[1]['total_return'])
    
    report.append(f"1. **Best Sharpe Ratio:** {best_sharpe[0].upper()} ({best_sharpe[1]:.2f})")
    report.append(f"2. **Best IC:** {best_ic[0].upper()} ({best_ic[1]['test_ic']:.4f})")
    report.append(f"3. **Best ICIR:** {best_icir[0].upper()} ({best_icir[1]['test_icir']:.4f})")
    report.append(f"4. **Best Return:** {best_return[0].upper()} ({best_return[1]['total_return']*100:.1f}%)")
    report.append("")
    
    # Plots
    report.append("## Visualizations\n")
    report.append("### Cumulative Returns\n")
    report.append("![Cumulative Returns](plots/cumulative_returns.png)\n")
    report.append("### Metrics Comparison\n")
    report.append("![Metrics](plots/metrics_comparison.png)\n")
    report.append("### Risk-Return Profile\n")
    report.append("![Risk-Return](plots/risk_return.png)\n")
    report.append("### Drawdown\n")
    report.append("![Drawdown](plots/drawdown.png)\n")
    
    # Comparison with multi-period
    report.append("## Comparison Note\n")
    report.append("This single-period backtest uses the entire dataset without time-period splitting.")
    report.append("The test period (2023-05 ~ 2025-11) represents the most recent market conditions.")
    report.append("")
    
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write('\n'.join(report))
    print(f"  Saved: REPORT.md")


def main():
    parser = argparse.ArgumentParser(description='Plot single-period backtest results')
    parser.add_argument('--result-dir', type=str, default='./results_single',
                       help='Directory containing results')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    plot_dir = result_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating Plots for Single Period Backtest")
    print("=" * 60)
    
    results, cumulative_df = load_results(result_dir)
    
    print("\nGenerating plots...")
    plot_cumulative_returns(results, cumulative_df, plot_dir)
    plot_metrics_comparison(results, plot_dir)
    plot_risk_return(results, plot_dir)
    plot_drawdown(results, cumulative_df, plot_dir)
    
    print("\nGenerating report...")
    generate_report(results, result_dir)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
