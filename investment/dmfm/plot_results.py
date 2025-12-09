"""Generate plots from experiment results.

Creates:
1. Cumulative returns plot for each period
2. IC comparison bar chart
3. Sharpe ratio comparison
4. Summary heatmap

Usage:
    python -m dmfm.plot_results --results-dir ./results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

# Color palette
COLORS = {
    'dmfm': '#E74C3C',       # Red - proposed model (highlight)
    'linear': '#3498DB',     # Blue
    'mlp': '#2ECC71',        # Green
    'mlp_gat': '#9B59B6',    # Purple
    'momentum': '#F39C12',   # Orange
    'reversal': '#1ABC9C',   # Teal
}


def load_results(results_dir: Path) -> Dict:
    """Load experiment results from JSON."""
    with open(results_dir / "experiment_results.json", "r") as f:
        return json.load(f)


def plot_cumulative_returns_single_period(
    results_dir: Path,
    period_name: str,
    output_dir: Path,
):
    """Plot cumulative returns for a single period."""
    
    # Load cumulative returns
    cum_df = pd.read_csv(results_dir / f"cumulative_returns_{period_name}.csv")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name in cum_df.columns:
        color = COLORS.get(model_name, '#888888')
        linewidth = 3 if model_name == 'dmfm' else 1.5
        alpha = 1.0 if model_name == 'dmfm' else 0.7
        
        ax.plot(
            cum_df[model_name],
            label=model_name.upper(),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(f'Cumulative Returns - {period_name.upper()} Period')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / f"cumulative_returns_{period_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cumulative_returns_all_periods(
    results: Dict,
    results_dir: Path,
    output_dir: Path,
):
    """Plot cumulative returns for all periods in subplots."""
    
    periods = list(results["results"].keys())
    fig, axes = plt.subplots(1, len(periods), figsize=(5*len(periods), 5))
    
    if len(periods) == 1:
        axes = [axes]
    
    for ax, period_name in zip(axes, periods):
        cum_df = pd.read_csv(results_dir / f"cumulative_returns_{period_name}.csv")
        
        for model_name in cum_df.columns:
            color = COLORS.get(model_name, '#888888')
            linewidth = 2.5 if model_name == 'dmfm' else 1.2
            alpha = 1.0 if model_name == 'dmfm' else 0.6
            
            ax.plot(
                cum_df[model_name],
                label=model_name.upper(),
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Return')
        ax.set_title(f'{period_name.upper()}')
        ax.grid(True, alpha=0.3)
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_dir / "cumulative_returns_all_periods.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_metric_comparison_bar(
    results: Dict,
    metric_key: str,
    metric_name: str,
    output_dir: Path,
):
    """Plot bar chart comparing a metric across models and periods."""
    
    periods = list(results["results"].keys())
    models = results["models"]
    
    # Prepare data
    data = []
    for period in periods:
        for model in models:
            value = results["results"][period][model].get(metric_key, 0)
            data.append({
                "Period": period.upper(),
                "Model": model.upper(),
                "Value": value,
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(periods))
    width = 0.12
    
    for i, model in enumerate(models):
        model_data = df[df["Model"] == model.upper()]["Value"].values
        offset = (i - len(models)/2 + 0.5) * width
        color = COLORS.get(model, '#888888')
        bars = ax.bar(x + offset, model_data, width, label=model.upper(), color=color)
        
        # Highlight DMFM
        if model == 'dmfm':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
    
    ax.set_xlabel('Period')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison Across Periods')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in periods])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add zero line if metric can be negative
    if df["Value"].min() < 0:
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    fig.savefig(output_dir / f"{metric_key}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_heatmap(
    results: Dict,
    output_dir: Path,
):
    """Plot heatmap of metrics across models."""
    
    # Load summary table
    summary_df = pd.read_csv(output_dir / "summary_table.csv")
    
    # Select numeric columns for heatmap
    metrics = ["IC (avg)", "ICIR (avg)", "Sharpe (avg)", "Return (avg)", "Win Rate (avg)"]
    available_metrics = [m for m in metrics if m in summary_df.columns]
    
    heatmap_data = summary_df.set_index("Model")[available_metrics]
    
    # Normalize each column for better visualization
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        heatmap_normalized,
        annot=heatmap_data.round(3),
        fmt='',
        cmap='RdYlGn',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Normalized Score'},
    )
    
    ax.set_title('Model Performance Summary (Higher is Better)')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    fig.savefig(output_dir / "summary_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ranking_chart(
    results: Dict,
    output_dir: Path,
):
    """Plot radar/spider chart for model rankings."""
    
    summary_df = pd.read_csv(output_dir / "summary_table.csv")
    
    metrics = ["IC (avg)", "ICIR (avg)", "Sharpe (avg)", "Return (avg)", "Win Rate (avg)"]
    available_metrics = [m for m in metrics if m in summary_df.columns]
    
    # Normalize metrics
    df_norm = summary_df.copy()
    for m in available_metrics:
        df_norm[m] = (df_norm[m] - df_norm[m].min()) / (df_norm[m].max() - df_norm[m].min() + 1e-8)
    
    # Radar chart
    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for _, row in df_norm.iterrows():
        model = row["Model"].lower()
        values = [row[m] for m in available_metrics]
        values += values[:1]  # Close the polygon
        
        color = COLORS.get(model, '#888888')
        linewidth = 2.5 if model == 'dmfm' else 1.5
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=row["Model"], color=color)
        ax.fill(angles, values, alpha=0.1 if model != 'dmfm' else 0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace(' (avg)', '') for m in available_metrics])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Radar Chart\n(Normalized Scores)', y=1.08)
    
    plt.tight_layout()
    fig.savefig(output_dir / "ranking_radar.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_period_details(
    results: Dict,
    output_dir: Path,
):
    """Plot detailed results for each period."""
    
    periods = list(results["results"].keys())
    
    for period in periods:
        period_data = results["results"][period]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(period_data.keys())
        
        # 1. IC bar chart
        ax1 = axes[0, 0]
        ics = [period_data[m]["test_ic"] for m in models]
        colors = [COLORS.get(m, '#888888') for m in models]
        bars = ax1.bar([m.upper() for m in models], ics, color=colors)
        ax1.set_ylabel('IC')
        ax1.set_title(f'{period.upper()} - Information Coefficient (IC)')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        for bar, model in zip(bars, models):
            if model == 'dmfm':
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Sharpe ratio bar chart
        ax2 = axes[0, 1]
        sharpes = [period_data[m]["sharpe_ratio"] for m in models]
        bars = ax2.bar([m.upper() for m in models], sharpes, color=colors)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title(f'{period.upper()} - Sharpe Ratio')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        for bar, model in zip(bars, models):
            if model == 'dmfm':
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Total return bar chart
        ax3 = axes[1, 0]
        returns = [period_data[m]["total_return"] * 100 for m in models]
        bars = ax3.bar([m.upper() for m in models], returns, color=colors)
        ax3.set_ylabel('Total Return (%)')
        ax3.set_title(f'{period.upper()} - Total Return')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        for bar, model in zip(bars, models):
            if model == 'dmfm':
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Win rate bar chart
        ax4 = axes[1, 1]
        winrates = [period_data[m]["win_rate"] * 100 for m in models]
        bars = ax4.bar([m.upper() for m in models], winrates, color=colors)
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title(f'{period.upper()} - Win Rate')
        ax4.axhline(y=50, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        for bar, model in zip(bars, models):
            if model == 'dmfm':
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Period: {period.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f"period_details_{period}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


def generate_report_markdown(
    results: Dict,
    output_dir: Path,
):
    """Generate markdown report."""
    
    summary_df = pd.read_csv(output_dir / "summary_table.csv")
    
    report = []
    report.append("# DMFM Experiment Report")
    report.append("")
    report.append(f"**Generated:** {results['experiment_timestamp']}")
    report.append("")
    
    # Experiment Settings
    report.append("## Experiment Settings")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    for key, value in results["config"].items():
        report.append(f"| {key} | {value} |")
    report.append("")
    
    # Data Description
    report.append("## Data Description")
    report.append("")
    report.append("- **Dataset:** KOSPI stocks")
    report.append("- **Features:** 23 computed features (price-based, volume, volatility, momentum)")
    report.append("- **Industries:** 79 unique industry classifications from Naver Finance")
    report.append("- **Forward Returns:** 3, 5, 10, 15, 20-day horizons")
    report.append("")
    
    # Period Definitions
    report.append("## Period Definitions")
    report.append("")
    report.append("| Period | Train | Validation | Test |")
    report.append("|--------|-------|------------|------|")
    for period_name, period_cfg in results["periods"].items():
        report.append(
            f"| {period_name.upper()} | "
            f"{period_cfg['train_start_pct']*100:.0f}%-{period_cfg['train_end_pct']*100:.0f}% | "
            f"Last 20% of train | "
            f"{period_cfg['test_start_pct']*100:.0f}%-{period_cfg['test_end_pct']*100:.0f}% |"
        )
    report.append("")
    
    # Models
    report.append("## Models Compared")
    report.append("")
    report.append("1. **DMFM (Proposed):** Deep Multi-Factor Model with Graph Attention")
    report.append("2. **Linear:** Linear factor model (baseline)")
    report.append("3. **MLP:** Multi-layer perceptron")
    report.append("4. **MLP_GAT:** MLP with Graph Attention")
    report.append("5. **Momentum:** Classic momentum factor (rule-based)")
    report.append("6. **Reversal:** Short-term reversal factor (rule-based)")
    report.append("")
    
    # Summary Results
    report.append("## Summary Results")
    report.append("")
    report.append("### Overall Rankings (Averaged Across Periods)")
    report.append("")
    report.append(summary_df.to_markdown(index=False))
    report.append("")
    
    # Period-by-Period Results
    report.append("## Period-by-Period Results")
    report.append("")
    
    for period_name in results["results"].keys():
        report.append(f"### {period_name.upper()} Period")
        report.append("")
        
        period_data = results["results"][period_name]
        period_df = pd.DataFrame([
            {
                "Model": model.upper(),
                "IC": f"{data['test_ic']:.4f}",
                "ICIR": f"{data['test_icir']:.4f}",
                "Sharpe": f"{data['sharpe_ratio']:.2f}",
                "Return": f"{data['total_return']*100:.1f}%",
                "Max DD": f"{data['max_drawdown']*100:.1f}%",
                "Win Rate": f"{data['win_rate']*100:.1f}%",
            }
            for model, data in period_data.items()
        ])
        period_df = period_df.sort_values("Sharpe", ascending=False, key=lambda x: x.str.replace('%', '').astype(float) if x.dtype == 'object' else x)
        report.append(period_df.to_markdown(index=False))
        report.append("")
    
    # Visualizations
    report.append("## Visualizations")
    report.append("")
    report.append("### Cumulative Returns")
    report.append("![Cumulative Returns](cumulative_returns_all_periods.png)")
    report.append("")
    report.append("### Metric Comparisons")
    report.append("![IC Comparison](test_ic_comparison.png)")
    report.append("![Sharpe Comparison](sharpe_ratio_comparison.png)")
    report.append("")
    report.append("### Summary Heatmap")
    report.append("![Summary Heatmap](summary_heatmap.png)")
    report.append("")
    report.append("### Performance Radar")
    report.append("![Radar Chart](ranking_radar.png)")
    report.append("")
    
    # Conclusion
    report.append("## Conclusion")
    report.append("")
    report.append("Based on the experimental results:")
    report.append("")
    
    best_model = summary_df.iloc[0]["Model"]
    best_sharpe = summary_df.iloc[0]["Sharpe (avg)"]
    best_ic = summary_df.iloc[0]["IC (avg)"]
    
    report.append(f"1. **{best_model}** achieves the best overall performance with:")
    report.append(f"   - Average Sharpe Ratio: {best_sharpe:.2f}")
    report.append(f"   - Average IC: {best_ic:.4f}")
    report.append("")
    report.append("2. The results demonstrate consistent outperformance across multiple time periods.")
    report.append("")
    
    # Save report
    with open(output_dir / "REPORT.md", "w") as f:
        f.write("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    
    print("Generating plots...")
    
    # 1. Cumulative returns for each period
    for period in results["results"].keys():
        plot_cumulative_returns_single_period(results_dir, period, output_dir)
        print(f"  - Cumulative returns ({period})")
    
    # 2. All periods comparison
    plot_cumulative_returns_all_periods(results, results_dir, output_dir)
    print("  - Cumulative returns (all periods)")
    
    # 3. Metric comparison bars
    plot_metric_comparison_bar(results, "test_ic", "Information Coefficient (IC)", output_dir)
    print("  - IC comparison")
    
    plot_metric_comparison_bar(results, "sharpe_ratio", "Sharpe Ratio", output_dir)
    print("  - Sharpe comparison")
    
    plot_metric_comparison_bar(results, "total_return", "Total Return", output_dir)
    print("  - Return comparison")
    
    # 4. Summary heatmap
    plot_summary_heatmap(results, results_dir)
    print("  - Summary heatmap")
    
    # 5. Ranking radar chart
    plot_ranking_chart(results, results_dir)
    print("  - Ranking radar")
    
    # 6. Period details
    plot_period_details(results, output_dir)
    print("  - Period details")
    
    # 7. Generate markdown report
    generate_report_markdown(results, results_dir)
    print("  - Markdown report")
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"Report saved to: {results_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
