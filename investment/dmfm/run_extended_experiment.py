#!/usr/bin/env python
"""
Multi-period backtest with extended fundamental + FF-style factors.

Usage:
    python -m dmfm.run_extended_experiment --data-npz ./dmfm_data_extended.npz --output-dir ./results_extended
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmfm.data import DMFMDataset
from dmfm.model import DMFM
from dmfm.config import ModelConfig, LossConfig
from dmfm.backtest import evaluate_model
from dmfm.baselines import LinearFactorModel, MLPFactorModel, MLPGATModel, MomentumModel, MeanReversionModel


def load_extended_data(npz_path, device):
    """Load extended data with fundamental + FF factors."""
    print("Loading extended data...")
    data = np.load(npz_path)
    
    features_np = data["features"]
    forward_returns = {int(k.split("_")[2][:-1]): torch.from_numpy(v).float() 
                       for k, v in data.items() if k.startswith("forward_return_")}
    industry_mask = torch.from_numpy(data["industry_mask"]).bool()
    universe_mask = torch.from_numpy(data["universe_mask"]).bool() if "universe_mask" in data else None
    dates = data["dates"]
    
    T, N, F = features_np.shape
    print(f"  Time steps: {T}")
    print(f"  Stocks: {N}")
    print(f"  Features: {F}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    features = torch.from_numpy(features_np).float()
    return features, forward_returns, industry_mask, universe_mask, dates


def split_data_by_period(features, forward_returns, industry_mask, universe_mask, dates, periods):
    """Split data into multiple time periods."""
    T = features.shape[0]
    splits = {}
    
    for period_name, (start_date, end_date) in periods.items():
        # Find indices by comparing dates (partial match)
        start_idx = None
        end_idx = None
        for i, d in enumerate(dates):
            if d >= start_date and start_idx is None:
                start_idx = i
            if d <= end_date:
                end_idx = i
        
        if start_idx is None or end_idx is None:
            print(f"  Warning: Period {period_name} dates not found, skipping")
            continue
        
        end_idx = end_idx + 1
        
        feat_split = features[start_idx:end_idx]
        fwd_ret_split = {h: forward_returns[h][start_idx:end_idx] 
                         for h in forward_returns}
        ind_mask_split = industry_mask[start_idx:end_idx] if industry_mask is not None else None
        univ_mask_split = universe_mask[start_idx:end_idx] if universe_mask is not None else None
        
        splits[period_name] = {
            'features': feat_split,
            'forward_returns': fwd_ret_split,
            'industry_mask': ind_mask_split,
            'universe_mask': univ_mask_split,
            'date_range': (dates[start_idx], dates[end_idx-1]),
            'n_days': end_idx - start_idx
        }
    
    return splits


def train_and_evaluate_model(
    model_class,
    model_name,
    train_data,
    val_data,
    test_data,
    config,
    device,
    horizons=[5, 10, 20]
):
    """Train and evaluate a single model."""
    
    if model_name == "momentum":
        model = MomentumModel()
        test_ic, test_icir, test_factor_return, metrics = evaluate_model(
            model, test_data, config, device
        )
        return {
            'test_ic': test_ic,
            'test_icir': test_icir,
            'test_factor_return': test_factor_return,
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'total_return': metrics.get('total_return', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
        }
    
    elif model_name == "reversal":
        model = MeanReversionModel()
        test_ic, test_icir, test_factor_return, metrics = evaluate_model(
            model, test_data, config, device
        )
        return {
            'test_ic': test_ic,
            'test_icir': test_icir,
            'test_factor_return': test_factor_return,
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'total_return': metrics.get('total_return', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
        }
    
    elif model_name == "linear":
        model = LinearFactorModel(feature_dim=train_data.features.shape[2])
    elif model_name == "mlp":
        model = MLPFactorModel(feature_dim=train_data.features.shape[2])
    elif model_name == "mlp_gat":
        model = MLPGATModel(feature_dim=train_data.features.shape[2])
    elif model_name == "dmfm":
        model_cfg = ModelConfig(feature_dim=train_data.features.shape[2], hidden_dim=128, horizons=horizons)
        model = DMFM(model_cfg)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # Train with early stopping
    best_val_ic = -np.inf
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.get('epochs', 30)):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            features = batch["features"].to(device)
            fwd_returns_batch = batch["forward_returns"]
            fwd_returns = {int(h): fwd_returns_batch[str(h)].to(device) 
                          for h in horizons}
            industry_mask = batch.get("industry_mask", None)
            if industry_mask is not None:
                industry_mask = industry_mask.to(device)
            universe_mask = batch.get("universe_mask", None)
            if universe_mask is not None:
                universe_mask = universe_mask.to(device)
            
            optimizer.zero_grad()
            
            if model_name == "dmfm":
                outputs = model(features, industry_mask, universe_mask)
                loss = 0
                for h in horizons:
                    pred = outputs[f"factors_{h}d"]
                    target = fwd_returns[h]
                    valid = ~torch.isnan(target)
                    if valid.sum() > 0:
                        loss += nn.MSELoss()(pred[valid], target[valid])
                loss = loss / len(horizons)
            else:
                outputs = model(features)
                target = fwd_returns[5]
                valid = ~torch.isnan(target)
                if valid.sum() > 0:
                    loss = nn.MSELoss()(outputs[valid], target[valid])
                else:
                    loss = torch.tensor(0.0, device=device)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_ics = []
            for batch in val_loader:
                features = batch["features"].to(device)
                fwd_returns_batch = batch["forward_returns"]
                fwd_returns_5d = fwd_returns_batch["5"].to(device)
                industry_mask = batch.get("industry_mask", None)
                if industry_mask is not None:
                    industry_mask = industry_mask.to(device)
                universe_mask = batch.get("universe_mask", None)
                if universe_mask is not None:
                    universe_mask = universe_mask.to(device)
                
                if model_name == "dmfm":
                    outputs = model(features, industry_mask, universe_mask)
                    pred = outputs["factors_5d"]
                else:
                    pred = model(features)
                
                # Compute IC
                pred_flat = pred.flatten().cpu().numpy()
                target_flat = fwd_returns_5d.flatten().cpu().numpy()
                valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
                if valid_mask.sum() > 0:
                    ic = np.corrcoef(pred_flat[valid_mask], target_flat[valid_mask])[0, 1]
                    if not np.isnan(ic):
                        val_ics.append(ic)
            
            val_ic = np.mean(val_ics) if val_ics else -np.inf
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [{model_name}] Epoch {epoch+1}/{config.get('epochs', 30)} | Val IC: {val_ic:.4f}")
            
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
                break
    
    # Test
    test_ic, test_icir, test_factor_return, metrics = evaluate_model(
        model, test_loader, config, device
    )
    
    return {
        'test_ic': test_ic,
        'test_icir': test_icir,
        'test_factor_return': test_factor_return,
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'total_return': metrics.get('total_return', 0),
        'max_drawdown': metrics.get('max_drawdown', 0),
        'win_rate': metrics.get('win_rate', 0),
    }


def main():
    parser = argparse.ArgumentParser(description='Run extended multi-period experiment')
    parser.add_argument('--data-npz', type=str, default='./dmfm_data_extended.npz',
                       help='Path to extended data NPZ file')
    parser.add_argument('--output-dir', type=str, default='./results_extended',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Multi-Period Extended Experiment")
    print("=" * 60)
    
    # Load data
    features, forward_returns, industry_mask, universe_mask, dates = load_extended_data(
        args.data_npz, device
    )
    
    # Define periods
    periods = {
        'early': ('2015-08-01', '2017-08-31'),
        'middle': ('2018-08-01', '2020-08-31'),
        'recent': ('2022-08-01', '2024-08-31'),
    }
    
    print(f"\nSplitting data into periods...")
    period_splits = split_data_by_period(features, forward_returns, industry_mask, universe_mask, dates, periods)
    
    horizons = [5, 10, 20]
    config = {
        'epochs': args.epochs,
        'top_k': 10,
        'bottom_k': 10,
        'eval_horizon': 5,
    }
    
    all_results = {
        'experiment_timestamp': pd.Timestamp.now().isoformat(),
        'experiment_type': 'multi_period_extended',
        'config': config,
        'models': ['dmfm', 'linear', 'mlp', 'mlp_gat', 'momentum', 'reversal'],
        'periods': {},
        'results': {}
    }
    
    # Run experiments for each period
    for period_name, period_data in period_splits.items():
        print(f"\n{'='*60}")
        print(f"Period: {period_name.upper()}")
        print(f"{'='*60}")
        print(f"  Date range: {period_data['date_range'][0]} ~ {period_data['date_range'][1]}")
        print(f"  Total days: {period_data['n_days']}")
        
        # Split into train/val/test (60/20/20)
        T = period_data['n_days']
        train_end = int(0.6 * T)
        val_end = int(0.8 * T)
        
        train_dataset = DMFMDataset(
            period_data['features'][:train_end],
            period_data['industry_mask'][:train_end] if period_data['industry_mask'] is not None else None,
            {h: period_data['forward_returns'][h][:train_end] for h in horizons},
            period_data['universe_mask'][:train_end] if period_data['universe_mask'] is not None else None,
        )
        
        val_dataset = DMFMDataset(
            period_data['features'][train_end:val_end],
            period_data['industry_mask'][train_end:val_end] if period_data['industry_mask'] is not None else None,
            {h: period_data['forward_returns'][h][train_end:val_end] for h in horizons},
            period_data['universe_mask'][train_end:val_end] if period_data['universe_mask'] is not None else None,
        )
        
        test_dataset = DMFMDataset(
            period_data['features'][val_end:],
            period_data['industry_mask'][val_end:] if period_data['industry_mask'] is not None else None,
            {h: period_data['forward_returns'][h][val_end:] for h in horizons},
            period_data['universe_mask'][val_end:] if period_data['universe_mask'] is not None else None,
        )
        
        period_results = {}
        
        for model_name in all_results['models']:
            print(f"  Training: {model_name}...", end=' ', flush=True)
            
            result = train_and_evaluate_model(
                None,
                model_name,
                train_dataset,
                val_dataset,
                test_dataset,
                config,
                device,
                horizons
            )
            
            period_results[model_name] = result
            print(f"IC={result['test_ic']:.4f}, Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']*100:.1f}%")
        
        all_results['periods'][period_name] = period_results
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("SUMMARY BY PERIOD")
    print(f"{'='*60}")
    
    for model_name in all_results['models']:
        results_dict = {}
        all_results['results'][model_name] = {}
        
        for period_name in period_splits.keys():
            period_result = all_results['periods'][period_name][model_name]
            results_dict[period_name] = period_result['sharpe_ratio']
            all_results['results'][model_name][period_name] = period_result
        
        avg_sharpe = np.mean(list(results_dict.values()))
        all_results['results'][model_name]['avg_sharpe'] = avg_sharpe
        
        print(f"{model_name:12} | Early: {results_dict.get('early', 0):6.2f} | "
              f"Middle: {results_dict.get('middle', 0):6.2f} | "
              f"Recent: {results_dict.get('recent', 0):6.2f} | "
              f"Avg: {avg_sharpe:6.2f}")
    
    # Save results
    with open(output_dir / 'extended_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir / 'extended_experiment_results.json'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
