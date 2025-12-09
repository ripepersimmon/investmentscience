"""Prepare data for DMFM experiment.

This script:
1. Loads raw KOSPI CSV files (per-stock)
2. Computes features using factor_builder
3. Computes forward returns
4. Saves preprocessed tensors as .npz for fast loading

Usage:
    python -m dmfm.prepare_data --data-dir ./kospi --industry-csv ./kospi_industry.csv --output ./dmfm_data.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_kospi_data(data_dir: Path, industry_csv: Path) -> pd.DataFrame:
    """Load all KOSPI stock data and merge with industry info."""
    
    print("Loading industry data...")
    industry_df = pd.read_csv(industry_csv)
    industry_df["symbol"] = industry_df["symbol"].astype(str).str.zfill(6)
    
    print("Loading stock data...")
    all_dfs = []
    csv_files = list(data_dir.glob("*.csv"))
    
    for csv_file in tqdm(csv_files, desc="Loading stocks"):
        symbol = csv_file.stem
        # Skip non-standard symbols
        if not symbol.isdigit():
            continue
        
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df["symbol"] = symbol.zfill(6)
        all_dfs.append(df)
    
    # Combine all stocks
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    
    # Merge industry
    combined = combined.merge(
        industry_df[["symbol", "industry_code"]],
        on="symbol",
        how="left"
    )
    
    print(f"  Total rows: {len(combined):,}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Unique symbols: {combined['symbol'].nunique()}")
    
    return combined


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features."""
    
    print("\nComputing features...")
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    
    # Price columns
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    prev_close = g["close"].shift(1)
    
    # 1-day return
    df["ret_1d"] = g["close"].pct_change()
    
    # Multi-period returns
    for w in [5, 10, 20, 60]:
        df[f"ret_{w}d"] = g["close"].pct_change(w)
        df[f"vol_{w}d"] = g["ret_1d"].transform(lambda s, win=w: s.rolling(win, min_periods=1).std())
    
    # Momentum indicators
    for w in [5, 10, 20, 60]:
        ma = g["close"].transform(lambda s, win=w: s.rolling(win, min_periods=1).mean())
        df[f"price_to_ma_{w}d"] = close / (ma + 1e-9) - 1
    
    # MACD-like
    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = (ema12 - ema26) / (close + 1e-9)
    
    # RSI
    delta = g["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = g["close"].transform(lambda s: pd.Series(s).ewm(span=14, adjust=False).mean().values)
    avg_loss = g["close"].transform(lambda s: pd.Series(-s.diff()).where(pd.Series(s.diff()) < 0, 0).ewm(span=14, adjust=False).mean().values)
    # Simple version
    df["rsi"] = g["ret_1d"].transform(lambda s: s.rolling(14, min_periods=1).apply(lambda x: (x > 0).sum() / len(x)))
    
    # Volume features
    df["volume_ratio_5d"] = volume / (g["volume"].transform(lambda s: s.rolling(5, min_periods=1).mean()) + 1)
    df["volume_ratio_20d"] = volume / (g["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean()) + 1)
    
    # True Range / ATR
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["atrp_14"] = df["atr_14"] / (close + 1e-9)
    
    # Range
    df["hl_range"] = (high - low) / (close + 1e-9)
    
    # Liquidity
    df["dollar_vol"] = close * volume
    df["amihud"] = (df["ret_1d"].abs() / (df["dollar_vol"] + 1e-9))
    df["amihud"] = df["amihud"].replace([np.inf, -np.inf], np.nan)
    df["amihud"] = df["amihud"].clip(-10, 10)
    
    # Bollinger position
    for w in [20]:
        ma = g["close"].transform(lambda s, win=w: s.rolling(win, min_periods=1).mean())
        std = g["close"].transform(lambda s, win=w: s.rolling(win, min_periods=1).std())
        df[f"bb_pos_{w}d"] = (close - ma) / (2 * std + 1e-9)
    
    return df


def compute_forward_returns(
    df: pd.DataFrame, 
    horizons: List[int] = [3, 5, 10, 15, 20],
    max_return: float = 0.5,  # Cap at ±50% to filter corporate events
) -> pd.DataFrame:
    """Compute forward returns for multiple horizons.
    
    Args:
        df: DataFrame with 'close' column
        horizons: List of forward return horizons
        max_return: Maximum absolute return to consider valid (filters splits/mergers)
    """
    
    print("Computing forward returns...")
    print(f"  Filtering returns with |r| > {max_return*100:.0f}% (corporate events)")
    
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    
    # First compute daily returns to detect corporate events
    df["daily_ret"] = g["close"].pct_change()
    
    # Mark days with extreme daily moves (likely stock split/merger)
    df["is_corporate_event"] = df["daily_ret"].abs() > max_return
    
    # Also mark the day before (to avoid forward return spanning the event)
    df["event_next_day"] = g["is_corporate_event"].shift(-1).fillna(False)
    
    for h in horizons:
        # Compute raw forward return
        fwd_ret = g["close"].transform(lambda s: s.shift(-h) / s - 1)
        
        # Check if any day in the forward window has a corporate event
        # This is a simplified check - mark as NaN if current day or next h days have event
        has_event_in_window = False
        for i in range(1, h + 1):
            has_event_in_window |= g["is_corporate_event"].shift(-i).fillna(False)
        
        # Filter out extreme returns
        fwd_ret = fwd_ret.where(
            (fwd_ret.abs() <= max_return * h / 5) & (~has_event_in_window),  # Scale max by horizon
            np.nan
        )
        
        df[f"forward_return_{h}d"] = fwd_ret
    
    # Count filtered
    for h in horizons:
        n_filtered = df[f"forward_return_{h}d"].isna().sum() - df["close"].isna().sum()
        print(f"  {h}d: {n_filtered} extreme returns filtered")
    
    # Clean up temp columns
    df = df.drop(columns=["daily_ret", "is_corporate_event", "event_next_day"])
    
    return df


def prepare_tensors(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: List[int] = [3, 5, 10, 15, 20],
) -> Dict:
    """Prepare tensors for DMFM model."""
    
    print("\nPreparing tensors...")
    
    # Get unique dates and symbols
    dates = sorted(df["date"].unique())
    symbols = sorted(df["symbol"].unique())
    
    date_to_idx = {d: i for i, d in enumerate(dates)}
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    
    T = len(dates)
    N = len(symbols)
    F = len(feature_cols)
    
    print(f"  Time steps: {T}")
    print(f"  Stocks: {N}")
    print(f"  Features: {F}")
    
    # Initialize arrays
    features = np.full((T, N, F), np.nan, dtype=np.float32)
    forward_returns = {h: np.full((T, N), np.nan, dtype=np.float32) for h in horizons}
    
    # Industry mask
    industry_codes = df.groupby("symbol")["industry_code"].first()
    unique_industries = sorted(industry_codes.dropna().unique())
    
    industry_mask = np.zeros((N, N), dtype=np.float32)
    for i, sym_i in enumerate(symbols):
        ind_i = industry_codes.get(sym_i)
        for j, sym_j in enumerate(symbols):
            ind_j = industry_codes.get(sym_j)
            if pd.notna(ind_i) and pd.notna(ind_j) and ind_i == ind_j:
                industry_mask[i, j] = 1.0
    
    # Universe mask (fully connected)
    universe_mask = np.ones((N, N), dtype=np.float32)
    
    print(f"  Industries: {len(unique_industries)}")
    
    # Fill tensors
    for (date, sym), group in tqdm(df.groupby(["date", "symbol"]), desc="Building tensors"):
        if date not in date_to_idx or sym not in symbol_to_idx:
            continue
        
        t = date_to_idx[date]
        n = symbol_to_idx[sym]
        
        row = group.iloc[0]
        
        # Features
        for f_idx, col in enumerate(feature_cols):
            if col in row.index and pd.notna(row[col]):
                features[t, n, f_idx] = row[col]
        
        # Forward returns
        for h in horizons:
            ret_col = f"forward_return_{h}d"
            if ret_col in row.index and pd.notna(row[ret_col]):
                forward_returns[h][t, n] = row[ret_col]
    
    # Z-score features cross-sectionally
    print("  Z-scoring features...")
    for t in range(T):
        for f_idx in range(F):
            col = features[t, :, f_idx]
            valid = np.isfinite(col)
            if valid.sum() > 1:
                mean = col[valid].mean()
                std = col[valid].std() + 1e-8
                features[t, valid, f_idx] = (col[valid] - mean) / std
    
    # Clip outliers
    features = np.clip(features, -5, 5)
    
    # Replace NaN with 0
    features = np.nan_to_num(features, nan=0.0)
    
    return {
        "features": features,
        "forward_returns": forward_returns,
        "industry_mask": industry_mask,
        "universe_mask": universe_mask,
        "dates": np.array(dates, dtype="datetime64[D]"),
        "symbols": np.array(symbols),
        "feature_cols": np.array(feature_cols),
        "horizons": np.array(horizons),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./kospi", help="KOSPI data directory")
    parser.add_argument("--industry-csv", type=str, default="./kospi_industry.csv", help="Industry CSV")
    parser.add_argument("--output", type=str, default="./dmfm_data.npz", help="Output file")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    industry_csv = Path(args.industry_csv)
    output_path = Path(args.output)
    
    # Load raw data
    df = load_kospi_data(data_dir, industry_csv)
    
    # Compute features
    df = compute_features(df)
    
    # Compute forward returns
    df = compute_forward_returns(df)
    
    # Define feature columns
    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "vol_5d", "vol_10d", "vol_20d", "vol_60d",
        "price_to_ma_5d", "price_to_ma_10d", "price_to_ma_20d", "price_to_ma_60d",
        "macd", "rsi",
        "volume_ratio_5d", "volume_ratio_20d",
        "atrp_14", "hl_range",
        "amihud",
        "bb_pos_20d",
    ]
    
    # Prepare tensors
    tensors = prepare_tensors(df, feature_cols)
    
    # Save
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        features=tensors["features"],
        forward_returns_3=tensors["forward_returns"][3],
        forward_returns_5=tensors["forward_returns"][5],
        forward_returns_10=tensors["forward_returns"][10],
        forward_returns_15=tensors["forward_returns"][15],
        forward_returns_20=tensors["forward_returns"][20],
        industry_mask=tensors["industry_mask"],
        universe_mask=tensors["universe_mask"],
        dates=tensors["dates"],
        symbols=tensors["symbols"],
        feature_cols=tensors["feature_cols"],
        horizons=tensors["horizons"],
    )
    
    print("Done!")
    print(f"  Shape: {tensors['features'].shape}")


if __name__ == "__main__":
    main()
