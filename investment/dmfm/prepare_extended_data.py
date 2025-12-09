#!/usr/bin/env python
"""
Prepare extended data with fundamental and FF-style factors.

Adds to the existing technical factors:
1. Fundamental factors: PER, PBR, EPS, BPS, dividend yield
2. Fama-French style factors: Size (log market cap), Value (B/M), etc.

Usage:
    python -m dmfm.prepare_extended_data \
        --price-dir ./kospi \
        --fundamental-csv ./kospi_fundamental/fundamental_data.csv \
        --industry-csv ./kospi_industry.csv \
        --output ./dmfm_data_extended.npz
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_price_data(data_dir: Path) -> pd.DataFrame:
    """Load all KOSPI stock price data."""
    print("Loading price data...")
    all_dfs = []
    csv_files = list(data_dir.glob("*.csv"))
    
    for csv_file in tqdm(csv_files, desc="Loading stocks"):
        symbol = csv_file.stem
        if not symbol.isdigit():
            continue
        
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df["symbol"] = symbol.zfill(6)
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    
    print(f"  Total rows: {len(combined):,}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Unique symbols: {combined['symbol'].nunique()}")
    
    return combined


def load_fundamental_data(csv_path: Path) -> pd.DataFrame:
    """Load fundamental data and forward-fill to daily."""
    print("\nLoading fundamental data...")
    fund_df = pd.read_csv(csv_path, parse_dates=['date'])
    fund_df['symbol'] = fund_df['symbol'].astype(str).str.zfill(6)
    
    print(f"  Rows: {len(fund_df):,}")
    print(f"  Date range: {fund_df['date'].min()} to {fund_df['date'].max()}")
    
    return fund_df


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features (same as before)."""
    print("\nComputing technical features...")
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    prev_close = g["close"].shift(1)
    
    # 1-day return
    df["ret_1d"] = g["close"].pct_change()
    
    # Multi-period returns
    for w in [5, 10, 20, 60]:
        df[f"ret_{w}d"] = g["close"].pct_change(w)
        df[f"vol_{w}d"] = g["ret_1d"].transform(lambda s, win=w: s.rolling(win, min_periods=1).std())
    
    # Price to MA
    for w in [5, 10, 20, 60]:
        ma = g["close"].transform(lambda s, win=w: s.rolling(win, min_periods=1).mean())
        df[f"price_to_ma_{w}d"] = close / (ma + 1e-9) - 1
    
    # MACD
    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = (ema12 - ema26) / (close + 1e-9)
    
    # RSI
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
    
    # Amihud
    df["dollar_vol"] = close * volume
    df["amihud"] = (df["ret_1d"].abs() / (df["dollar_vol"] + 1e-9))
    df["amihud"] = df["amihud"].replace([np.inf, -np.inf], np.nan).clip(-10, 10)
    
    # Bollinger position
    ma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    std20 = g["close"].transform(lambda s: s.rolling(20, min_periods=1).std())
    df["bb_pos_20d"] = (close - ma20) / (2 * std20 + 1e-9)
    
    return df


def merge_fundamental_features(
    price_df: pd.DataFrame, 
    fund_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge fundamental data with price data.
    Forward-fill fundamental data (monthly -> daily).
    """
    print("\nMerging fundamental features...")
    
    # Fundamental columns to merge
    fund_cols = ['per', 'pbr', 'eps', 'bps', 'dps', 'div_yield', 'market_cap', 'shares_outstanding']
    available_cols = [c for c in fund_cols if c in fund_df.columns]
    
    # Create a merge key
    fund_df = fund_df[['symbol', 'date'] + available_cols].copy()
    
    # Merge using asof (forward fill fundamentals to daily data)
    price_df = price_df.sort_values(['symbol', 'date'])
    fund_df = fund_df.sort_values(['symbol', 'date'])
    
    # For each symbol, forward-fill fundamental data
    merged_dfs = []
    
    for symbol in tqdm(price_df['symbol'].unique(), desc="Merging fundamentals"):
        price_sym = price_df[price_df['symbol'] == symbol].copy()
        fund_sym = fund_df[fund_df['symbol'] == symbol].copy()
        
        if fund_sym.empty:
            # No fundamental data for this symbol, add NaN columns
            for col in available_cols:
                price_sym[col] = np.nan
        else:
            # Merge asof (forward fill)
            price_sym = pd.merge_asof(
                price_sym.sort_values('date'),
                fund_sym.sort_values('date'),
                on='date',
                by='symbol',
                direction='backward',
            )
        
        merged_dfs.append(price_sym)
    
    merged = pd.concat(merged_dfs, ignore_index=True)
    merged = merged.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    print(f"  Merged rows: {len(merged):,}")
    
    return merged


def compute_ff_style_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Fama-French style factors.
    """
    print("\nComputing Fama-French style factors...")
    df = df.copy()
    
    # 1. Size factor: log(market cap)
    if 'market_cap' in df.columns:
        df['log_market_cap'] = np.log(df['market_cap'].clip(lower=1e6))
    
    # 2. Value factor: Book-to-Market (B/M) = 1/PBR
    if 'pbr' in df.columns:
        df['book_to_market'] = 1 / df['pbr'].clip(lower=0.01)
        df['book_to_market'] = df['book_to_market'].clip(upper=100)  # Cap extreme values
    
    # 3. Profitability (using EPS/Price as proxy for ROE)
    if 'eps' in df.columns and 'close' in df.columns:
        df['earnings_yield'] = df['eps'] / df['close'].clip(lower=1)
        df['earnings_yield'] = df['earnings_yield'].clip(-1, 1)
    
    # 4. Investment factor: Asset growth proxy (use market cap growth)
    if 'market_cap' in df.columns:
        g = df.groupby('symbol', sort=False)
        df['market_cap_growth_1y'] = g['market_cap'].pct_change(252)  # ~1 year
        df['market_cap_growth_1y'] = df['market_cap_growth_1y'].clip(-2, 2)
    
    # 5. Dividend Yield (already have it)
    if 'div_yield' in df.columns:
        df['div_yield'] = df['div_yield'].clip(0, 30)  # Cap at 30%
    
    # 6. PER inverse (earnings yield alternative)
    if 'per' in df.columns:
        df['per_inverse'] = 1 / df['per'].clip(lower=1)
        df['per_inverse'] = df['per_inverse'].clip(upper=1)
    
    return df


def compute_forward_returns(
    df: pd.DataFrame, 
    horizons: List[int] = [3, 5, 10, 15, 20],
    max_return: float = 0.5,
) -> pd.DataFrame:
    """Compute forward returns with corporate event filtering."""
    print("\nComputing forward returns...")
    print(f"  Filtering returns with |r| > {max_return*100:.0f}% (corporate events)")
    
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    
    df["daily_ret"] = g["close"].pct_change()
    df["is_corporate_event"] = df["daily_ret"].abs() > max_return
    
    for h in horizons:
        fwd_ret = g["close"].transform(lambda s: s.shift(-h) / s - 1)
        
        has_event_in_window = False
        for i in range(1, h + 1):
            has_event_in_window |= g["is_corporate_event"].shift(-i).fillna(False)
        
        fwd_ret = fwd_ret.where(
            (fwd_ret.abs() <= max_return * h / 5) & (~has_event_in_window),
            np.nan
        )
        
        df[f"forward_return_{h}d"] = fwd_ret
    
    # Count filtered
    for h in horizons:
        n_filtered = df[f"forward_return_{h}d"].isna().sum() - df["close"].isna().sum()
        print(f"  {h}d: {n_filtered:,} extreme returns filtered")
    
    df = df.drop(columns=["daily_ret", "is_corporate_event"])
    
    return df


def prepare_tensors(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: List[int] = [3, 5, 10, 15, 20],
    industry_csv: Optional[Path] = None,
) -> Dict:
    """Prepare tensors for DMFM model."""
    print("\nPreparing tensors...")
    
    # Load industry data if provided
    if industry_csv:
        industry_df = pd.read_csv(industry_csv)
        industry_df["symbol"] = industry_df["symbol"].astype(str).str.zfill(6)
        df = df.merge(industry_df[["symbol", "industry_code"]], on="symbol", how="left")
    
    dates = sorted(df["date"].unique())
    symbols = sorted(df["symbol"].unique())
    
    T = len(dates)
    N = len(symbols)
    F = len(feature_cols)
    
    print(f"  Time steps: {T}")
    print(f"  Stocks: {N}")
    print(f"  Features: {F}")
    
    date_to_idx = {d: i for i, d in enumerate(dates)}
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    
    # Initialize arrays
    features = np.full((T, N, F), np.nan, dtype=np.float32)
    forward_returns = {h: np.full((T, N), np.nan, dtype=np.float32) for h in horizons}
    industry_codes = np.full((T, N), -1, dtype=np.int32)
    
    # Fill arrays
    print("  Filling arrays...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        t = date_to_idx.get(row["date"])
        n = symbol_to_idx.get(row["symbol"])
        
        if t is None or n is None:
            continue
        
        # Features
        for f_idx, col in enumerate(feature_cols):
            if col in row and pd.notna(row[col]):
                features[t, n, f_idx] = row[col]
        
        # Forward returns
        for h in horizons:
            col = f"forward_return_{h}d"
            if col in row and pd.notna(row[col]):
                forward_returns[h][t, n] = row[col]
        
        # Industry
        if "industry_code" in row and pd.notna(row["industry_code"]):
            industry_codes[t, n] = int(row["industry_code"])
    
    # Z-score features cross-sectionally
    print("  Z-scoring features...")
    for t in range(T):
        for f_idx in range(F):
            col = features[t, :, f_idx]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                mean = np.nanmean(col[valid])
                std = np.nanstd(col[valid])
                if std > 1e-8:
                    features[t, valid, f_idx] = (col[valid] - mean) / std
    
    # Clip extremes
    features = np.clip(features, -5, 5)
    features = np.nan_to_num(features, nan=0.0)
    
    # Build industry mask
    print("  Building industry mask...")
    unique_industries = np.unique(industry_codes[industry_codes >= 0])
    industry_mask = np.zeros((N, N), dtype=bool)
    
    # Use last day's industry assignment
    last_day_industries = industry_codes[-1]
    for ind in unique_industries:
        members = np.where(last_day_industries == ind)[0]
        for i in members:
            for j in members:
                industry_mask[i, j] = True
    
    # Add self-loops
    np.fill_diagonal(industry_mask, True)
    
    return {
        "features": features,
        "forward_returns": forward_returns,
        "industry_mask": industry_mask,
        "dates": np.array([str(d)[:10] for d in dates]),
        "symbols": np.array(symbols),
        "feature_names": np.array(feature_cols),
        "horizons": np.array(horizons),
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare extended data with fundamental factors')
    parser.add_argument('--price-dir', type=str, default='./kospi',
                       help='Directory with price CSVs')
    parser.add_argument('--fundamental-csv', type=str, default='./kospi_fundamental/fundamental_data.csv',
                       help='Fundamental data CSV')
    parser.add_argument('--industry-csv', type=str, default='./kospi_industry.csv',
                       help='Industry mapping CSV')
    parser.add_argument('--output', type=str, default='./dmfm_data_extended.npz',
                       help='Output NPZ file')
    
    args = parser.parse_args()
    
    # Load data
    price_df = load_price_data(Path(args.price_dir))
    fund_df = load_fundamental_data(Path(args.fundamental_csv))
    
    # Compute technical features
    df = compute_technical_features(price_df)
    
    # Merge fundamental data
    df = merge_fundamental_features(df, fund_df)
    
    # Compute FF-style factors
    df = compute_ff_style_factors(df)
    
    # Compute forward returns
    df = compute_forward_returns(df)
    
    # Define feature columns (21 technical + ~8 fundamental/FF)
    technical_features = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
        "vol_5d", "vol_10d", "vol_20d", "vol_60d",
        "price_to_ma_5d", "price_to_ma_10d", "price_to_ma_20d", "price_to_ma_60d",
        "macd", "rsi",
        "volume_ratio_5d", "volume_ratio_20d",
        "atrp_14", "hl_range", "amihud", "bb_pos_20d",
    ]
    
    fundamental_features = [
        "per_inverse",       # 1/PER (earnings yield proxy)
        "book_to_market",    # 1/PBR (value factor)
        "div_yield",         # Dividend yield
        "log_market_cap",    # Size factor
        "earnings_yield",    # EPS/Price
    ]
    
    ff_features = [
        "market_cap_growth_1y",  # Investment factor proxy
    ]
    
    # Use only features that exist
    all_features = technical_features + fundamental_features + ff_features
    feature_cols = [f for f in all_features if f in df.columns]
    
    print(f"\nUsing {len(feature_cols)} features:")
    for i, f in enumerate(feature_cols):
        print(f"  {i+1}. {f}")
    
    # Prepare tensors
    data = prepare_tensors(
        df, 
        feature_cols,
        horizons=[3, 5, 10, 15, 20],
        industry_csv=Path(args.industry_csv) if args.industry_csv else None,
    )
    
    # Save
    print(f"\nSaving to {args.output}...")
    np.savez_compressed(
        args.output,
        features=data["features"],
        forward_return_3d=data["forward_returns"][3],
        forward_return_5d=data["forward_returns"][5],
        forward_return_10d=data["forward_returns"][10],
        forward_return_15d=data["forward_returns"][15],
        forward_return_20d=data["forward_returns"][20],
        industry_mask=data["industry_mask"],
        dates=data["dates"],
        symbols=data["symbols"],
        feature_names=data["feature_names"],
        horizons=data["horizons"],
    )
    
    print(f"\nDone!")
    print(f"  Shape: {data['features'].shape}")
    print(f"  Features: {len(feature_cols)}")


if __name__ == '__main__':
    main()
