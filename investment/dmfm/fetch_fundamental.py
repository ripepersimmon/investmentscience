#!/usr/bin/env python
"""
Fetch fundamental data from KRX for KOSPI stocks.

Collects:
- PER, PBR, EPS, BPS, DIV (배당수익률)
- Market Cap (시가총액)

Usage:
    python -m dmfm.fetch_fundamental --output-dir ./kospi_fundamental
"""

import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_krx_fundamental(date: str, market: str = "STK") -> pd.DataFrame:
    """
    Fetch fundamental data from KRX for a specific date.
    
    Args:
        date: Date string in 'YYYYMMDD' format
        market: 'STK' for KOSPI, 'KSQ' for KOSDAQ
    
    Returns:
        DataFrame with fundamental data
    """
    # KRX API endpoint for PER/PBR/배당수익률
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101",
    }
    
    # PER/PBR/배당수익률 데이터
    params = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT03501",
        "locale": "ko_KR",
        "searchType": "1",
        "mktId": market,
        "trdDd": date,
    }
    
    try:
        response = requests.post(url, data=params, headers=headers, timeout=30)
        data = response.json()
        
        # API returns data in 'output' key
        if 'output' not in data or len(data['output']) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['output'])
        
        # Rename columns
        col_map = {
            'ISU_SRT_CD': 'symbol',
            'ISU_ABBRV': 'name',
            'TDD_CLSPRC': 'close',
            'PER': 'per',
            'PBR': 'pbr',
            'EPS': 'eps',
            'BPS': 'bps',
            'DPS': 'dps',
            'DVD_YLD': 'div_yield',
            'FWD_EPS': 'fwd_eps',
            'FWD_PER': 'fwd_per',
        }
        
        # Select and rename columns that exist
        available_cols = [c for c in col_map.keys() if c in df.columns]
        df = df[available_cols].rename(columns=col_map)
        
        # Convert numeric columns
        numeric_cols = ['close', 'per', 'pbr', 'eps', 'bps', 'dps', 'div_yield', 'fwd_eps', 'fwd_per']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        df['date'] = pd.to_datetime(date)
        
        return df
        
    except Exception as e:
        print(f"  Error fetching {date}: {e}")
        return pd.DataFrame()


def get_krx_market_cap(date: str, market: str = "STK") -> pd.DataFrame:
    """
    Fetch market cap data from KRX for a specific date.
    """
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd",
    }
    
    params = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT01501",
        "locale": "ko_KR",
        "mktId": market,
        "trdDd": date,
        "share": "1",
        "money": "1",
    }
    
    try:
        response = requests.post(url, data=params, headers=headers, timeout=30)
        data = response.json()
        
        if 'OutBlock_1' not in data or len(data['OutBlock_1']) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['OutBlock_1'])
        
        # Select relevant columns
        df = df[['ISU_SRT_CD', 'MKTCAP', 'LIST_SHRS']].rename(columns={
            'ISU_SRT_CD': 'symbol',
            'MKTCAP': 'market_cap',
            'LIST_SHRS': 'shares_outstanding',
        })
        
        # Convert numeric
        for col in ['market_cap', 'shares_outstanding']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        return df
        
    except Exception as e:
        return pd.DataFrame()


def get_trading_dates(start_date: str, end_date: str) -> list:
    """Get list of trading dates from KRX."""
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd",
    }
    
    params = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT03901",
        "locale": "ko_KR",
        "tboxindIdx_finder_equidx0_0": "코스피",
        "indIdx": "1",
        "indIdx2": "001",
        "codeNmindIdx_finder_equidx0_0": "코스피",
        "strtDd": start_date,
        "endDd": end_date,
        "share": "1",
        "money": "1",
        "csvxls_isNo": "false",
    }
    
    try:
        response = requests.post(url, data=params, headers=headers, timeout=30)
        data = response.json()
        
        if 'OutBlock_1' not in data:
            return []
        
        dates = [item['TRD_DD'].replace('/', '') for item in data['OutBlock_1']]
        return sorted(dates)
        
    except Exception as e:
        print(f"Error getting trading dates: {e}")
        return []


def fetch_fundamental_data(
    start_date: str,
    end_date: str,
    output_dir: Path,
    market: str = "STK",
    sample_freq: str = "M",  # M for monthly, W for weekly, D for daily
):
    """
    Fetch fundamental data for date range.
    
    Args:
        start_date: Start date in 'YYYYMMDD' format
        end_date: End date in 'YYYYMMDD' format
        output_dir: Directory to save data
        market: 'STK' for KOSPI
        sample_freq: Sampling frequency - M (monthly), W (weekly), D (daily)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching fundamental data from {start_date} to {end_date}")
    print(f"Market: {market}, Sampling: {sample_freq}")
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Sample based on frequency
    if sample_freq == 'M':
        # Get last business day of each month
        date_range = date_range.to_series().groupby(pd.Grouper(freq='M')).last()
    elif sample_freq == 'W':
        # Get last business day of each week
        date_range = date_range.to_series().groupby(pd.Grouper(freq='W')).last()
    
    dates_to_fetch = [d.strftime('%Y%m%d') for d in date_range]
    
    print(f"Total dates to fetch: {len(dates_to_fetch)}")
    
    all_data = []
    
    for date in tqdm(dates_to_fetch, desc="Fetching"):
        # Get fundamental data (PER, PBR, etc)
        df = get_krx_fundamental(date, market)
        
        if not df.empty:
            # Also get market cap data
            mcap_df = get_krx_market_cap(date, market)
            if not mcap_df.empty:
                df = df.merge(mcap_df, on='symbol', how='left')
            
            all_data.append(df)
        
        time.sleep(0.5)  # Rate limiting
    
    if not all_data:
        print("No data fetched!")
        return
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save combined data
    output_file = output_dir / "fundamental_data.csv"
    combined.to_csv(output_file, index=False)
    print(f"\nSaved combined data to {output_file}")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Unique symbols: {combined['symbol'].nunique()}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    
    # Also save per-symbol files
    symbol_dir = output_dir / "by_symbol"
    symbol_dir.mkdir(exist_ok=True)
    
    for symbol, group in combined.groupby('symbol'):
        group.to_csv(symbol_dir / f"{symbol}.csv", index=False)
    
    print(f"  Saved {combined['symbol'].nunique()} symbol files to {symbol_dir}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Fetch fundamental data from KRX')
    parser.add_argument('--start-date', type=str, default='20130101',
                       help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default='20251130',
                       help='End date (YYYYMMDD)')
    parser.add_argument('--output-dir', type=str, default='./kospi_fundamental',
                       help='Output directory')
    parser.add_argument('--market', type=str, default='STK',
                       choices=['STK', 'KSQ'], help='Market (STK=KOSPI, KSQ=KOSDAQ)')
    parser.add_argument('--freq', type=str, default='M',
                       choices=['D', 'W', 'M'], help='Sampling frequency')
    
    args = parser.parse_args()
    
    fetch_fundamental_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=Path(args.output_dir),
        market=args.market,
        sample_freq=args.freq,
    )


if __name__ == '__main__':
    main()
