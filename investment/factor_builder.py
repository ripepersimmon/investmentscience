"""Feature engineering helpers for DMFM: build factors from OHLCV + fundamentals.

Inputs (long-form pandas DataFrame):
- Required: date, symbol, open, high, low, close, volume
- Optional: shares_outstanding, revenue, gross_profit, total_assets, total_equity,
  total_liabilities, net_income, operating_cf

Usage example:
```python
import pandas as pd
from factor_builder import compute_factor_panel

raw = pd.read_csv("your_prices_and_fundamentals.csv", parse_dates=["date"])
factors = compute_factor_panel(raw)
# factors contains per-date, per-symbol engineered columns ready for z-scoring
```
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["symbol", "date"]).copy()


def zscore_cross_section(df: pd.DataFrame, cols: list[str], date_col: str = "date", clip: float = 5.0) -> pd.DataFrame:
    """Cross-sectional z-score per date; useful before feeding the model."""
    out = df.copy()
    for c in cols:
        out[c + "_z"] = out.groupby(date_col)[c].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
        out[c + "_z"] = out[c + "_z"].clip(-clip, clip)
    return out


def compute_ohlcv_factors(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add OHLCV-derived factors.

    Expects columns: date, symbol, open, high, low, close, volume. Optionally shares_outstanding.
    """
    df = _group_sort(df)
    g = df.groupby("symbol", sort=False)

    # 1-day return as base
    ret1 = g[price_col].pct_change()
    df["ret_1d"] = ret1

    windows = [5, 21, 63, 126]
    for w in windows:
        df[f"ret_{w}d"] = g[price_col].pct_change(w)
        df[f"mom_{w}d"] = g[price_col].transform(lambda s, w=w: s / (s.rolling(w).mean() + 1e-9) - 1)
        df[f"vol_{w}d"] = g["ret_1d"].transform(lambda s, w=w: s.rolling(w).std())

    # True Range / ATR
    high = df["high"]
    low = df["low"]
    close = df[price_col]
    prev_close = g[price_col].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14).mean())
    df["atrp_14"] = df["atr_14"] / close

    # Range and gap
    df["hl_range"] = (df["high"] - df["low"]) / close
    df["gap"] = (df["open"] - prev_close) / prev_close

    # Liquidity / microstructure
    df["dollar_vol"] = df[price_col] * df["volume"]
    df["amihud"] = (ret1.abs() / (df["dollar_vol"] + 1e-9)).replace([np.inf, -np.inf], np.nan)
    df["turnover"] = np.nan
    if "shares_outstanding" in df.columns:
        df["turnover"] = df["volume"] / df["shares_outstanding"]

    # Roll spread approximation
    delta_p = g[price_col].diff()
    cov = delta_p * delta_p.shift(1)
    roll_cov = g[price_col].transform(lambda s: s.diff() * s.diff().shift(1))
    roll_cov = roll_cov.groupby(df["symbol"]).transform(lambda s: s.rolling(21).mean())
    df["roll_spread"] = 2 * (roll_cov.abs().pow(0.5)) / (close + 1e-9)

    return df


def compute_fundamental_factors(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add valuation/quality factors from fundamentals (with forward-fill)."""
    df = _group_sort(df)
    fundamentals = [
        "revenue",
        "gross_profit",
        "total_assets",
        "total_equity",
        "total_liabilities",
        "net_income",
        "operating_cf",
        "shares_outstanding",
    ]
    for col in fundamentals:
        if col in df.columns:
            df[col] = df.groupby("symbol")[col].ffill()

    # Market cap
    if "shares_outstanding" in df.columns:
        df["market_cap"] = df[price_col] * df["shares_outstanding"]
    else:
        df["market_cap"] = np.nan

    # Trailing 4-period sums (assumes quarterly fundamentals)
    def ttm(col: str) -> pd.Series:
        return df.groupby("symbol")[col].transform(lambda s: s.rolling(4, min_periods=1).sum())

    if "revenue" in df.columns:
        df["revenue_ttm"] = ttm("revenue")
    if "gross_profit" in df.columns:
        df["gross_profit_ttm"] = ttm("gross_profit")
    if "net_income" in df.columns:
        df["net_income_ttm"] = ttm("net_income")
    if "operating_cf" in df.columns:
        df["operating_cf_ttm"] = ttm("operating_cf")

    # Valuation ratios
    if "market_cap" in df.columns and "net_income_ttm" in df.columns:
        df["pe"] = df["market_cap"] / (df["net_income_ttm"] + 1e-9)
    if "market_cap" in df.columns and "total_equity" in df.columns:
        df["pb"] = df["market_cap"] / (df["total_equity"] + 1e-9)
    if "market_cap" in df.columns and "revenue_ttm" in df.columns:
        df["ps"] = df["market_cap"] / (df["revenue_ttm"] + 1e-9)

    # Quality & profitability
    if "net_income_ttm" in df.columns and "total_equity" in df.columns:
        df["roe"] = df["net_income_ttm"] / (df["total_equity"] + 1e-9)
    if "net_income_ttm" in df.columns and "total_assets" in df.columns:
        df["roa"] = df["net_income_ttm"] / (df["total_assets"] + 1e-9)
    if "gross_profit_ttm" in df.columns and "revenue_ttm" in df.columns:
        df["gross_margin"] = df["gross_profit_ttm"] / (df["revenue_ttm"] + 1e-9)
    if "operating_cf_ttm" in df.columns and "revenue_ttm" in df.columns:
        df["ocf_margin"] = df["operating_cf_ttm"] / (df["revenue_ttm"] + 1e-9)

    # Leverage
    if "total_liabilities" in df.columns and "total_assets" in df.columns:
        df["leverage"] = df["total_liabilities"] / (df["total_assets"] + 1e-9)

    return df


def compute_factor_panel(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end factor construction from OHLCV + fundamentals.

    Returns a DataFrame with added factor columns; apply cross-sectional z-score
    downstream if desired via `zscore_cross_section`.
    """
    df1 = compute_ohlcv_factors(df)
    df2 = compute_fundamental_factors(df1)
    return df2


__all__ = [
    "compute_factor_panel",
    "compute_ohlcv_factors",
    "compute_fundamental_factors",
    "zscore_cross_section",
]
