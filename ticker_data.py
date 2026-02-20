from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf
import os
import numpy as np

os.environ["HTTP_PROXY"]  = "http://proxy.isoad.isogmbh.de:81"
os.environ["HTTPS_PROXY"] = "http://proxy.isoad.isogmbh.de:81"

_OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _ensure_dt(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts

def _cache_path(cache_dir: Path, ticker: str, interval: str, auto_adjust: bool) -> Path:
    aa = "aa1" if auto_adjust else "aa0"
    return cache_dir / f"{ticker.upper()}__{interval}__{aa}.parquet"

def _normalize_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure flat OHLCV columns and unique names."""
    if df is None or df.empty:
        return df

    # If MultiIndex columns ever appear, drop to the price-type level (level 0)
    # yfinance returns (price_type, ticker) e.g. ('Close', 'AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c) for c in df.columns]

    # Keep expected columns if present, in stable order
    keep = [c for c in _OHLCV if c in df.columns]
    if keep:
        df = df[keep]

    # De-duplicate if needed
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).last().T

    return df

def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return _normalize_flat(df).sort_index()

def _save_cache(path: Path, df: pd.DataFrame) -> None:
    df.sort_index().to_parquet(path)

def _covers(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if df is None or df.empty:
        return False
    # Treat end as exclusive, like yfinance
    need_end = end - pd.Timedelta(days=1)
    return (df.index.min() <= start) and (df.index.max() >= need_end)

def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index < end)]

def get_ticker(
    ticker: str,
    start,
    end,
    cache_dir: str = "data_cache/yfinance_single",
    interval: str = "1d",
    auto_adjust: bool = False,
    **download_kwargs,
) -> pd.DataFrame:
    """
    Returns a FLAT OHLCV DataFrame for ONE ticker.
    If requested [start, end) is fully inside cache -> slice from cache.
    Else -> download the whole requested period, merge into cache, then slice.
    """
    ticker = ticker.upper().strip()
    start = _ensure_dt(start)
    end = _ensure_dt(end)

    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cdir, ticker, interval, auto_adjust)

    cached = _load_cache(path)

    if cached is not None and _covers(cached, start, end):
        return _slice(cached, start, end)

    # Download the whole requested period (not just missing part)
    dl = yf.download(
        ticker,
        start=str(start.date()),
        end=str(end.date()),
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="column",
        **download_kwargs,
    )
    dl.index = dl.index.tz_localize(None)
    dl = _normalize_flat(dl).sort_index()

    if cached is None or cached.empty:
        merged = dl
    else:
        merged = pd.concat([cached, dl])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    _save_cache(path, merged)
    return _slice(merged, start, end)

def get_df_for_period(tickers, period,  tickers_with_vol=None, tickers_with_ohlc=None ):
    if tickers_with_ohlc is None:
        tickers_with_ohlc = tickers
    if tickers_with_vol is None:
        tickers_with_vol = tickers
        
    df_raw_multi = yf.download(tickers,start=period['start'], end=period['end'])

    df_close_log = np.log(df_raw_multi['Close'] / df_raw_multi['Close'].shift(1))
    df_open_log  = np.log(df_raw_multi['Open']  / df_raw_multi['Open'].shift(1))
    df_high_log  = np.log(df_raw_multi['High']  / df_raw_multi['High'].shift(1))
    df_low_log   = np.log(df_raw_multi['Low']   / df_raw_multi['Low'].shift(1))
    df_vol_log   = np.log(df_raw_multi['Volume']).diff()

    # PatchTST does not support hist_exog_list — each feature is a separate unique_id
    df_list = []
    for ticker in tickers:
        ticker_name_clean = ticker.replace("^", "")
        df_list.append(pd.DataFrame({
            'ds':        df_close_log.index,
            'unique_id': f'{ticker_name_clean}_price',
            'y':         df_close_log[ticker],            
        }).dropna())
        if ticker in tickers_with_ohlc:
            for suffix, series in [('open', df_open_log), ('high', df_high_log), ('low', df_low_log)]:
                df_list.append(pd.DataFrame({
                    'ds':        series.index,
                    'unique_id': f'{ticker_name_clean}_{suffix}',
                    'y':         series[ticker],
                }).dropna())
        if ticker in tickers_with_vol:
            df_list.append(pd.DataFrame({
                'ds':        df_vol_log.index,
                'unique_id': f'{ticker_name_clean}_vol',
                'y':         df_vol_log[ticker],
            }).dropna())

    df = pd.concat(df_list).reset_index(drop=True)

    # --- Sanity checks ---
    # dropna() above removes NaN but NOT inf — replace inf with NaN then drop
    inf_count = np.isinf(df['y']).sum()
    if inf_count > 0:
        print(f"WARNING: {inf_count} inf values found — replacing with NaN and dropping")
        print(df[np.isinf(df['y'])].groupby('unique_id').size().to_string())
        df['y'] = df['y'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['y'])

    nan_count = df['y'].isna().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values remain after cleanup")

    print(df['unique_id'].value_counts())
    all_dates = np.sort(df['ds'].unique())
    n_total = len(all_dates)
    print(f"Total days: {n_total}  |  Total rows: {len(df)}  |  Series: {df['unique_id'].nunique()}")
    print(f"y range: [{df['y'].min():.6f}, {df['y'].max():.6f}]  |  mean: {df['y'].mean():.6f}  |  std: {df['y'].std():.6f}")
    return df
    