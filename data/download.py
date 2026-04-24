"""
Data download module for Nifty 50 stocks and benchmark.

Note: massive lib does not support NSE/Indian stocks (Polygon.io covers US markets only).
yfinance is used as the data source for Yahoo Finance NSE data, which is what the paper
explicitly specifies. This is a necessary deviation from the "use massive lib" guideline
because massive does not provide the required data.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from data.nifty50_tickers import BENCHMARK_TICKER, NIFTY50_TICKERS

logger = logging.getLogger(__name__)

TRAIN_START = "2010-01-01"
TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2020-12-31"
FULL_START = "2010-01-01"
FULL_END = "2020-12-31"


def download_nifty50_data(
    tickers: list[str] = NIFTY50_TICKERS,
    start: str = FULL_START,
    end: str = FULL_END,
    cache_dir: Optional[str] = None,
    price_field: str = "Close",
) -> pd.DataFrame:
    """
    Download daily OHLC data for Nifty 50 stocks from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols (e.g., 'RELIANCE.NS').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    cache_dir : str, optional
        Directory to cache downloaded data as CSV. If file exists, loads from cache.
    price_field : str
        Price field to extract ('Close', 'Open', 'High', 'Low', or 'All').

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and ticker columns containing Close prices.
        If price_field='All', returns MultiIndex columns (field, ticker).
    """
    if not tickers:
        raise ValueError("tickers list cannot be empty")

    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"nifty50_{start}_{end}.csv"
        if cache_path.exists():
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_csv(cache_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df

    logger.info("Downloading data for %d tickers from %s to %s", len(tickers), start, end)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError(f"No data downloaded for tickers {tickers}")

    # yfinance ≥0.2 always returns MultiIndex columns: (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        if price_field == "All":
            result = raw
        else:
            # Extract the requested price field for all tickers
            result = raw[price_field]
            if isinstance(result, pd.Series):
                # Single ticker returns a Series; convert to DataFrame
                result = result.to_frame(name=tickers[0])
            else:
                result = result[tickers]  # ensure column order
    else:
        # Flat columns (older yfinance or single-ticker edge case)
        if price_field == "All":
            result = raw
        else:
            result = raw[[price_field]].rename(columns={price_field: tickers[0]})

    if cache_path:
        result.to_csv(cache_path)
        logger.info("Cached data to %s", cache_path)

    return result


def download_benchmark(
    ticker: str = BENCHMARK_TICKER,
    start: str = FULL_START,
    end: str = FULL_END,
    cache_dir: Optional[str] = None,
) -> pd.Series:
    """
    Download daily Nifty 50 index close prices for benchmarking.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker for the benchmark index (default: '^NSEI').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    cache_dir : str, optional
        Directory to cache downloaded data.

    Returns
    -------
    pd.Series
        Daily close prices with DatetimeIndex.
    """
    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"benchmark_{ticker.replace('^', '')}_{start}_{end}.csv"
        if cache_path.exists():
            logger.info("Loading cached benchmark from %s", cache_path)
            df = pd.read_csv(cache_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            s = df.squeeze()
            s.name = ticker
            return s

    logger.info("Downloading benchmark %s from %s to %s", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if raw.empty:
        raise RuntimeError(f"No benchmark data downloaded for {ticker}")

    # yfinance ≥0.2 returns MultiIndex columns: (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].squeeze()
    else:
        close = raw["Close"].squeeze()
    close = pd.Series(close.values, index=close.index, name=ticker)

    if cache_path:
        close.to_frame().to_csv(cache_path)

    return close


def align_dates(
    price_df: pd.DataFrame,
    benchmark: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align stock prices and benchmark to common trading dates.

    Parameters
    ----------
    price_df : pd.DataFrame
        Stock close prices with DatetimeIndex.
    benchmark : pd.Series
        Benchmark close prices with DatetimeIndex.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        Aligned price DataFrame and benchmark Series on common dates.
    """
    common_idx = price_df.index.intersection(benchmark.index)
    if len(common_idx) == 0:
        raise ValueError("No common dates between stock data and benchmark")
    logger.info(
        "Aligned to %d common trading dates (stocks: %d, benchmark: %d)",
        len(common_idx),
        len(price_df),
        len(benchmark),
    )
    return price_df.loc[common_idx], benchmark.loc[common_idx]


def handle_missing_data(
    price_df: pd.DataFrame,
    method: str = "ffill",
    max_consecutive_missing: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Handle missing close prices according to the frozen assumption sheet.

    Policy: forward-fill up to max_consecutive_missing days, then drop remaining NaN rows.

    Parameters
    ----------
    price_df : pd.DataFrame
        Stock close prices, possibly with NaN values.
    method : str
        Fill method: 'ffill' (forward fill) or 'drop'.
    max_consecutive_missing : int
        Maximum consecutive missing values to forward-fill before dropping.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Cleaned DataFrame and exclusion log dictionary.
    """
    exclusion_log: dict = {
        "original_rows": len(price_df),
        "original_cols": list(price_df.columns),
        "missing_before": price_df.isna().sum().to_dict(),
        "method": method,
    }

    if method == "ffill":
        price_df = price_df.ffill(limit=max_consecutive_missing)

    # Drop rows where any ticker still has NaN after filling
    rows_before = len(price_df)
    price_df = price_df.dropna()
    rows_dropped = rows_before - len(price_df)

    exclusion_log["rows_dropped"] = rows_dropped
    exclusion_log["missing_after"] = price_df.isna().sum().to_dict()
    exclusion_log["final_rows"] = len(price_df)

    if rows_dropped > 0:
        logger.warning("Dropped %d rows with missing data", rows_dropped)

    return price_df, exclusion_log


def load_or_download_all(
    tickers: list[str] = NIFTY50_TICKERS,
    benchmark_ticker: str = BENCHMARK_TICKER,
    start: str = FULL_START,
    end: str = FULL_END,
    cache_dir: str = "data/cache",
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full pipeline: download, align, and clean all data.

    Parameters
    ----------
    tickers : list of str
        Stock tickers to download.
    benchmark_ticker : str
        Benchmark index ticker.
    start : str
        Start date.
    end : str
        End date.
    cache_dir : str
        Cache directory for downloaded data.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series, dict)
        Cleaned price panel, benchmark series, and exclusion log.
    """
    prices = download_nifty50_data(tickers, start, end, cache_dir)
    benchmark = download_benchmark(benchmark_ticker, start, end, cache_dir)
    prices, benchmark = align_dates(prices, benchmark)
    prices, exclusion_log = handle_missing_data(prices)
    # Re-align benchmark after dropping rows
    benchmark = benchmark.loc[prices.index]
    return prices, benchmark, exclusion_log
