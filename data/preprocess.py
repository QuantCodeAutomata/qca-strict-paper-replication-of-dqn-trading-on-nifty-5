"""
Data preprocessing module for DQN trading experiment.

Implements:
- Daily close-to-close return computation: r_t = (p_t - p_{t-1}) / p_{t-1}
- Min-Max normalization: x' = (x - min(x)) / (max(x) - min(x))  [paper formula → [0,1]]
- Sliding window state construction
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily close-to-close returns for each stock.

    Formula (paper-explicit): r_t = (p_t - p_{t-1}) / p_{t-1}

    The first row per stock will be NaN due to lagging and is dropped.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily close prices with DatetimeIndex and ticker columns.

    Returns
    -------
    pd.DataFrame
        Daily returns with the first row dropped (NaN from lag).
    """
    if prices.empty:
        raise ValueError("prices DataFrame is empty")

    returns = prices.pct_change()
    # Drop first row (NaN from lagging)
    returns = returns.iloc[1:]

    # Validate: no infinite values
    inf_count = np.isinf(returns.values).sum()
    if inf_count > 0:
        logger.warning("Found %d infinite return values; replacing with NaN then ffill", inf_count)
        returns = returns.replace([np.inf, -np.inf], np.nan).ffill()

    return returns


def minmax_normalize(
    returns: pd.DataFrame,
    fit_on: Optional[pd.DataFrame] = None,
    epsilon: float = 1e-10,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply Min-Max normalization to return series.

    Paper formula (literal): x' = (x - min(x)) / (max(x) - min(x))
    This maps values to [0, 1].

    Note: The paper text claims [-1, 1] but the formula gives [0, 1].
    We use the formula-based [0, 1] as the strict interpretation.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns to normalize.
    fit_on : pd.DataFrame, optional
        DataFrame to fit scaler extrema on. If None, fits on `returns` itself.
        Use this to implement training-only fitting (A4) vs full-sample fitting (A3).
    epsilon : float
        Small constant to avoid division by zero when max == min.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Normalized returns in [0, 1] and scaler parameters dict.
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty")

    fit_data = fit_on if fit_on is not None else returns

    scaler_params: dict = {}
    normalized = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for col in returns.columns:
        col_min = fit_data[col].min()
        col_max = fit_data[col].max()
        denom = col_max - col_min

        if denom < epsilon:
            logger.warning(
                "Zero or near-zero range for ticker %s (min=%.6f, max=%.6f); "
                "setting normalized values to 0.5",
                col, col_min, col_max,
            )
            normalized[col] = 0.5
        else:
            normalized[col] = (returns[col] - col_min) / denom
        scaler_params[col] = {"min": col_min, "max": col_max}

    # Clip to [0, 1] to handle out-of-sample extremes
    normalized = normalized.clip(0.0, 1.0)

    return normalized, scaler_params


def minmax_normalize_to_neg1_pos1(
    returns: pd.DataFrame,
    fit_on: Optional[pd.DataFrame] = None,
    epsilon: float = 1e-10,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply Min-Max normalization to [-1, 1] (text-consistent variant A2).

    Formula: x' = 2 * (x - min(x)) / (max(x) - min(x)) - 1

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns to normalize.
    fit_on : pd.DataFrame, optional
        DataFrame to fit scaler extrema on.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Normalized returns in [-1, 1] and scaler parameters dict.
    """
    normalized_01, scaler_params = minmax_normalize(returns, fit_on, epsilon)
    normalized = 2.0 * normalized_01 - 1.0
    return normalized, scaler_params


def build_state_matrix(
    normalized_returns: pd.DataFrame,
    window: int = 10,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Construct sliding-window state vectors from normalized returns.

    State at time t = flattened array of last `window` normalized returns
    across all stocks, ordered by date (oldest first) then by ticker column order.

    Shape: (T - window, window * n_stocks)

    Parameters
    ----------
    normalized_returns : pd.DataFrame
        Normalized daily returns with DatetimeIndex.
    window : int
        Number of past days to include in each state vector (n in paper).

    Returns
    -------
    tuple of (np.ndarray, pd.DatetimeIndex)
        State matrix of shape (T - window, window * n_stocks) and
        corresponding decision dates (dates for which a state is available).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(normalized_returns) <= window:
        raise ValueError(
            f"Not enough data: need > {window} rows, got {len(normalized_returns)}"
        )

    values = normalized_returns.values  # shape (T, n_stocks)
    T, n_stocks = values.shape
    n_states = T - window

    states = np.zeros((n_states, window * n_stocks), dtype=np.float32)
    for i in range(n_states):
        # Window: rows i to i+window-1 (inclusive), shape (window, n_stocks)
        window_data = values[i : i + window]
        # Flatten: [day0_stock0, day0_stock1, ..., day(n-1)_stock(n_stocks-1)]
        states[i] = window_data.flatten()

    # Decision dates: the date at position i+window (the day we act)
    decision_dates = normalized_returns.index[window:]

    assert len(states) == len(decision_dates), (
        f"State count {len(states)} != decision date count {len(decision_dates)}"
    )

    return states, decision_dates


def split_train_test(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split price panel and benchmark into train and test periods.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price panel with DatetimeIndex.
    benchmark : pd.Series
        Full benchmark series with DatetimeIndex.
    train_end : str
        Last date of training period.
    test_start : str
        First date of test period.

    Returns
    -------
    tuple of (train_prices, test_prices, train_benchmark, test_benchmark)
    """
    train_prices = prices.loc[:train_end]
    test_prices = prices.loc[test_start:]
    train_benchmark = benchmark.loc[:train_end]
    test_benchmark = benchmark.loc[test_start:]

    if train_prices.empty:
        raise ValueError(f"No training data found up to {train_end}")
    if test_prices.empty:
        raise ValueError(f"No test data found from {test_start}")

    logger.info(
        "Train: %d rows (%s to %s), Test: %d rows (%s to %s)",
        len(train_prices), train_prices.index[0].date(), train_prices.index[-1].date(),
        len(test_prices), test_prices.index[0].date(), test_prices.index[-1].date(),
    )

    return train_prices, test_prices, train_benchmark, test_benchmark


def prepare_features(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    window: int = 10,
    normalization_range: str = "0_1",
    normalization_fit_scope: str = "full_sample",
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
) -> dict:
    """
    Full feature preparation pipeline.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price panel.
    benchmark : pd.Series
        Full benchmark series.
    window : int
        State window size.
    normalization_range : str
        '0_1' for [0,1] (paper formula) or 'neg1_1' for [-1,1] (paper text).
    normalization_fit_scope : str
        'full_sample' to fit on all data, 'train_only' to fit on train only.
    train_end : str
        Last training date.
    test_start : str
        First test date.

    Returns
    -------
    dict with keys:
        train_prices, test_prices, train_benchmark, test_benchmark,
        train_returns, test_returns,
        train_normalized, test_normalized,
        scaler_params,
        train_states, train_dates,
        test_states, test_dates
    """
    train_prices, test_prices, train_benchmark, test_benchmark = split_train_test(
        prices, benchmark, train_end, test_start
    )

    # Compute returns on full panel (need overlap for first-day return)
    all_returns = compute_returns(prices)
    train_returns = all_returns.loc[:train_end]
    test_returns = all_returns.loc[test_start:]

    # Normalization
    if normalization_fit_scope == "full_sample":
        fit_data = all_returns
    elif normalization_fit_scope == "train_only":
        fit_data = train_returns
    else:
        raise ValueError(f"Unknown normalization_fit_scope: {normalization_fit_scope}")

    normalize_fn = (
        minmax_normalize if normalization_range == "0_1"
        else minmax_normalize_to_neg1_pos1
    )

    all_normalized, scaler_params = normalize_fn(all_returns, fit_on=fit_data)
    train_normalized = all_normalized.loc[:train_end]
    test_normalized = all_normalized.loc[test_start:]

    # Build state matrices
    train_states, train_dates = build_state_matrix(train_normalized, window)
    test_states, test_dates = build_state_matrix(test_normalized, window)

    return {
        "train_prices": train_prices,
        "test_prices": test_prices,
        "train_benchmark": train_benchmark,
        "test_benchmark": test_benchmark,
        "train_returns": train_returns,
        "test_returns": test_returns,
        "train_normalized": train_normalized,
        "test_normalized": test_normalized,
        "scaler_params": scaler_params,
        "train_states": train_states,
        "train_dates": train_dates,
        "test_states": test_states,
        "test_dates": test_dates,
    }
