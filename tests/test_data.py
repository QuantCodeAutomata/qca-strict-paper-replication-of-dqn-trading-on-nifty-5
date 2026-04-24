"""
Tests for data download and preprocessing modules.

Tests verify:
- Return computation formula: r_t = (p_t - p_{t-1}) / p_{t-1}
- Min-Max normalization: x' = (x - min(x)) / (max(x) - min(x)) → [0,1]
- State matrix construction
- Edge cases: empty data, single row, zero denominator
"""

import numpy as np
import pandas as pd
import pytest

from data.preprocess import (
    build_state_matrix,
    compute_returns,
    minmax_normalize,
    minmax_normalize_to_neg1_pos1,
    prepare_features,
    split_train_test,
)
from data.nifty50_tickers import NIFTY50_TICKERS, BENCHMARK_TICKER, SINGLE_STOCK_TICKER


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_price_df(n_rows: int = 100, n_stocks: int = 2, seed: int = 42) -> pd.DataFrame:
    """Create synthetic price DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_rows, n_stocks)), axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"STOCK{i}" for i in range(n_stocks)])


def make_returns_df(n_rows: int = 100, n_stocks: int = 2, seed: int = 42) -> pd.DataFrame:
    """Create synthetic returns DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    returns = rng.normal(0, 0.01, (n_rows, n_stocks))
    return pd.DataFrame(returns, index=dates, columns=[f"STOCK{i}" for i in range(n_stocks)])


# ─── Tests: compute_returns ──────────────────────────────────────────────────

def test_compute_returns_formula():
    """Verify r_t = (p_t - p_{t-1}) / p_{t-1} formula."""
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0, 121.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    returns = compute_returns(prices)

    assert len(returns) == 3, "First row should be dropped"
    assert abs(returns["A"].iloc[0] - 0.10) < 1e-10, "110/100 - 1 = 0.10"
    assert abs(returns["A"].iloc[1] - (-0.10)) < 1e-9, "99/110 - 1 ≈ -0.10"
    assert abs(returns["A"].iloc[2] - (121/99 - 1)) < 1e-10


def test_compute_returns_drops_first_row():
    """First row should be NaN and dropped."""
    prices = make_price_df(n_rows=50)
    returns = compute_returns(prices)
    assert len(returns) == 49
    assert not returns.isna().any().any()


def test_compute_returns_empty_raises():
    """Empty DataFrame should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_returns(pd.DataFrame())


def test_compute_returns_single_row():
    """Single row produces empty returns (no lag possible)."""
    prices = pd.DataFrame(
        {"A": [100.0]},
        index=pd.date_range("2020-01-01", periods=1, freq="D"),
    )
    returns = compute_returns(prices)
    assert len(returns) == 0


# ─── Tests: minmax_normalize ─────────────────────────────────────────────────

def test_minmax_normalize_range():
    """Normalized values must be in [0, 1]."""
    returns = make_returns_df(n_rows=100)
    normalized, _ = minmax_normalize(returns)
    assert normalized.min().min() >= 0.0 - 1e-10
    assert normalized.max().max() <= 1.0 + 1e-10


def test_minmax_normalize_formula():
    """Verify x' = (x - min) / (max - min)."""
    returns = pd.DataFrame(
        {"A": [0.0, 0.5, 1.0, -0.5, 0.25]},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    normalized, params = minmax_normalize(returns)

    col_min = -0.5
    col_max = 1.0
    expected = (returns["A"] - col_min) / (col_max - col_min)
    np.testing.assert_allclose(normalized["A"].values, expected.values, atol=1e-10)


def test_minmax_normalize_zero_denominator():
    """Constant series should not raise; returns 0.5."""
    returns = pd.DataFrame(
        {"A": [0.01, 0.01, 0.01, 0.01]},
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    normalized, _ = minmax_normalize(returns)
    assert (normalized["A"] == 0.5).all()


def test_minmax_normalize_neg1_pos1_range():
    """Text-consistent [-1,1] normalization range."""
    returns = make_returns_df(n_rows=100)
    normalized, _ = minmax_normalize_to_neg1_pos1(returns)
    assert normalized.min().min() >= -1.0 - 1e-10
    assert normalized.max().max() <= 1.0 + 1e-10


def test_minmax_normalize_fit_on_subset():
    """Fitting on subset should apply to full data."""
    returns = make_returns_df(n_rows=100)
    fit_data = returns.iloc[:50]
    normalized, params = minmax_normalize(returns, fit_on=fit_data)

    # Values outside fit range may be clipped to [0,1]
    assert normalized.min().min() >= 0.0 - 1e-10
    assert normalized.max().max() <= 1.0 + 1e-10


def test_minmax_normalize_empty_raises():
    """Empty DataFrame should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        minmax_normalize(pd.DataFrame())


# ─── Tests: build_state_matrix ───────────────────────────────────────────────

def test_build_state_matrix_shape():
    """State matrix shape should be (T - window, window * n_stocks)."""
    returns = make_returns_df(n_rows=50, n_stocks=3)
    window = 10
    states, dates = build_state_matrix(returns, window=window)

    assert states.shape == (50 - window, window * 3)
    assert len(dates) == 50 - window


def test_build_state_matrix_values():
    """State at index i should contain rows i to i+window-1."""
    returns = make_returns_df(n_rows=20, n_stocks=1)
    window = 5
    states, dates = build_state_matrix(returns, window=window)

    # State 0 should be rows 0..4 of returns
    expected = returns.values[:5].flatten()
    np.testing.assert_allclose(states[0], expected, atol=1e-6)


def test_build_state_matrix_decision_dates():
    """Decision dates should start at index `window`."""
    returns = make_returns_df(n_rows=30)
    window = 10
    states, dates = build_state_matrix(returns, window=window)

    assert dates[0] == returns.index[window]
    assert dates[-1] == returns.index[-1]


def test_build_state_matrix_insufficient_data():
    """Should raise ValueError if not enough data."""
    returns = make_returns_df(n_rows=5)
    with pytest.raises(ValueError):
        build_state_matrix(returns, window=10)


def test_build_state_matrix_window_one():
    """Window of 1 should work correctly."""
    returns = make_returns_df(n_rows=10, n_stocks=2)
    states, dates = build_state_matrix(returns, window=1)
    assert states.shape == (9, 2)


# ─── Tests: split_train_test ─────────────────────────────────────────────────

def test_split_train_test_sizes():
    """Train and test splits should cover correct date ranges."""
    prices = make_price_df(n_rows=500)
    benchmark = prices.iloc[:, 0].rename("BM")

    # Use dates within the synthetic range
    train_end = str(prices.index[300].date())
    test_start = str(prices.index[301].date())

    train_p, test_p, train_b, test_b = split_train_test(
        prices, benchmark, train_end=train_end, test_start=test_start
    )

    assert len(train_p) == 301
    assert len(test_p) == 199
    assert len(train_b) == 301
    assert len(test_b) == 199


def test_split_train_test_no_overlap():
    """Train and test periods should not overlap."""
    prices = make_price_df(n_rows=500)
    benchmark = prices.iloc[:, 0].rename("BM")
    train_end = str(prices.index[300].date())
    test_start = str(prices.index[301].date())

    train_p, test_p, _, _ = split_train_test(
        prices, benchmark, train_end=train_end, test_start=test_start
    )

    assert train_p.index[-1] < test_p.index[0]


# ─── Tests: ticker constants ─────────────────────────────────────────────────

def test_nifty50_tickers_not_empty():
    """Ticker list should not be empty."""
    assert len(NIFTY50_TICKERS) > 0


def test_single_stock_in_nifty50():
    """Single stock ticker should be in the Nifty 50 list."""
    assert SINGLE_STOCK_TICKER in NIFTY50_TICKERS


def test_benchmark_ticker_format():
    """Benchmark ticker should be ^NSEI."""
    assert BENCHMARK_TICKER == "^NSEI"


def test_nifty50_tickers_ns_suffix():
    """All Nifty 50 tickers should have .NS suffix."""
    for ticker in NIFTY50_TICKERS:
        assert ticker.endswith(".NS"), f"Ticker {ticker} missing .NS suffix"
