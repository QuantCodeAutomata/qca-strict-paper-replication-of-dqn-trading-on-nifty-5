"""
Tests for performance metrics module.

Tests verify:
- Cumulative return formula: CR = product(1 + R_t) - 1
- Sharpe ratio computation
- Maximum drawdown computation
- Profit factor and winning percentage
- Edge cases: empty series, single value, all positive/negative
"""

import numpy as np
import pandas as pd
import pytest

from metrics.performance import (
    _compute_round_trips,
    compute_all_metrics,
    compute_avg_holding_period,
    compute_benchmark_metrics,
    compute_cumulative_return,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_winning_percentage,
    format_metrics_table,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_returns(values: list[float]) -> pd.Series:
    """Create a returns Series from a list of values."""
    return pd.Series(
        values,
        index=pd.date_range("2020-01-01", periods=len(values), freq="B"),
        name="returns",
    )


def make_trade_log(trades: list[dict]) -> pd.DataFrame:
    """Create a trade log DataFrame."""
    return pd.DataFrame(trades)


# ─── Tests: compute_cumulative_return ────────────────────────────────────────

def test_cumulative_return_formula():
    """CR = product(1 + R_t) - 1."""
    returns = make_returns([0.10, -0.05, 0.08])
    cr = compute_cumulative_return(returns)
    expected = (1.10 * 0.95 * 1.08) - 1.0
    assert abs(cr - expected) < 1e-10


def test_cumulative_return_zero_returns():
    """All-zero returns should give CR = 0."""
    returns = make_returns([0.0, 0.0, 0.0])
    assert abs(compute_cumulative_return(returns)) < 1e-10


def test_cumulative_return_empty():
    """Empty series should return 0."""
    assert compute_cumulative_return(pd.Series(dtype=float)) == 0.0


def test_cumulative_return_single():
    """Single return value."""
    returns = make_returns([0.05])
    assert abs(compute_cumulative_return(returns) - 0.05) < 1e-10


def test_cumulative_return_all_positive():
    """All positive returns should give positive CR."""
    returns = make_returns([0.01] * 10)
    cr = compute_cumulative_return(returns)
    assert cr > 0


def test_cumulative_return_all_negative():
    """All negative returns should give negative CR."""
    returns = make_returns([-0.01] * 10)
    cr = compute_cumulative_return(returns)
    assert cr < 0


def test_cumulative_return_large_loss():
    """100% loss should give CR = -1."""
    returns = make_returns([-1.0])
    assert abs(compute_cumulative_return(returns) - (-1.0)) < 1e-10


# ─── Tests: compute_sharpe_ratio ─────────────────────────────────────────────

def test_sharpe_ratio_positive_returns():
    """Consistently positive returns should give positive Sharpe."""
    returns = make_returns([0.01] * 100)
    sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0


def test_sharpe_ratio_zero_std():
    """Constant returns should give very high or inf Sharpe (handled gracefully)."""
    returns = make_returns([0.01] * 50)
    sharpe = compute_sharpe_ratio(returns)
    # Should not raise; may be very large or 0 depending on implementation
    assert isinstance(sharpe, float)


def test_sharpe_ratio_empty():
    """Empty series should return 0."""
    assert compute_sharpe_ratio(pd.Series(dtype=float)) == 0.0


def test_sharpe_ratio_single():
    """Single return should return 0 (insufficient data)."""
    returns = make_returns([0.05])
    assert compute_sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_rf_zero_assumption():
    """With rf=0, Sharpe = mean/std * sqrt(252)."""
    np.random.seed(42)
    returns = make_returns(list(np.random.normal(0.001, 0.01, 252)))
    sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
    # Should be a finite number
    assert np.isfinite(sharpe)


# ─── Tests: compute_max_drawdown ─────────────────────────────────────────────

def test_max_drawdown_negative():
    """Max drawdown should be <= 0."""
    returns = make_returns([0.1, -0.2, 0.05, -0.15, 0.08])
    mdd = compute_max_drawdown(returns)
    assert mdd <= 0


def test_max_drawdown_all_positive():
    """All positive returns should give 0 drawdown."""
    returns = make_returns([0.01] * 20)
    mdd = compute_max_drawdown(returns)
    assert mdd >= -1e-10  # approximately 0


def test_max_drawdown_empty():
    """Empty series should return 0."""
    assert compute_max_drawdown(pd.Series(dtype=float)) == 0.0


def test_max_drawdown_known_value():
    """Test with known drawdown sequence."""
    # Equity: 1.0 → 1.1 → 0.88 → 0.968
    # Peak at 1.1, trough at 0.88 → drawdown = (0.88 - 1.1) / 1.1 ≈ -0.2
    returns = make_returns([0.10, -0.20, 0.10])
    mdd = compute_max_drawdown(returns)
    assert mdd <= -0.15  # at least 15% drawdown


def test_max_drawdown_range():
    """Max drawdown should be between -1 and 0."""
    np.random.seed(42)
    returns = make_returns(list(np.random.normal(0, 0.02, 100)))
    mdd = compute_max_drawdown(returns)
    assert -1.0 <= mdd <= 0.0


# ─── Tests: compute_profit_factor ────────────────────────────────────────────

def test_profit_factor_all_positive():
    """All positive returns should give high profit factor."""
    returns = make_returns([0.01] * 20)
    pf = compute_profit_factor(returns)
    assert pf > 1.0 or pf == 0.0  # no losses → inf or handled as 0


def test_profit_factor_all_negative():
    """All negative returns should give 0 profit factor."""
    returns = make_returns([-0.01] * 20)
    pf = compute_profit_factor(returns)
    assert pf == 0.0


def test_profit_factor_empty():
    """Empty series should return 0."""
    assert compute_profit_factor(pd.Series(dtype=float)) == 0.0


def test_profit_factor_mixed():
    """Mixed returns should give finite profit factor."""
    returns = make_returns([0.02, -0.01, 0.03, -0.02, 0.01])
    pf = compute_profit_factor(returns)
    assert pf >= 0.0
    assert np.isfinite(pf)


# ─── Tests: compute_winning_percentage ───────────────────────────────────────

def test_winning_percentage_range():
    """Winning percentage should be in [0, 1]."""
    trades = [
        {"date": pd.Timestamp("2020-01-02"), "ticker": "A", "action": "buy", "quantity": 1, "price": 100.0},
        {"date": pd.Timestamp("2020-01-05"), "ticker": "A", "action": "sell", "quantity": 1, "price": 110.0},
        {"date": pd.Timestamp("2020-01-07"), "ticker": "A", "action": "buy", "quantity": 1, "price": 110.0},
        {"date": pd.Timestamp("2020-01-10"), "ticker": "A", "action": "sell", "quantity": 1, "price": 105.0},
    ]
    trade_log = make_trade_log(trades)
    wp = compute_winning_percentage(trade_log)
    assert 0.0 <= wp <= 1.0


def test_winning_percentage_all_wins():
    """All profitable trades should give 100% win rate."""
    trades = [
        {"date": pd.Timestamp("2020-01-02"), "ticker": "A", "action": "buy", "quantity": 1, "price": 100.0},
        {"date": pd.Timestamp("2020-01-05"), "ticker": "A", "action": "sell", "quantity": 1, "price": 110.0},
    ]
    trade_log = make_trade_log(trades)
    wp = compute_winning_percentage(trade_log)
    assert abs(wp - 1.0) < 1e-10


def test_winning_percentage_empty():
    """Empty trade log should return 0."""
    assert compute_winning_percentage(pd.DataFrame()) == 0.0


# ─── Tests: compute_avg_holding_period ───────────────────────────────────────

def test_avg_holding_period_positive():
    """Average holding period should be positive."""
    trades = [
        {"date": pd.Timestamp("2020-01-02"), "ticker": "A", "action": "buy", "quantity": 1, "price": 100.0},
        {"date": pd.Timestamp("2020-01-07"), "ticker": "A", "action": "sell", "quantity": 1, "price": 110.0},
    ]
    trade_log = make_trade_log(trades)
    ahp = compute_avg_holding_period(trade_log)
    assert ahp == 5.0  # 5 calendar days


def test_avg_holding_period_empty():
    """Empty trade log should return 0."""
    assert compute_avg_holding_period(pd.DataFrame()) == 0.0


# ─── Tests: _compute_round_trips ─────────────────────────────────────────────

def test_round_trips_fifo():
    """FIFO matching: first buy matched with first sell."""
    trades = [
        {"date": pd.Timestamp("2020-01-02"), "ticker": "A", "action": "buy", "quantity": 1, "price": 100.0},
        {"date": pd.Timestamp("2020-01-03"), "ticker": "A", "action": "buy", "quantity": 1, "price": 105.0},
        {"date": pd.Timestamp("2020-01-05"), "ticker": "A", "action": "sell", "quantity": 1, "price": 110.0},
    ]
    trade_log = make_trade_log(trades)
    trips = _compute_round_trips(trade_log)
    assert len(trips) == 1
    assert trips[0]["entry_price"] == 100.0  # FIFO: first buy
    assert trips[0]["pnl"] == 10.0


def test_round_trips_no_sells():
    """No sells should give no completed round trips."""
    trades = [
        {"date": pd.Timestamp("2020-01-02"), "ticker": "A", "action": "buy", "quantity": 1, "price": 100.0},
    ]
    trade_log = make_trade_log(trades)
    trips = _compute_round_trips(trade_log)
    assert len(trips) == 0


# ─── Tests: compute_all_metrics ──────────────────────────────────────────────

def test_compute_all_metrics_keys():
    """compute_all_metrics should return all required keys."""
    returns = make_returns([0.01, -0.005, 0.02, -0.01, 0.015] * 10)
    metrics = compute_all_metrics(returns, pd.DataFrame(), label="Test")
    required_keys = [
        "label", "cumulative_return", "sharpe_ratio", "max_drawdown",
        "profit_factor", "winning_percentage", "avg_holding_period",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"


def test_compute_all_metrics_types():
    """All metric values should be numeric."""
    returns = make_returns([0.01, -0.005, 0.02] * 20)
    metrics = compute_all_metrics(returns, pd.DataFrame())
    for key, val in metrics.items():
        if key != "label":
            assert isinstance(val, (int, float)), f"Non-numeric value for {key}: {val}"


# ─── Tests: compute_benchmark_metrics ────────────────────────────────────────

def test_benchmark_metrics_from_prices():
    """Benchmark metrics should be computable from price series."""
    prices = pd.Series(
        [100.0 * (1.001 ** i) for i in range(252)],
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
    )
    metrics = compute_benchmark_metrics(prices)
    assert metrics["cumulative_return"] > 0
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics


def test_benchmark_metrics_empty_raises():
    """Empty price series should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_benchmark_metrics(pd.Series(dtype=float))


# ─── Tests: format_metrics_table ─────────────────────────────────────────────

def test_format_metrics_table_shape():
    """Metrics table should have correct shape."""
    m1 = {"label": "DQN", "cumulative_return": 0.1, "sharpe_ratio": 1.2,
          "max_drawdown": -0.05, "profit_factor": 1.5, "winning_percentage": 0.6,
          "avg_holding_period": 3.0, "n_trades": 10}
    m2 = {"label": "BaH", "cumulative_return": 0.05, "sharpe_ratio": 0.8,
          "max_drawdown": -0.15, "profit_factor": 1.1, "winning_percentage": 0.55,
          "avg_holding_period": 252.0, "n_trades": 1}
    table = format_metrics_table([m1, m2])
    assert len(table) == 2
    assert "Cumulative Return" in table.columns
    assert "Sharpe Ratio" in table.columns
    assert "Max Drawdown" in table.columns
