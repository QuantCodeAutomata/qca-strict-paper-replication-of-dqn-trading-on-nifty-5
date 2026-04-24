"""
Performance metrics for DQN trading strategy evaluation.

Implements all metrics explicitly mentioned in the paper:
- Cumulative return
- Sharpe ratio
- Maximum drawdown
- Profit factor
- Winning percentage
- Average holding period

Using quantstats where available (Context7 confirmed).
# Using quantstats — Context7 confirmed for Sharpe, max_drawdown, profit_factor, win_rate
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Using quantstats — Context7 confirmed
import quantstats as qs

logger = logging.getLogger(__name__)


def compute_cumulative_return(daily_returns: pd.Series) -> float:
    """
    Compute cumulative return over the test period.

    Formula (paper-explicit): CR = product(1 + R_t) - 1

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns (not percentage, e.g., 0.01 for 1%).

    Returns
    -------
    float
        Cumulative return as a decimal (e.g., 0.15 for 15%).
    """
    if daily_returns.empty:
        return 0.0
    # Remove NaN
    returns = daily_returns.dropna()
    if returns.empty:
        return 0.0
    # Paper formula: CR = product(1 + R_t) - 1
    cumulative = (1.0 + returns).prod() - 1.0
    return float(cumulative)


def compute_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Formula: (R_p - R_f) / σ_p * sqrt(periods_per_year)

    Assumption: R_f = 0 (paper does not specify; documented in assumptions.md).

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    risk_free_rate : float
        Annual risk-free rate (default: 0, per assumption sheet).
    periods_per_year : int
        Trading days per year for annualization (default: 252).

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if daily_returns.empty:
        return 0.0
    returns = daily_returns.dropna()
    if len(returns) < 2:
        return 0.0

    # Using quantstats — Context7 confirmed
    try:
        sharpe = qs.stats.sharpe(returns, rf=risk_free_rate, periods=periods_per_year)
        return float(sharpe) if not np.isnan(sharpe) else 0.0
    except Exception as e:
        logger.warning("quantstats sharpe failed: %s; computing manually", e)
        daily_rf = risk_free_rate / periods_per_year
        excess = returns - daily_rf
        std = excess.std()
        if std < 1e-10:
            return 0.0
        return float(excess.mean() / std * np.sqrt(periods_per_year))


def compute_max_drawdown(daily_returns: pd.Series) -> float:
    """
    Compute maximum drawdown.

    Formula: max peak-to-trough decline divided by peak wealth.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g., -0.20 for 20% drawdown).
    """
    if daily_returns.empty:
        return 0.0
    returns = daily_returns.dropna()
    if returns.empty:
        return 0.0

    # Using quantstats — Context7 confirmed
    try:
        mdd = qs.stats.max_drawdown(returns)
        return float(mdd) if not np.isnan(mdd) else 0.0
    except Exception as e:
        logger.warning("quantstats max_drawdown failed: %s; computing manually", e)
        wealth = (1.0 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth - peak) / peak
        return float(drawdown.min())


def compute_profit_factor(daily_returns: pd.Series) -> float:
    """
    Compute profit factor from daily returns.

    Formula: total profit from winning days / total loss from losing days.

    Note: Paper defines this at trade level; we compute at daily return level
    as a proxy when trade-level data is insufficient.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    float
        Profit factor (> 1 is profitable). Returns 0.0 if no losing days.
    """
    if daily_returns.empty:
        return 0.0
    returns = daily_returns.dropna()

    # Using quantstats — Context7 confirmed
    try:
        pf = qs.stats.profit_factor(returns)
        return float(pf) if not np.isnan(pf) and not np.isinf(pf) else 0.0
    except Exception as e:
        logger.warning("quantstats profit_factor failed: %s; computing manually", e)
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses < 1e-10:
            return float("inf") if wins > 0 else 0.0
        return float(wins / losses)


def compute_profit_factor_from_trades(trade_log: pd.DataFrame) -> float:
    """
    Compute profit factor from completed round-trip trades.

    Formula: sum(profits from winning trades) / sum(losses from losing trades)

    Parameters
    ----------
    trade_log : pd.DataFrame
        Trade log with columns: date, ticker, action, quantity, price.

    Returns
    -------
    float
        Profit factor from completed trades.
    """
    if trade_log.empty:
        return 0.0

    completed_trades = _compute_round_trips(trade_log)
    if not completed_trades:
        return 0.0

    profits = [t["pnl"] for t in completed_trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in completed_trades if t["pnl"] < 0]

    total_profit = sum(profits)
    total_loss = sum(losses)

    if total_loss < 1e-10:
        return float("inf") if total_profit > 0 else 0.0
    return float(total_profit / total_loss)


def compute_winning_percentage(trade_log: pd.DataFrame) -> float:
    """
    Compute winning percentage from completed round-trip trades.

    Formula: number of profitable trades / total completed trades

    Parameters
    ----------
    trade_log : pd.DataFrame
        Trade log with columns: date, ticker, action, quantity, price.

    Returns
    -------
    float
        Winning percentage as decimal (e.g., 0.6 for 60%).
    """
    if trade_log.empty:
        return 0.0

    completed_trades = _compute_round_trips(trade_log)
    if not completed_trades:
        return 0.0

    n_winning = sum(1 for t in completed_trades if t["pnl"] > 0)
    return float(n_winning / len(completed_trades))


def compute_avg_holding_period(trade_log: pd.DataFrame) -> float:
    """
    Compute average holding period in days across completed round-trip trades.

    Formula: average (exit_date - entry_date) across completed trades.

    Parameters
    ----------
    trade_log : pd.DataFrame
        Trade log with columns: date, ticker, action, quantity, price.

    Returns
    -------
    float
        Average holding period in calendar days.
    """
    if trade_log.empty:
        return 0.0

    completed_trades = _compute_round_trips(trade_log)
    if not completed_trades:
        return 0.0

    holding_periods = [t["holding_days"] for t in completed_trades]
    return float(np.mean(holding_periods))


def _compute_round_trips(trade_log: pd.DataFrame) -> list[dict]:
    """
    Match buy and sell trades into completed round trips.

    Uses FIFO matching: each sell is matched against the earliest unmatched buy.

    Parameters
    ----------
    trade_log : pd.DataFrame
        Trade log with columns: date, ticker, action, quantity, price.

    Returns
    -------
    list of dict
        Completed round trips with keys: ticker, entry_date, exit_date,
        entry_price, exit_price, quantity, pnl, holding_days.
    """
    if trade_log.empty:
        return []

    completed = []

    for ticker in trade_log["ticker"].unique() if "ticker" in trade_log.columns else [None]:
        if ticker is not None:
            ticker_trades = trade_log[trade_log["ticker"] == ticker].copy()
        else:
            ticker_trades = trade_log.copy()

        ticker_trades = ticker_trades.sort_values("date")
        buy_queue: list[dict] = []

        for _, row in ticker_trades.iterrows():
            if row["action"] == "buy":
                buy_queue.append({
                    "date": row["date"],
                    "price": row["price"],
                    "quantity": row["quantity"],
                    "ticker": ticker,
                })
            elif row["action"] == "sell" and buy_queue:
                entry = buy_queue.pop(0)
                pnl = (row["price"] - entry["price"]) * min(
                    row["quantity"], entry["quantity"]
                )
                holding_days = (
                    (row["date"] - entry["date"]).days
                    if hasattr(row["date"], "days") or isinstance(row["date"], pd.Timestamp)
                    else 0
                )
                completed.append({
                    "ticker": ticker,
                    "entry_date": entry["date"],
                    "exit_date": row["date"],
                    "entry_price": entry["price"],
                    "exit_price": row["price"],
                    "quantity": min(row["quantity"], entry["quantity"]),
                    "pnl": pnl,
                    "holding_days": holding_days,
                })

    return completed


def compute_all_metrics(
    daily_returns: pd.Series,
    trade_log: pd.DataFrame,
    risk_free_rate: float = 0.0,
    label: str = "Strategy",
) -> dict:
    """
    Compute all paper-specified evaluation metrics.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    trade_log : pd.DataFrame
        Trade log for trade-level metrics.
    risk_free_rate : float
        Annual risk-free rate (default: 0).
    label : str
        Label for the strategy.

    Returns
    -------
    dict
        Dictionary with all metrics.
    """
    metrics = {
        "label": label,
        "cumulative_return": compute_cumulative_return(daily_returns),
        "sharpe_ratio": compute_sharpe_ratio(daily_returns, risk_free_rate),
        "max_drawdown": compute_max_drawdown(daily_returns),
        "profit_factor": (
            compute_profit_factor_from_trades(trade_log)
            if not trade_log.empty
            else compute_profit_factor(daily_returns)
        ),
        "winning_percentage": (
            compute_winning_percentage(trade_log)
            if not trade_log.empty
            else _compute_daily_win_rate(daily_returns)
        ),
        "avg_holding_period": compute_avg_holding_period(trade_log),
        "n_trades": len(_compute_round_trips(trade_log)) if not trade_log.empty else 0,
        "n_days": len(daily_returns.dropna()),
    }
    return metrics


def _compute_daily_win_rate(daily_returns: pd.Series) -> float:
    """Compute fraction of positive return days as fallback win rate."""
    if daily_returns.empty:
        return 0.0
    returns = daily_returns.dropna()
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def compute_benchmark_metrics(
    benchmark_prices: pd.Series,
    risk_free_rate: float = 0.0,
    label: str = "Buy-and-Hold Nifty 50",
) -> dict:
    """
    Compute buy-and-hold benchmark metrics from price series.

    Parameters
    ----------
    benchmark_prices : pd.Series
        Daily benchmark close prices.
    risk_free_rate : float
        Annual risk-free rate.
    label : str
        Label for the benchmark.

    Returns
    -------
    dict
        Dictionary with benchmark metrics.
    """
    if benchmark_prices.empty:
        raise ValueError("benchmark_prices is empty")

    # Compute daily returns from prices
    bm_returns = benchmark_prices.pct_change().dropna()

    metrics = {
        "label": label,
        "cumulative_return": compute_cumulative_return(bm_returns),
        "sharpe_ratio": compute_sharpe_ratio(bm_returns, risk_free_rate),
        "max_drawdown": compute_max_drawdown(bm_returns),
        "profit_factor": compute_profit_factor(bm_returns),
        "winning_percentage": _compute_daily_win_rate(bm_returns),
        "avg_holding_period": float(len(bm_returns)),  # held entire period
        "n_trades": 1,  # single buy at start
        "n_days": len(bm_returns),
    }
    return metrics


def format_metrics_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Format a list of metrics dicts into a comparison DataFrame.

    Parameters
    ----------
    metrics_list : list of dict
        List of metrics dictionaries from compute_all_metrics.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Strategy": m.get("label", "Unknown"),
            "Cumulative Return": f"{m.get('cumulative_return', 0):.2%}",
            "Sharpe Ratio": f"{m.get('sharpe_ratio', 0):.4f}",
            "Max Drawdown": f"{m.get('max_drawdown', 0):.2%}",
            "Profit Factor": f"{m.get('profit_factor', 0):.4f}",
            "Winning %": f"{m.get('winning_percentage', 0):.2%}",
            "Avg Holding (days)": f"{m.get('avg_holding_period', 0):.1f}",
            "N Trades": m.get("n_trades", 0),
        })
    return pd.DataFrame(rows).set_index("Strategy")
