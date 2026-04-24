"""
Trading environment for DQN-based trading strategy.

Implements a daily decision-frequency portfolio environment with:
- Actions: {0: hold, 1: buy, 2: sell} (paper-explicit)
- Portfolio wealth: V_t = C_t + sum_i h_{i,t} * p_{i,t} (paper-explicit)
- Reward: r_t = (V_t - V_{t-1}) / V_{t-1} (B1: daily percentage return)
- Constraints: no shorting, no over-spending cash, no over-selling holdings

Assumptions (frozen):
- Trade size: 1 share per action
- No transaction costs
- Execution at same-day close price
- Initial cash: 100,000 INR
"""

import logging
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Discrete action set as specified by the paper."""
    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnvironment:
    """
    Daily trading environment for DQN agent.

    Supports both single-stock and multi-stock portfolio modes.
    In single-stock mode, actions apply to the single stock.
    In multi-stock mode, actions apply to each stock independently
    (one action per stock per step, or one global action applied uniformly).

    Parameters
    ----------
    prices : pd.DataFrame
        Daily close prices with DatetimeIndex and ticker columns.
    states : np.ndarray
        Pre-computed state vectors, shape (T, state_dim).
    decision_dates : pd.DatetimeIndex
        Dates corresponding to each state/step.
    initial_cash : float
        Starting cash balance.
    trade_size : int
        Number of shares per buy/sell action.
    reward_type : str
        'pct_return' for percentage portfolio return (B1),
        'wealth_change' for absolute wealth change (B2).
    allow_short : bool
        Whether short selling is permitted.
    transaction_cost : float
        Fractional transaction cost per trade (0 = no cost).
    mode : str
        'single_stock' or 'multi_stock'.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        states: np.ndarray,
        decision_dates: pd.DatetimeIndex,
        initial_cash: float = 100_000.0,
        trade_size: int = 1,
        reward_type: str = "pct_return",
        allow_short: bool = False,
        transaction_cost: float = 0.0,
        mode: str = "single_stock",
    ) -> None:
        if len(states) != len(decision_dates):
            raise ValueError(
                f"states length {len(states)} != decision_dates length {len(decision_dates)}"
            )
        if prices.empty:
            raise ValueError("prices DataFrame is empty")

        self.prices = prices
        self.states = states
        self.decision_dates = decision_dates
        self.initial_cash = initial_cash
        self.trade_size = trade_size
        self.reward_type = reward_type
        self.allow_short = allow_short
        self.transaction_cost = transaction_cost
        self.mode = mode
        self.tickers = list(prices.columns)
        self.n_stocks = len(self.tickers)
        self.state_dim = states.shape[1]
        self.n_steps = len(states)

        # State variables (reset on each episode)
        self._step: int = 0
        self._cash: float = initial_cash
        self._holdings: dict[str, int] = {t: 0 for t in self.tickers}
        self._prev_wealth: float = initial_cash
        self._done: bool = False

        # History for logging
        self._history: list[dict] = []
        self._trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns
        -------
        np.ndarray
            Initial state vector.
        """
        self._step = 0
        self._cash = self.initial_cash
        self._holdings = {t: 0 for t in self.tickers}
        self._prev_wealth = self.initial_cash
        self._done = False
        self._history = []
        self._trade_log = []
        return self.states[0].copy()

    def step(
        self, action: int
    ) -> tuple[Optional[np.ndarray], float, bool, dict]:
        """
        Execute one trading step.

        In single-stock mode, `action` is applied to the single stock.
        In multi-stock mode, `action` is applied uniformly to all stocks.

        Parameters
        ----------
        action : int
            Action from {0: hold, 1: buy, 2: sell}.

        Returns
        -------
        tuple of (next_state, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Environment is done; call reset() first")

        date = self.decision_dates[self._step]
        current_prices = self._get_prices_at(date)

        # Execute trades
        executed_trades = self._execute_action(action, current_prices, date)

        # Compute new portfolio wealth
        new_wealth = self._compute_wealth(current_prices)

        # Compute reward
        reward = self._compute_reward(new_wealth, self._prev_wealth)

        # Log step
        step_info = {
            "date": date,
            "step": self._step,
            "action": action,
            "cash": self._cash,
            "holdings": dict(self._holdings),
            "wealth": new_wealth,
            "prev_wealth": self._prev_wealth,
            "reward": reward,
            "prices": dict(current_prices),
            "executed_trades": executed_trades,
        }
        self._history.append(step_info)

        self._prev_wealth = new_wealth
        self._step += 1
        self._done = self._step >= self.n_steps

        next_state = None if self._done else self.states[self._step].copy()

        return next_state, reward, self._done, step_info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_prices_at(self, date: pd.Timestamp) -> dict[str, float]:
        """Get close prices for all tickers at a given date."""
        if date not in self.prices.index:
            # Use nearest available date
            idx = self.prices.index.get_indexer([date], method="nearest")[0]
            date = self.prices.index[idx]
        row = self.prices.loc[date]
        return {t: float(row[t]) for t in self.tickers}

    def _execute_action(
        self,
        action: int,
        prices: dict[str, float],
        date: pd.Timestamp,
    ) -> list[dict]:
        """
        Execute buy/sell/hold action with portfolio constraints.

        In single-stock mode: action applies to self.tickers[0].
        In multi-stock mode: action applies uniformly to all stocks.

        Returns list of executed trade records.
        """
        executed = []

        if self.mode == "single_stock":
            ticker = self.tickers[0]
            trade = self._execute_single_trade(action, ticker, prices[ticker], date)
            if trade:
                executed.append(trade)
        else:
            # Multi-stock: apply same action to each stock
            for ticker in self.tickers:
                trade = self._execute_single_trade(action, ticker, prices[ticker], date)
                if trade:
                    executed.append(trade)

        return executed

    def _execute_single_trade(
        self,
        action: int,
        ticker: str,
        price: float,
        date: pd.Timestamp,
    ) -> Optional[dict]:
        """Execute a single stock trade with constraint checking."""
        if action == Action.HOLD:
            return None

        cost = price * self.trade_size * (1.0 + self.transaction_cost)

        if action == Action.BUY:
            if self._cash < cost:
                # Insufficient cash — skip trade
                return None
            self._cash -= cost
            self._holdings[ticker] += self.trade_size
            trade_type = "buy"

        elif action == Action.SELL:
            if self._holdings[ticker] < self.trade_size:
                # Insufficient holdings — skip trade (no shorting)
                if not self.allow_short:
                    return None
            proceeds = price * self.trade_size * (1.0 - self.transaction_cost)
            self._cash += proceeds
            self._holdings[ticker] -= self.trade_size
            trade_type = "sell"
        else:
            return None

        trade_record = {
            "date": date,
            "ticker": ticker,
            "action": trade_type,
            "quantity": self.trade_size,
            "price": price,
            "cash_after": self._cash,
            "holdings_after": self._holdings[ticker],
        }
        self._trade_log.append(trade_record)
        return trade_record

    def _compute_wealth(self, prices: dict[str, float]) -> float:
        """
        Compute total portfolio wealth.

        V_t = C_t + sum_i h_{i,t} * p_{i,t}  (paper-explicit formula)
        """
        stock_value = sum(
            self._holdings[t] * prices[t] for t in self.tickers
        )
        return self._cash + stock_value

    def _compute_reward(self, new_wealth: float, prev_wealth: float) -> float:
        """
        Compute step reward.

        B1 (primary): r_t = (V_t - V_{t-1}) / V_{t-1}
        B2 (alternative): r_t = V_t - V_{t-1}
        """
        if self.reward_type == "pct_return":
            if abs(prev_wealth) < 1e-10:
                return 0.0
            return (new_wealth - prev_wealth) / prev_wealth
        elif self.reward_type == "wealth_change":
            return new_wealth - prev_wealth
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_history(self) -> pd.DataFrame:
        """Return step history as a DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history).set_index("date")

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade log as a DataFrame."""
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    def get_wealth_series(self) -> pd.Series:
        """Return portfolio wealth time series."""
        if not self._history:
            return pd.Series(dtype=float)
        history = pd.DataFrame(self._history)
        return pd.Series(
            history["wealth"].values,
            index=pd.DatetimeIndex(history["date"]),
            name="portfolio_wealth",
        )

    def get_daily_returns(self) -> pd.Series:
        """Return daily portfolio return series."""
        wealth = self.get_wealth_series()
        if wealth.empty:
            return pd.Series(dtype=float)
        returns = wealth.pct_change().dropna()
        returns.name = "portfolio_return"
        return returns

    @property
    def current_state(self) -> Optional[np.ndarray]:
        """Current state vector, or None if done."""
        if self._done or self._step >= self.n_steps:
            return None
        return self.states[self._step].copy()

    @property
    def is_done(self) -> bool:
        """Whether the episode has ended."""
        return self._done
