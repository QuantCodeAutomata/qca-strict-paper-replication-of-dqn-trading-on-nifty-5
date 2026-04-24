"""
Tests for the trading environment.

Tests verify:
- Portfolio wealth formula: V_t = C_t + sum_i h_{i,t} * p_{i,t}
- Action constraints: no over-spending, no over-selling
- Reward computation
- Episode reset and termination
"""

import numpy as np
import pandas as pd
import pytest

from environment.trading_env import Action, TradingEnvironment


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_env(
    n_steps: int = 20,
    n_stocks: int = 1,
    initial_cash: float = 10_000.0,
    reward_type: str = "pct_return",
    mode: str = "single_stock",
    price_value: float = 100.0,
) -> TradingEnvironment:
    """Create a simple trading environment for testing."""
    dates = pd.date_range("2020-01-01", periods=n_steps + 10, freq="B")
    prices = pd.DataFrame(
        {f"STOCK{i}": price_value for i in range(n_stocks)},
        index=dates,
    )
    state_dim = 10 * n_stocks
    states = np.random.rand(n_steps, state_dim).astype(np.float32)
    decision_dates = dates[:n_steps]

    return TradingEnvironment(
        prices=prices,
        states=states,
        decision_dates=decision_dates,
        initial_cash=initial_cash,
        trade_size=1,
        reward_type=reward_type,
        allow_short=False,
        transaction_cost=0.0,
        mode=mode,
    )


# ─── Tests: initialization ───────────────────────────────────────────────────

def test_env_initial_state():
    """Environment should start with correct initial state."""
    env = make_env()
    state = env.reset()
    assert state is not None
    assert state.shape == (10,)
    assert env._cash == 10_000.0
    assert env._holdings == {"STOCK0": 0}
    assert not env.is_done


def test_env_empty_prices_raises():
    """Empty prices should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        TradingEnvironment(
            prices=pd.DataFrame(),
            states=np.zeros((5, 10)),
            decision_dates=pd.date_range("2020-01-01", periods=5),
        )


def test_env_mismatched_states_dates_raises():
    """Mismatched states and dates should raise ValueError."""
    prices = pd.DataFrame(
        {"A": [100.0] * 20},
        index=pd.date_range("2020-01-01", periods=20),
    )
    with pytest.raises(ValueError):
        TradingEnvironment(
            prices=prices,
            states=np.zeros((10, 5)),
            decision_dates=pd.date_range("2020-01-01", periods=15),
        )


# ─── Tests: portfolio wealth formula ─────────────────────────────────────────

def test_wealth_formula_no_holdings():
    """V_t = C_t when no holdings."""
    env = make_env(initial_cash=10_000.0, price_value=100.0)
    env.reset()
    # Hold action — wealth should equal cash
    _, _, _, info = env.step(Action.HOLD)
    assert abs(info["wealth"] - 10_000.0) < 1e-6


def test_wealth_formula_after_buy():
    """V_t = C_t + h * p after buying 1 share."""
    env = make_env(initial_cash=10_000.0, price_value=100.0)
    env.reset()
    _, _, _, info = env.step(Action.BUY)
    # Cash reduced by 100, holdings = 1 share at 100
    expected_cash = 10_000.0 - 100.0
    expected_wealth = expected_cash + 1 * 100.0
    assert abs(info["cash"] - expected_cash) < 1e-6
    assert abs(info["wealth"] - expected_wealth) < 1e-6


def test_wealth_formula_after_sell():
    """V_t = C_t after selling all holdings."""
    env = make_env(initial_cash=10_000.0, price_value=100.0)
    env.reset()
    env.step(Action.BUY)   # buy 1 share
    _, _, _, info = env.step(Action.SELL)  # sell 1 share
    # Should be back to initial cash
    assert abs(info["cash"] - 10_000.0) < 1e-6
    assert abs(info["wealth"] - 10_000.0) < 1e-6


# ─── Tests: action constraints ───────────────────────────────────────────────

def test_no_buy_without_cash():
    """Cannot buy if insufficient cash."""
    env = make_env(initial_cash=50.0, price_value=100.0)
    env.reset()
    _, _, _, info = env.step(Action.BUY)
    # Trade should be skipped — cash unchanged
    assert abs(info["cash"] - 50.0) < 1e-6
    assert info["holdings"]["STOCK0"] == 0


def test_no_sell_without_holdings():
    """Cannot sell if no holdings (no shorting)."""
    env = make_env(initial_cash=10_000.0, price_value=100.0)
    env.reset()
    _, _, _, info = env.step(Action.SELL)
    # Trade should be skipped — cash unchanged
    assert abs(info["cash"] - 10_000.0) < 1e-6
    assert info["holdings"]["STOCK0"] == 0


def test_no_negative_holdings():
    """Holdings should never go negative when shorting is disabled."""
    env = make_env(initial_cash=10_000.0, price_value=100.0)
    env.reset()
    for _ in range(5):
        env.step(Action.SELL)
    assert env._holdings["STOCK0"] >= 0


# ─── Tests: reward computation ───────────────────────────────────────────────

def test_reward_pct_return_hold():
    """Hold action with no price change should give ~0 reward."""
    env = make_env(initial_cash=10_000.0, price_value=100.0, reward_type="pct_return")
    env.reset()
    _, reward, _, _ = env.step(Action.HOLD)
    assert abs(reward) < 1e-10


def test_reward_wealth_change_hold():
    """Hold action with no price change should give 0 wealth change reward."""
    env = make_env(initial_cash=10_000.0, price_value=100.0, reward_type="wealth_change")
    env.reset()
    _, reward, _, _ = env.step(Action.HOLD)
    assert abs(reward) < 1e-10


def test_reward_type_invalid():
    """Invalid reward type should raise ValueError."""
    env = make_env()
    env.reward_type = "invalid_type"
    env.reset()
    with pytest.raises(ValueError, match="Unknown reward_type"):
        env.step(Action.HOLD)


# ─── Tests: episode lifecycle ────────────────────────────────────────────────

def test_episode_terminates():
    """Episode should terminate after n_steps."""
    n_steps = 10
    env = make_env(n_steps=n_steps)
    env.reset()
    done = False
    step_count = 0
    while not done:
        _, _, done, _ = env.step(Action.HOLD)
        step_count += 1
    assert step_count == n_steps
    assert env.is_done


def test_step_after_done_raises():
    """Stepping after episode ends should raise RuntimeError."""
    env = make_env(n_steps=3)
    env.reset()
    for _ in range(3):
        env.step(Action.HOLD)
    with pytest.raises(RuntimeError, match="done"):
        env.step(Action.HOLD)


def test_reset_clears_state():
    """Reset should restore initial state."""
    env = make_env(initial_cash=10_000.0)
    env.reset()
    env.step(Action.BUY)
    env.reset()
    assert abs(env._cash - 10_000.0) < 1e-6
    assert env._holdings["STOCK0"] == 0
    assert not env.is_done


# ─── Tests: history and trade log ────────────────────────────────────────────

def test_history_length():
    """History should have one entry per step."""
    n_steps = 15
    env = make_env(n_steps=n_steps)
    env.reset()
    for _ in range(n_steps):
        env.step(Action.HOLD)
    history = env.get_history()
    assert len(history) == n_steps


def test_trade_log_records_buys():
    """Trade log should record buy actions."""
    env = make_env(initial_cash=10_000.0, price_value=100.0)
    env.reset()
    env.step(Action.BUY)
    trade_log = env.get_trade_log()
    assert len(trade_log) == 1
    assert trade_log.iloc[0]["action"] == "buy"


def test_wealth_series_length():
    """Wealth series should have one entry per step."""
    n_steps = 10
    env = make_env(n_steps=n_steps)
    env.reset()
    for _ in range(n_steps):
        env.step(Action.HOLD)
    wealth = env.get_wealth_series()
    assert len(wealth) == n_steps


def test_daily_returns_from_wealth():
    """Daily returns should be computable from wealth series."""
    env = make_env(n_steps=10, initial_cash=10_000.0)
    env.reset()
    for _ in range(10):
        env.step(Action.HOLD)
    returns = env.get_daily_returns()
    assert len(returns) == 9  # pct_change drops first


# ─── Tests: action enum ──────────────────────────────────────────────────────

def test_action_values():
    """Action enum should have correct integer values."""
    assert Action.HOLD == 0
    assert Action.BUY == 1
    assert Action.SELL == 2
