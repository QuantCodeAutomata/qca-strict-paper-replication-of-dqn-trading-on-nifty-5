"""
Implementation tests for experiment methodology.

These tests verify that the methodology from the paper is implemented correctly.
They test the structural correctness of the implementation without running
full experiments (which require data downloads and long training).

Tests verify:
- Paper-specified hyperparameters are used
- Normalization formula matches paper
- Reward formula matches paper
- Portfolio wealth formula matches paper
- Action set matches paper
- State construction matches paper description
"""

import numpy as np
import pandas as pd
import pytest
import torch

from agent.dqn_agent import DQNAgent, QNetwork
from data.preprocess import (
    build_state_matrix,
    compute_returns,
    minmax_normalize,
    prepare_features,
)
from environment.trading_env import Action, TradingEnvironment
from experiments.exp1_strict_replication import (
    GAMMA,
    LEARNING_RATE,
    N_EPISODES,
    NORMALIZATION_FIT_SCOPE,
    NORMALIZATION_RANGE,
    REWARD_TYPE,
    RISK_FREE_RATE,
    STATE_WINDOW_N,
    TRAIN_END,
    TEST_START,
)
from metrics.performance import compute_cumulative_return


# ─── Tests: Paper hyperparameters ────────────────────────────────────────────

def test_paper_learning_rate():
    """Learning rate must be 0.001 as specified by paper."""
    assert LEARNING_RATE == 0.001, "Paper specifies learning rate = 0.001"


def test_paper_discount_factor():
    """Discount factor must be 0.99 as specified by paper."""
    assert GAMMA == 0.99, "Paper specifies discount factor = 0.99"


def test_paper_n_episodes():
    """Paper specifies 10,000 training episodes."""
    assert N_EPISODES == 10_000, "Paper specifies 10,000 episodes"


def test_paper_train_test_split():
    """Train/test split must match paper specification."""
    assert TRAIN_END == "2019-12-31", "Paper: train ends 2019-12-31"
    assert TEST_START == "2020-01-01", "Paper: test starts 2020-01-01"


def test_paper_normalization_formula():
    """Normalization must use [0,1] formula (paper formula, not text)."""
    assert NORMALIZATION_RANGE == "0_1", "Paper formula gives [0,1]"


def test_paper_reward_type():
    """Reward must be percentage portfolio return (B1 interpretation)."""
    assert REWARD_TYPE == "pct_return", "B1: daily percentage portfolio return"


def test_paper_risk_free_rate():
    """Risk-free rate must be 0 (documented assumption)."""
    assert RISK_FREE_RATE == 0.0, "Assumption: R_f = 0"


# ─── Tests: Normalization formula adherence ───────────────────────────────────

def test_normalization_maps_to_0_1():
    """Paper formula x' = (x - min) / (max - min) must map to [0,1]."""
    returns = pd.DataFrame(
        {"A": [-0.05, 0.0, 0.03, -0.02, 0.07]},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    normalized, _ = minmax_normalize(returns)
    assert normalized["A"].min() >= 0.0 - 1e-10
    assert normalized["A"].max() <= 1.0 + 1e-10
    # Min should map to 0, max to 1
    assert abs(normalized["A"].min()) < 1e-10
    assert abs(normalized["A"].max() - 1.0) < 1e-10


def test_normalization_formula_exact():
    """Verify exact formula: x' = (x - min) / (max - min)."""
    values = [-0.1, 0.0, 0.1, 0.2, -0.05]
    returns = pd.DataFrame(
        {"A": values},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    normalized, params = minmax_normalize(returns)
    col_min = min(values)
    col_max = max(values)
    for i, v in enumerate(values):
        expected = (v - col_min) / (col_max - col_min)
        assert abs(normalized["A"].iloc[i] - expected) < 1e-10


# ─── Tests: Return formula adherence ─────────────────────────────────────────

def test_return_formula_exact():
    """Verify r_t = (p_t - p_{t-1}) / p_{t-1}."""
    prices = pd.DataFrame(
        {"A": [100.0, 105.0, 98.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    returns = compute_returns(prices)
    assert abs(returns["A"].iloc[0] - (105.0 - 100.0) / 100.0) < 1e-10
    assert abs(returns["A"].iloc[1] - (98.0 - 105.0) / 105.0) < 1e-10


# ─── Tests: Portfolio wealth formula adherence ────────────────────────────────

def test_portfolio_wealth_formula():
    """V_t = C_t + sum_i h_{i,t} * p_{i,t} (paper-explicit)."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"STOCK0": 100.0, "STOCK1": 200.0}, index=dates)
    states = np.zeros((5, 20), dtype=np.float32)
    env = TradingEnvironment(
        prices=prices,
        states=states,
        decision_dates=dates[:5],
        initial_cash=10_000.0,
        mode="multi_stock",
    )
    env.reset()
    env._holdings = {"STOCK0": 3, "STOCK1": 2}
    env._cash = 5_000.0
    wealth = env._compute_wealth({"STOCK0": 100.0, "STOCK1": 200.0})
    expected = 5_000.0 + 3 * 100.0 + 2 * 200.0
    assert abs(wealth - expected) < 1e-6


# ─── Tests: Action set adherence ─────────────────────────────────────────────

def test_action_set_has_three_actions():
    """Paper specifies buy, sell, hold — exactly 3 actions."""
    assert len(Action) == 3


def test_action_set_contains_buy_sell_hold():
    """Action set must contain buy, sell, and hold."""
    action_names = {a.name for a in Action}
    assert "BUY" in action_names
    assert "SELL" in action_names
    assert "HOLD" in action_names


# ─── Tests: State construction adherence ─────────────────────────────────────

def test_state_is_sliding_window():
    """State must be a sliding window of last n normalized returns."""
    returns = pd.DataFrame(
        {"A": np.arange(20, dtype=float)},
        index=pd.date_range("2020-01-01", periods=20, freq="D"),
    )
    window = 5
    states, dates = build_state_matrix(returns, window=window)

    # State at index 0 should contain returns[0:5]
    expected_state_0 = returns.values[:5].flatten()
    np.testing.assert_allclose(states[0], expected_state_0, atol=1e-6)

    # State at index 1 should contain returns[1:6]
    expected_state_1 = returns.values[1:6].flatten()
    np.testing.assert_allclose(states[1], expected_state_1, atol=1e-6)


def test_state_window_n_is_10():
    """State window n must be 10 (frozen assumption)."""
    assert STATE_WINDOW_N == 10


# ─── Tests: Cumulative return formula adherence ───────────────────────────────

def test_cumulative_return_paper_formula():
    """CR = product(1 + R_t) - 1 (paper-explicit)."""
    returns = pd.Series([0.05, -0.03, 0.07, -0.02])
    cr = compute_cumulative_return(returns)
    expected = (1.05 * 0.97 * 1.07 * 0.98) - 1.0
    assert abs(cr - expected) < 1e-10


# ─── Tests: DQN architecture adherence ───────────────────────────────────────

def test_dqn_has_three_output_actions():
    """DQN must output Q-values for 3 actions (buy, sell, hold)."""
    net = QNetwork(state_dim=10, n_actions=3)
    x = torch.randn(1, 10)
    out = net(x)
    assert out.shape[1] == 3


def test_dqn_uses_relu_activation():
    """DQN architecture must use ReLU activations."""
    net = QNetwork(state_dim=10)
    has_relu = any(isinstance(m, torch.nn.ReLU) for m in net.modules())
    assert has_relu, "DQN must use ReLU activations"


def test_dqn_optimizer_learning_rate():
    """DQN optimizer must use paper-specified learning rate 0.001."""
    agent = DQNAgent(state_dim=10, learning_rate=0.001)
    for pg in agent.optimizer.param_groups:
        assert abs(pg["lr"] - 0.001) < 1e-10


def test_dqn_discount_factor():
    """DQN must use paper-specified discount factor 0.99."""
    agent = DQNAgent(state_dim=10, gamma=0.99)
    assert agent.gamma == 0.99


# ─── Tests: Bellman update correctness ───────────────────────────────────────

def test_bellman_target_terminal():
    """For terminal transitions, Bellman target = reward (no future value)."""
    agent = DQNAgent(state_dim=4, n_actions=3, batch_size=4, buffer_capacity=20)
    state = np.zeros(4, dtype=np.float32)
    # Push terminal transitions
    for _ in range(5):
        agent.replay_buffer.push(state, 0, 1.0, state, True)  # done=True
    loss = agent.update()
    # Should not raise and should return a valid loss
    assert loss is not None
    assert np.isfinite(loss)


# ─── Tests: Epsilon-greedy exploration ───────────────────────────────────────

def test_epsilon_greedy_explores_at_high_epsilon():
    """With epsilon=1.0, all actions should be random (not always same)."""
    agent = DQNAgent(state_dim=10, epsilon_start=1.0, seed=42)
    state = np.zeros(10, dtype=np.float32)
    actions = [agent.select_action(state, training=True) for _ in range(50)]
    # With epsilon=1.0, should see multiple different actions
    assert len(set(actions)) > 1


def test_epsilon_greedy_exploits_at_zero_epsilon():
    """With epsilon=0.0, action should be deterministic (greedy)."""
    agent = DQNAgent(state_dim=10, epsilon_start=0.0, seed=42)
    state = np.random.rand(10).astype(np.float32)
    actions = [agent.select_action(state, training=True) for _ in range(10)]
    assert len(set(actions)) == 1  # all same action


# ─── Tests: Reward formula adherence ─────────────────────────────────────────

def test_reward_pct_return_formula():
    """B1 reward: r_t = (V_t - V_{t-1}) / V_{t-1}."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"STOCK0": 100.0}, index=dates)
    states = np.zeros((5, 10), dtype=np.float32)
    env = TradingEnvironment(
        prices=prices,
        states=states,
        decision_dates=dates[:5],
        initial_cash=10_000.0,
        reward_type="pct_return",
    )
    env.reset()
    env._prev_wealth = 10_000.0
    reward = env._compute_reward(10_100.0, 10_000.0)
    expected = (10_100.0 - 10_000.0) / 10_000.0
    assert abs(reward - expected) < 1e-10


def test_reward_wealth_change_formula():
    """B2 reward: r_t = V_t - V_{t-1}."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"STOCK0": 100.0}, index=dates)
    states = np.zeros((5, 10), dtype=np.float32)
    env = TradingEnvironment(
        prices=prices,
        states=states,
        decision_dates=dates[:5],
        initial_cash=10_000.0,
        reward_type="wealth_change",
    )
    env.reset()
    reward = env._compute_reward(10_500.0, 10_000.0)
    assert abs(reward - 500.0) < 1e-10
