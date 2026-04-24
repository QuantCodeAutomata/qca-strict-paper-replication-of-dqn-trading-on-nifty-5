"""
Tests for DQN agent and replay buffer.

Tests verify:
- Replay buffer push/sample operations
- Q-network forward pass
- Epsilon-greedy action selection
- Training loop mechanics
- Model save/load
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from agent.dqn_agent import DQNAgent, QNetwork
from agent.replay_buffer import ReplayBuffer, Transition
from environment.trading_env import TradingEnvironment
import pandas as pd


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_simple_env(n_steps: int = 30, state_dim: int = 10) -> TradingEnvironment:
    """Create a simple environment for agent testing."""
    dates = pd.date_range("2020-01-01", periods=n_steps + 5, freq="B")
    prices = pd.DataFrame({"STOCK0": 100.0}, index=dates)
    states = np.random.rand(n_steps, state_dim).astype(np.float32)
    decision_dates = dates[:n_steps]
    return TradingEnvironment(
        prices=prices,
        states=states,
        decision_dates=decision_dates,
        initial_cash=10_000.0,
    )


# ─── Tests: ReplayBuffer ─────────────────────────────────────────────────────

def test_replay_buffer_push_and_len():
    """Buffer length should increase with each push."""
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.push(np.zeros(5), 0, 1.0, np.zeros(5), False)
    assert len(buf) == 1
    buf.push(np.zeros(5), 1, -0.5, np.zeros(5), True)
    assert len(buf) == 2


def test_replay_buffer_capacity():
    """Buffer should not exceed capacity."""
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(np.zeros(3), 0, float(i), np.zeros(3), False)
    assert len(buf) == 5


def test_replay_buffer_sample_size():
    """Sample should return exactly batch_size transitions."""
    buf = ReplayBuffer(capacity=100)
    for _ in range(50):
        buf.push(np.zeros(5), 0, 1.0, np.zeros(5), False)
    batch = buf.sample(10)
    assert len(batch) == 10


def test_replay_buffer_sample_insufficient_raises():
    """Sampling more than available should raise ValueError."""
    buf = ReplayBuffer(capacity=100)
    buf.push(np.zeros(5), 0, 1.0, np.zeros(5), False)
    with pytest.raises(ValueError):
        buf.sample(10)


def test_replay_buffer_is_ready():
    """is_ready should return True only when enough transitions exist."""
    buf = ReplayBuffer(capacity=100)
    assert not buf.is_ready(10)
    for _ in range(10):
        buf.push(np.zeros(5), 0, 1.0, np.zeros(5), False)
    assert buf.is_ready(10)
    assert not buf.is_ready(11)


def test_replay_buffer_sample_arrays_shapes():
    """sample_arrays should return correct shapes."""
    buf = ReplayBuffer(capacity=100)
    state_dim = 8
    for _ in range(50):
        buf.push(np.zeros(state_dim), 1, 0.5, np.ones(state_dim), False)
    states, actions, rewards, next_states, dones = buf.sample_arrays(16)
    assert states.shape == (16, state_dim)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_states.shape == (16, state_dim)
    assert dones.shape == (16,)


def test_replay_buffer_transition_types():
    """Stored transitions should have correct types."""
    buf = ReplayBuffer(capacity=10)
    buf.push(np.array([1.0, 2.0]), 2, -0.3, np.array([3.0, 4.0]), True)
    t = buf.buffer[0]
    assert isinstance(t, Transition)
    assert isinstance(t.state, np.ndarray)
    assert isinstance(t.action, int)
    assert isinstance(t.reward, float)
    assert isinstance(t.done, bool)


# ─── Tests: QNetwork ─────────────────────────────────────────────────────────

def test_qnetwork_output_shape():
    """Q-network should output (batch, n_actions) tensor."""
    net = QNetwork(state_dim=10, n_actions=3)
    x = torch.randn(4, 10)
    out = net(x)
    assert out.shape == (4, 3)


def test_qnetwork_single_input():
    """Q-network should handle single input."""
    net = QNetwork(state_dim=5, n_actions=3)
    x = torch.randn(1, 5)
    out = net(x)
    assert out.shape == (1, 3)


def test_qnetwork_custom_hidden():
    """Q-network should respect custom hidden layer sizes."""
    net = QNetwork(state_dim=20, n_actions=3, hidden1=64, hidden2=32)
    x = torch.randn(8, 20)
    out = net(x)
    assert out.shape == (8, 3)


# ─── Tests: DQNAgent ─────────────────────────────────────────────────────────

def test_agent_initialization():
    """Agent should initialize with correct parameters."""
    agent = DQNAgent(state_dim=10, n_actions=3, learning_rate=0.001, gamma=0.99)
    assert agent.state_dim == 10
    assert agent.n_actions == 3
    assert agent.epsilon == 1.0


def test_agent_select_action_training():
    """Action should be in valid range during training."""
    agent = DQNAgent(state_dim=10, n_actions=3)
    state = np.random.rand(10).astype(np.float32)
    for _ in range(20):
        action = agent.select_action(state, training=True)
        assert 0 <= action <= 2


def test_agent_select_action_greedy():
    """Greedy action (training=False) should be deterministic."""
    agent = DQNAgent(state_dim=10, n_actions=3, seed=42)
    state = np.random.rand(10).astype(np.float32)
    action1 = agent.select_action(state, training=False)
    action2 = agent.select_action(state, training=False)
    assert action1 == action2


def test_agent_epsilon_decay():
    """Epsilon should decay after each call to decay_epsilon."""
    agent = DQNAgent(state_dim=10, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9)
    initial_eps = agent.epsilon
    agent.decay_epsilon()
    assert agent.epsilon < initial_eps
    assert agent.epsilon >= 0.01


def test_agent_epsilon_floor():
    """Epsilon should not go below epsilon_min."""
    agent = DQNAgent(state_dim=10, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.1)
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon >= 0.05


def test_agent_update_returns_none_when_buffer_empty():
    """Update should return None when buffer is not ready."""
    agent = DQNAgent(state_dim=10, batch_size=64)
    result = agent.update()
    assert result is None


def test_agent_update_returns_loss_when_ready():
    """Update should return a float loss when buffer is ready."""
    agent = DQNAgent(state_dim=10, batch_size=16, buffer_capacity=100)
    state = np.zeros(10, dtype=np.float32)
    for _ in range(20):
        agent.replay_buffer.push(state, 0, 1.0, state, False)
    loss = agent.update()
    assert loss is not None
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_agent_train_short_episode():
    """Agent should complete training on a short environment."""
    env = make_simple_env(n_steps=20, state_dim=10)
    agent = DQNAgent(state_dim=10, batch_size=8, buffer_capacity=50, seed=42)
    history = agent.train(env, n_episodes=5, log_interval=5)
    assert len(history["episode_rewards"]) == 5
    assert len(history["episode_losses"]) == 5
    assert len(history["epsilon_history"]) == 5


def test_agent_evaluate_returns_results():
    """Evaluate should return history, wealth, returns, and trade log."""
    env = make_simple_env(n_steps=20, state_dim=10)
    agent = DQNAgent(state_dim=10, batch_size=8, buffer_capacity=50, seed=42)
    agent.train(env, n_episodes=3, log_interval=3)

    test_env = make_simple_env(n_steps=15, state_dim=10)
    results = agent.evaluate(test_env)

    assert "history" in results
    assert "wealth_series" in results
    assert "daily_returns" in results
    assert "trade_log" in results


def test_agent_save_load():
    """Agent should save and load correctly."""
    agent = DQNAgent(state_dim=10, n_actions=3, seed=42)
    state = np.random.rand(10).astype(np.float32)
    action_before = agent.select_action(state, training=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        agent.save(path)

        agent2 = DQNAgent(state_dim=10, n_actions=3, seed=99)
        agent2.load(path)
        action_after = agent2.select_action(state, training=False)

    assert action_before == action_after


def test_agent_paper_hyperparameters():
    """Agent should accept paper-specified hyperparameters."""
    agent = DQNAgent(
        state_dim=10,
        learning_rate=0.001,   # paper-explicit
        gamma=0.99,            # paper-explicit
    )
    assert agent.gamma == 0.99
    # Check optimizer learning rate
    for pg in agent.optimizer.param_groups:
        assert abs(pg["lr"] - 0.001) < 1e-10
