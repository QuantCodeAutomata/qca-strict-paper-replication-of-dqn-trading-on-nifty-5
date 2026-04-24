"""
Deep Q-Network (DQN) agent for trading.

Paper-explicit components:
- Epsilon-greedy exploration
- Experience replay buffer
- Learning rate: 0.001
- Discount factor: 0.99
- Training episodes: 10,000

Assumptions (frozen, documented in assumptions.md):
- Network: Linear(state_dim,128)->ReLU->Linear(128,64)->ReLU->Linear(64,3)
- Optimizer: Adam
- Batch size: 64
- Replay buffer capacity: 10,000
- No target network (simplified)
- Epsilon: 1.0 → 0.01, decay 0.995 per episode
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.replay_buffer import ReplayBuffer
from environment.trading_env import TradingEnvironment

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Q-value network for DQN.

    Architecture (assumption — paper does not specify):
    Linear(state_dim, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, n_actions)

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector.
    n_actions : int
        Number of discrete actions (3: hold, buy, sell).
    hidden1 : int
        Size of first hidden layer.
    hidden2 : int
        Size of second hidden layer.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        hidden1: int = 128,
        hidden2: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for each action."""
        return self.net(x)


class DQNAgent:
    """
    DQN agent with epsilon-greedy exploration and experience replay.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector.
    n_actions : int
        Number of discrete actions.
    learning_rate : float
        Adam optimizer learning rate (paper: 0.001).
    gamma : float
        Discount factor (paper: 0.99).
    epsilon_start : float
        Initial exploration rate.
    epsilon_min : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiplicative decay per episode.
    batch_size : int
        Mini-batch size for Q-network updates.
    buffer_capacity : int
        Replay buffer capacity.
    hidden1 : int
        First hidden layer size.
    hidden2 : int
        Second hidden layer size.
    device : str
        PyTorch device ('cpu' or 'cuda').
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        hidden1: int = 128,
        hidden2: int = 64,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = seed

        # Reproducibility
        self._set_seeds(seed)

        # Q-network (single network — no target network, assumption)
        self.q_network = QNetwork(state_dim, n_actions, hidden1, hidden2).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        # Training metrics
        self.episode_rewards: list[float] = []
        self.episode_losses: list[float] = []
        self.epsilon_history: list[float] = []

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        training : bool
            If True, use epsilon-greedy; if False, use greedy (ε=0).

        Returns
        -------
        int
            Selected action index.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch from replay buffer and update Q-network.

        Uses Bellman target:
        y = r + γ * max_a' Q(s', a')  if not done
        y = r                          if done

        Returns
        -------
        float or None
            Training loss, or None if buffer not ready.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample_arrays(self.batch_size)
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for taken actions
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target
        with torch.no_grad():
            next_q = self.q_network(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env: TradingEnvironment,
        n_episodes: int = 10_000,
        log_interval: int = 100,
        update_every: int = 1,
    ) -> dict:
        """
        Train the DQN agent on the trading environment.

        Paper-explicit: 10,000 training episodes with epsilon-greedy exploration.

        Parameters
        ----------
        env : TradingEnvironment
            Training environment.
        n_episodes : int
            Number of training episodes (paper: 10,000).
        log_interval : int
            Frequency of logging progress.
        update_every : int
            Update Q-network every N steps.

        Returns
        -------
        dict
            Training history with episode rewards, losses, and epsilon values.
        """
        logger.info("Starting DQN training for %d episodes", n_episodes)

        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_loss_sum = 0.0
            episode_loss_count = 0
            step_count = 0

            while True:
                # Select action (epsilon-greedy)
                action = self.select_action(state, training=True)

                # Execute action in environment
                next_state, reward, done, _ = env.step(action)

                # Store transition
                next_state_stored = (
                    next_state if next_state is not None
                    else np.zeros(self.state_dim, dtype=np.float32)
                )
                self.replay_buffer.push(state, action, reward, next_state_stored, done)

                # Update Q-network
                if step_count % update_every == 0:
                    loss = self.update()
                    if loss is not None:
                        episode_loss_sum += loss
                        episode_loss_count += 1

                episode_reward += reward
                step_count += 1

                if done:
                    break
                state = next_state

            # Decay epsilon after each episode
            self.decay_epsilon()

            # Record metrics
            self.episode_rewards.append(episode_reward)
            avg_loss = episode_loss_sum / max(episode_loss_count, 1)
            self.episode_losses.append(avg_loss)
            self.epsilon_history.append(self.epsilon)

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                logger.info(
                    "Episode %d/%d | Avg Reward: %.4f | Epsilon: %.4f | Loss: %.6f",
                    episode + 1, n_episodes, avg_reward, self.epsilon, avg_loss,
                )

        logger.info("Training complete. Final epsilon: %.4f", self.epsilon)
        return {
            "episode_rewards": self.episode_rewards,
            "episode_losses": self.episode_losses,
            "epsilon_history": self.epsilon_history,
        }

    def evaluate(self, env: TradingEnvironment) -> dict:
        """
        Evaluate the trained agent deterministically (ε=0).

        Parameters
        ----------
        env : TradingEnvironment
            Test environment.

        Returns
        -------
        dict
            Evaluation results including wealth series, returns, and trade log.
        """
        state = env.reset()
        self.q_network.eval()

        while True:
            action = self.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state

        self.q_network.train()

        return {
            "history": env.get_history(),
            "wealth_series": env.get_wealth_series(),
            "daily_returns": env.get_daily_returns(),
            "trade_log": env.get_trade_log(),
        }

    def save(self, path: str) -> None:
        """
        Save model checkpoint.

        Parameters
        ----------
        path : str
            File path to save the model.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode_rewards": self.episode_rewards,
                "episode_losses": self.episode_losses,
                "epsilon_history": self.epsilon_history,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "seed": self.seed,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """
        Load model checkpoint.

        Parameters
        ----------
        path : str
            File path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_losses = checkpoint.get("episode_losses", [])
        self.epsilon_history = checkpoint.get("epsilon_history", [])
        logger.info("Model loaded from %s", path)
