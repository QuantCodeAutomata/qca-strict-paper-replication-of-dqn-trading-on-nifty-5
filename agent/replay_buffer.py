"""
Experience replay buffer for DQN training.

Paper-explicit: DQN uses a replay buffer (experience replay).
Capacity and batch size are assumptions documented in assumptions.md.
"""

import random
from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    """Single experience transition stored in replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer.

    Stores (state, action, reward, next_state, done) transitions and
    supports uniform random sampling for DQN training.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    seed : int
        Random seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 10_000, seed: int = 42) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Next state vector (can be zeros if done=True).
        done : bool
            Whether the episode ended.
        """
        self.buffer.append(
            Transition(
                state=np.array(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.array(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Sample a random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        list of Transition
            Randomly sampled transitions.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self.buffer)} transitions, "
                f"cannot sample {batch_size}"
            )
        return self._rng.sample(list(self.buffer), batch_size)

    def sample_arrays(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch and return as separate numpy arrays.

        Returns
        -------
        tuple of (states, actions, rewards, next_states, dones)
        """
        batch = self.sample(batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough transitions to sample."""
        return len(self.buffer) >= batch_size
