"""Base class shared by the tabular agents.

Holds the epsilon-greedy machinery, the epsilon decay schedule and
(de)serialization.  Sub-classes only need to provide the Q-value lookup, the
TD update rule and a ``state_dict`` of the arrays to persist.
"""

from __future__ import annotations

import numpy as np


class TabularAgent:
    #: display name / file-name slug, overridden by subclasses
    name: str = "Tabular"
    slug: str = "tabular"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def _argmax(self, values: np.ndarray) -> int:
        """Greedy action with random tie-breaking.

        Random tie-breaking matters early in training when many Q-values are
        identical (e.g. all zero): plain ``np.argmax`` would always pick
        action 0 and bias exploration.
        """
        max_v = values.max()
        candidates = np.flatnonzero(values == max_v)
        if candidates.size == 1:
            return int(candidates[0])
        return int(self.rng.choice(candidates))

    def q_values(self, state: int) -> np.ndarray:
        """Return the action-value vector used for greedy decisions."""
        raise NotImplementedError

    def greedy_action(self, state: int) -> int:
        """Best action under the current value estimate (used at evaluation)."""
        return self._argmax(self.q_values(state))

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action used during training."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    # ------------------------------------------------------------------ #
    # Schedules / learning
    # ------------------------------------------------------------------ #
    def decay_epsilon(self) -> None:
        """Multiplicative per-episode decay, clipped at ``epsilon_min``."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, terminated) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, np.ndarray]:
        """Arrays to persist; subclasses override."""
        raise NotImplementedError

    def load_state_dict(self, data) -> None:
        raise NotImplementedError

    def q_table(self) -> np.ndarray:
        """A single (n_states, n_actions) table representing the greedy policy."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist the agent's tables to ``path`` (``.npz`` is appended)."""
        np.savez(path, **self.state_dict())

    def load(self, path: str) -> "TabularAgent":
        with np.load(path) as data:
            self.load_state_dict(data)
        return self
