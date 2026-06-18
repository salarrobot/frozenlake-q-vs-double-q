"""Tabular Q-Learning.

Update rule (off-policy TD control):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

The bootstrap term is dropped when the transition is *terminal* (goal reached
or hole entered).  Time-limit truncation is intentionally **not** treated as
terminal, so the future value is still bootstrapped in that case.
"""

from __future__ import annotations

import numpy as np

from .base import TabularAgent


class QLearningAgent(TabularAgent):
    name = "Q-Learning"
    slug = "q_learning"

    def __init__(self, n_states: int, n_actions: int, **kwargs) -> None:
        super().__init__(n_states, n_actions, **kwargs)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def q_values(self, state: int) -> np.ndarray:
        return self.Q[state]

    def update(self, state, action, reward, next_state, terminated) -> None:
        best_next = 0.0 if terminated else self.Q[next_state].max()
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    # ----- persistence ----- #
    def state_dict(self) -> dict[str, np.ndarray]:
        return {"Q": self.Q}

    def load_state_dict(self, data) -> None:
        self.Q = np.array(data["Q"], dtype=np.float64)

    def q_table(self) -> np.ndarray:
        return self.Q
