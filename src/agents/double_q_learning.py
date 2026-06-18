"""Tabular Double Q-Learning (van Hasselt, 2010).

Two independent tables Q1 and Q2 decouple *action selection* from *action
evaluation*, which removes the systematic maximization (overestimation) bias of
plain Q-Learning.

On each step, with probability 1/2 update Q1, otherwise Q2.  When updating Q1:

    a*      = argmax_a Q1(s', a)            # select with Q1
    target  = r + gamma * Q2(s', a*)        # evaluate with Q2
    Q1(s,a) <- Q1(s,a) + alpha * [target - Q1(s,a)]

and symmetrically when updating Q2.  Because the selecting table and the
evaluating table are statistically independent, ``E[Q2(s', argmax Q1)]`` is an
unbiased estimate of the value of the greedily chosen action, instead of the
upward-biased ``max`` used by Q-Learning.

Behaviour and greedy decisions use the sum ``Q1 + Q2`` (equivalently the mean).
"""

from __future__ import annotations

import numpy as np

from .base import TabularAgent


class DoubleQLearningAgent(TabularAgent):
    name = "Double Q-Learning"
    slug = "double_q_learning"

    def __init__(self, n_states: int, n_actions: int, **kwargs) -> None:
        super().__init__(n_states, n_actions, **kwargs)
        self.Q1 = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.Q2 = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def q_values(self, state: int) -> np.ndarray:
        # Sum of both estimates drives exploration and the greedy policy.
        return self.Q1[state] + self.Q2[state]

    def update(self, state, action, reward, next_state, terminated) -> None:
        if self.rng.random() < 0.5:
            # Update Q1: select with Q1, evaluate with Q2.
            a_star = self._argmax(self.Q1[next_state])
            best_next = 0.0 if terminated else self.Q2[next_state, a_star]
            td_target = reward + self.gamma * best_next
            self.Q1[state, action] += self.alpha * (td_target - self.Q1[state, action])
        else:
            # Update Q2: select with Q2, evaluate with Q1.
            a_star = self._argmax(self.Q2[next_state])
            best_next = 0.0 if terminated else self.Q1[next_state, a_star]
            td_target = reward + self.gamma * best_next
            self.Q2[state, action] += self.alpha * (td_target - self.Q2[state, action])

    # ----- persistence ----- #
    def state_dict(self) -> dict[str, np.ndarray]:
        return {"Q1": self.Q1, "Q2": self.Q2}

    def load_state_dict(self, data) -> None:
        self.Q1 = np.array(data["Q1"], dtype=np.float64)
        self.Q2 = np.array(data["Q2"], dtype=np.float64)

    def q_table(self) -> np.ndarray:
        # Mean of the two tables is the conventional combined estimate.
        return 0.5 * (self.Q1 + self.Q2)
