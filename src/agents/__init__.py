"""Tabular RL agents implemented from scratch."""

from .base import TabularAgent
from .q_learning import QLearningAgent
from .double_q_learning import DoubleQLearningAgent

AGENTS = {
    "q_learning": QLearningAgent,
    "double_q_learning": DoubleQLearningAgent,
}

# FrozenLake always exposes 4 discrete actions.
N_ACTIONS = 4


def load_trained_agent(algo: str, n_states: int, qpath, n_actions: int = N_ACTIONS):
    """Build an agent of type ``algo`` and load a saved Q-table from ``qpath``.

    Exploration hyper-parameters are irrelevant for a loaded, greedy-only agent,
    so they are set to inert values.
    """
    agent = AGENTS[algo](
        n_states=n_states, n_actions=n_actions,
        alpha=0.0, gamma=0.0,
        epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=1.0,
    )
    agent.load(str(qpath))
    return agent


__all__ = [
    "TabularAgent", "QLearningAgent", "DoubleQLearningAgent",
    "AGENTS", "N_ACTIONS", "load_trained_agent",
]
