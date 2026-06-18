"""Greedy-policy evaluation of a trained agent."""

from __future__ import annotations

import numpy as np

from .agents.base import TabularAgent
from .config import ExperimentConfig
from .environment import make_env


def evaluate(agent: TabularAgent, config: ExperimentConfig,
             n_episodes: int | None = None, seed: int | None = None) -> dict:
    """Run the greedy (epsilon = 0) policy and report aggregate metrics.

    A separate seed offset is used so evaluation episodes differ from the
    seeded training episodes while remaining reproducible.
    """
    n_episodes = n_episodes or config.eval_episodes
    seed = config.seed + 10_000 if seed is None else seed

    env = make_env(config.map_name, config.is_slippery, max_steps=config.max_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    rewards = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int32)

    for i in range(n_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        total = 0.0
        steps = 0
        while not (terminated or truncated):
            action = agent.greedy_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
        rewards[i] = total
        lengths[i] = steps

    env.close()

    success = rewards > 0
    return {
        "eval_episodes": n_episodes,
        "avg_reward": float(rewards.mean()),
        "success_rate": float(success.mean()),
        "avg_length": float(lengths.mean()),
        # Average steps for *successful* episodes only (None if never succeeds).
        "avg_length_success": float(lengths[success].mean()) if success.any() else None,
        "reward_std": float(rewards.std()),
    }
