"""Training loop and per-experiment orchestration."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import AGENTS
from .config import ExperimentConfig
from .environment import make_env
from .utils import ensure_dir, episodes_to_threshold, format_time, rolling_mean


def build_agent(algo: str, n_states: int, n_actions: int,
                config: ExperimentConfig, rng: np.random.Generator):
    """Instantiate an agent of type ``algo`` from a config."""
    cls = AGENTS[algo]
    return cls(
        n_states=n_states,
        n_actions=n_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        rng=rng,
    )


def train(config: ExperimentConfig, algo: str, verbose: bool = True):
    """Train one agent and return ``(agent, stats_df, train_time_seconds)``.

    ``stats_df`` has one row per episode with columns
    ``episode, reward, length, epsilon, success``.
    """
    # Dedicated RNG for the agent (action selection + Double-Q coin flip).
    rng = np.random.default_rng(config.seed)

    env = make_env(config.map_name, config.is_slippery, max_steps=config.max_steps)
    # Seed the environment's transition RNG and action space for reproducibility.
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = build_agent(algo, n_states, n_actions, config, rng)

    n = config.n_episodes
    rewards = np.zeros(n, dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int32)
    epsilons = np.zeros(n, dtype=np.float64)
    successes = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    for ep in range(n):
        state, _ = env.reset()
        epsilons[ep] = agent.epsilon
        terminated = truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Bootstrap on `terminated` only: a time-limit truncation is not a
            # true terminal state, so its future value must still count.
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            steps += 1

        rewards[ep] = total_reward
        lengths[ep] = steps
        successes[ep] = 1.0 if total_reward > 0 else 0.0
        agent.decay_epsilon()

        if verbose and (ep + 1) % max(1, n // 10) == 0:
            window = config.success_window
            recent = successes[max(0, ep + 1 - window):ep + 1].mean()
            print(
                f"  [{agent.name:<17}] ep {ep + 1:>6}/{n}  "
                f"eps={agent.epsilon:5.3f}  "
                f"success(last {window})={recent:5.1%}"
            )

    train_time = time.perf_counter() - t0
    env.close()

    stats = pd.DataFrame({
        "episode": np.arange(1, n + 1),
        "reward": rewards,
        "length": lengths,
        "epsilon": epsilons,
        "success": successes,
    })

    if verbose:
        print(f"  [{agent.name}] trained {n} episodes in {format_time(train_time)}")

    return agent, stats, train_time


def training_summary(stats: pd.DataFrame, config: ExperimentConfig,
                     train_time: float) -> dict:
    """Compute convergence speed and final-window training metrics."""
    successes = stats["success"].to_numpy()
    window = config.success_window
    conv = episodes_to_threshold(successes, window, config.convergence_threshold)
    final_success = successes[-window:].mean()
    return {
        "train_time_s": train_time,
        "convergence_episode": conv,
        "convergence_threshold": config.convergence_threshold,
        "final_train_success_rate": final_success,
    }


def run_training(config: ExperimentConfig, algo: str, results_dir: str,
                 verbose: bool = True):
    """Train ``algo`` and persist its Q-table and per-episode stats CSV."""
    exp_dir = ensure_dir(Path(results_dir) / config.name)
    agent, stats, train_time = train(config, algo, verbose=verbose)

    stats.to_csv(exp_dir / f"{algo}_training_stats.csv", index=False)
    agent.save(str(exp_dir / f"{algo}_qtable"))   # writes <algo>_qtable.npz

    summary = training_summary(stats, config, train_time)
    return agent, stats, summary
