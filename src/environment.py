"""FrozenLake environment factory and small descriptive helpers."""

from __future__ import annotations

import gymnasium as gym

# Human-readable action names (matches Gymnasium's FrozenLake action encoding).
ACTIONS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

# The standard fixed maps used by FrozenLake-v1.
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


def make_env(
    map_name: str = "4x4",
    is_slippery: bool = False,
    render_mode: str | None = None,
    max_steps: int | None = None,
) -> gym.Env:
    """Create a ``FrozenLake-v1`` environment.

    Parameters
    ----------
    map_name:     "4x4" or "8x8".
    is_slippery:  stochastic transitions when True.
    render_mode:  None, "ansi", "human" or "rgb_array".
    max_steps:    optional override of the built-in TimeLimit.
    """
    kwargs = dict(map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    if max_steps is not None:
        return gym.make("FrozenLake-v1", max_episode_steps=max_steps, **kwargs)
    return gym.make("FrozenLake-v1", **kwargs)


def state_to_rowcol(state: int, ncols: int) -> tuple[int, int]:
    """Convert a flat FrozenLake state index to (row, col) grid coordinates."""
    return divmod(state, ncols)
