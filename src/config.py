"""Experiment configuration.

A single :class:`ExperimentConfig` dataclass carries every hyper-parameter for a
run.  Sensible, environment-aware defaults are chosen automatically (e.g. lower
learning rates and higher discount for stochastic / larger maps), while every
value can still be overridden explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    # ----- Environment ----------------------------------------------------- #
    map_name: str = "4x4"            # "4x4" or "8x8"
    is_slippery: bool = False        # stochastic transitions when True
    max_steps: int | None = None     # None -> Gymnasium default TimeLimit

    # ----- Training -------------------------------------------------------- #
    n_episodes: int = 10_000
    alpha: float = 0.8               # learning rate
    gamma: float = 0.95              # discount factor

    # ----- Epsilon-greedy exploration ------------------------------------- #
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float | None = None    # per-episode multiplier (auto if None)
    decay_fraction: float = 0.6      # reach epsilon_min after this fraction of episodes

    # ----- Evaluation ------------------------------------------------------ #
    eval_episodes: int = 1_000
    success_window: int = 100        # rolling window for success-rate curves
    convergence_threshold: float | None = None   # auto by slipperiness

    # ----- Reproducibility ------------------------------------------------- #
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epsilon_decay is None:
            # Geometric decay that reaches epsilon_min after
            # ``decay_fraction * n_episodes`` episodes:  decay**steps == ratio.
            steps = max(1, int(self.decay_fraction * self.n_episodes))
            ratio = self.epsilon_min / self.epsilon_start
            self.epsilon_decay = ratio ** (1.0 / steps)
        if self.convergence_threshold is None:
            # Slippery FrozenLake has a much lower attainable success rate, so a
            # lower convergence bar is appropriate.
            self.convergence_threshold = 0.40 if self.is_slippery else 0.90

    # ----- Convenience ----------------------------------------------------- #
    @property
    def name(self) -> str:
        suffix = "slippery" if self.is_slippery else "deterministic"
        return f"{self.map_name}_{suffix}"

    @property
    def n_states(self) -> int:
        return 16 if self.map_name == "4x4" else 64

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Named experiment factory
# --------------------------------------------------------------------------- #
def make_config(map_name: str, is_slippery: bool, **overrides) -> ExperimentConfig:
    """Build a config with map/stochasticity-aware default hyper-parameters."""
    if map_name == "4x4":
        n_episodes = 10_000 if not is_slippery else 20_000
    elif map_name == "8x8":
        n_episodes = 50_000
    else:
        raise ValueError(f"Unknown map_name {map_name!r} (use '4x4' or '8x8')")

    if is_slippery:
        # Stochastic transitions: small, stable steps and a high discount so the
        # single terminal reward propagates back along long, noisy paths.
        alpha, gamma = 0.10, 0.99
        epsilon_min = 0.05
    else:
        alpha = 0.80
        gamma = 0.95 if map_name == "4x4" else 0.99
        epsilon_min = 0.01

    defaults = dict(
        map_name=map_name,
        is_slippery=is_slippery,
        n_episodes=n_episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon_min=epsilon_min,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# The default experiment matrix used by ``main.py all``.
DEFAULT_EXPERIMENTS = [
    ("4x4", False),
    ("4x4", True),
    ("8x8", False),
    ("8x8", True),
]


def default_configs() -> list[ExperimentConfig]:
    return [make_config(m, s) for m, s in DEFAULT_EXPERIMENTS]


def config_by_name(name: str) -> ExperimentConfig:
    """Resolve an experiment name such as ``8x8_slippery`` to a config."""
    mapping = {make_config(m, s).name: (m, s) for m, s in DEFAULT_EXPERIMENTS}
    if name not in mapping:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown experiment {name!r}. Valid names: {valid}")
    m, s = mapping[name]
    return make_config(m, s)


ALGORITHMS = ["q_learning", "double_q_learning"]
