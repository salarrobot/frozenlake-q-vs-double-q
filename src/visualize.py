"""Publication-quality plots for a single experiment (both algorithms).

All figures carry a title, axis labels, a legend and grid lines and are written
to ``plots/<experiment_name>/``.  The six required plot types are:

    1. Reward vs Training Episodes        -> reward_vs_episodes.png
    2. Success Rate vs Episodes           -> success_rate_vs_episodes.png
    3. Moving Average Reward              -> moving_average_reward.png
    4. Epsilon Decay Curve               -> epsilon_decay.png
    5. Q-Learning vs Double Q-Learning   -> comparison.png   (multi-panel)
    6. Convergence Comparison            -> convergence_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file-only backend
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .utils import ensure_dir, rolling_mean

# Consistent per-algorithm styling across every figure.
STYLE = {
    "q_learning": {"label": "Q-Learning", "color": "#1f77b4"},
    "double_q_learning": {"label": "Double Q-Learning", "color": "#d62728"},
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _title_suffix(config: ExperimentConfig) -> str:
    kind = "Slippery / Stochastic" if config.is_slippery else "Deterministic"
    return f"FrozenLake {config.map_name} — {kind}"


# --------------------------------------------------------------------------- #
# 1. Reward vs episodes
# --------------------------------------------------------------------------- #
def plot_reward_vs_episodes(runs: dict, config: ExperimentConfig, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    window = config.success_window
    for algo, data in runs.items():
        stats = data["stats"]
        ep = stats["episode"].to_numpy()
        style = STYLE[algo]
        smooth = rolling_mean(stats["reward"].to_numpy(), window)
        ax.plot(ep, stats["reward"].to_numpy(), color=style["color"], alpha=0.12, lw=0.6)
        ax.plot(ep, smooth, color=style["color"], lw=2,
                label=f"{style['label']} ({window}-ep mean)")
    ax.set_title(f"Reward vs Training Episodes\n{_title_suffix(config)}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    return _save(fig, out / "reward_vs_episodes.png")


# --------------------------------------------------------------------------- #
# 2. Success rate vs episodes
# --------------------------------------------------------------------------- #
def plot_success_rate(runs: dict, config: ExperimentConfig, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    window = config.success_window
    for algo, data in runs.items():
        stats = data["stats"]
        ep = stats["episode"].to_numpy()
        style = STYLE[algo]
        rate = rolling_mean(stats["success"].to_numpy(), window)
        ax.plot(ep, rate, color=style["color"], lw=2, label=style["label"])
    ax.axhline(config.convergence_threshold, color="gray", ls="--", lw=1,
               label=f"Convergence threshold ({config.convergence_threshold:.0%})")
    ax.set_title(f"Success Rate vs Episodes ({window}-episode rolling)\n{_title_suffix(config)}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    return _save(fig, out / "success_rate_vs_episodes.png")


# --------------------------------------------------------------------------- #
# 3. Moving average reward
# --------------------------------------------------------------------------- #
def plot_moving_average_reward(runs: dict, config: ExperimentConfig, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    window = config.success_window
    for algo, data in runs.items():
        stats = data["stats"]
        ep = stats["episode"].to_numpy()
        style = STYLE[algo]
        smooth = rolling_mean(stats["reward"].to_numpy(), window)
        ax.plot(ep, smooth, color=style["color"], lw=2, label=style["label"])
    ax.set_title(f"Moving Average Reward ({window}-episode window)\n{_title_suffix(config)}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean Reward (last {window} episodes)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    return _save(fig, out / "moving_average_reward.png")


# --------------------------------------------------------------------------- #
# 4. Epsilon decay curve
# --------------------------------------------------------------------------- #
def plot_epsilon_decay(runs: dict, config: ExperimentConfig, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    # Both algorithms share the same schedule; plot the first available run.
    data = next(iter(runs.values()))
    stats = data["stats"]
    ax.plot(stats["episode"].to_numpy(), stats["epsilon"].to_numpy(),
            color="#2ca02c", lw=2, label="Epsilon (ε)")
    ax.axhline(config.epsilon_min, color="gray", ls="--", lw=1,
               label=f"ε_min = {config.epsilon_min}")
    ax.set_title(f"Epsilon Decay Schedule\n{_title_suffix(config)}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration Rate ε")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    return _save(fig, out / "epsilon_decay.png")


# --------------------------------------------------------------------------- #
# 5. Multi-panel Q vs Double-Q comparison
# --------------------------------------------------------------------------- #
def plot_comparison(runs: dict, config: ExperimentConfig, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    window = config.success_window
    (ax1, ax2), (ax3, ax4) = axes

    algos = list(runs.keys())
    mean_vals, peak_vals, colors = [], [], []
    for algo in algos:
        data = runs[algo]
        stats = data["stats"]
        ep = stats["episode"].to_numpy()
        style = STYLE[algo]
        c = style["color"]
        ax1.plot(ep, rolling_mean(stats["reward"].to_numpy(), window),
                 color=c, lw=2, label=style["label"])
        ax2.plot(ep, rolling_mean(stats["success"].to_numpy(), window),
                 color=c, lw=2, label=style["label"])
        ax3.plot(ep, rolling_mean(stats["length"].to_numpy(), window),
                 color=c, lw=2, label=style["label"])
        # Overestimation diagnostics from the learned table:
        #   mean state value V(s)=max_a Q(s,a) accumulates the bias across all
        #   states; peak Q is the single largest entry.
        q = data["agent"].q_table()
        mean_vals.append(float(q.max(axis=1).mean()))
        peak_vals.append(float(q.max()))
        colors.append(c)

    ax1.set_title("Moving Average Reward")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Mean Reward"); ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="lower right")

    ax2.set_title("Success Rate")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Success Rate"); ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="lower right")

    ax3.set_title("Episode Length")
    ax3.set_xlabel("Episode"); ax3.set_ylabel(f"Mean Steps (last {window})")
    ax3.legend(loc="upper right")

    # Grouped bars: mean V(s) and peak Q for each algorithm.
    x = np.arange(2)  # 0 -> mean V(s), 1 -> peak Q
    width = 0.38
    offsets = np.linspace(-width / 2, width / 2, len(algos))
    for i, algo in enumerate(algos):
        vals = [mean_vals[i], peak_vals[i]]
        bars = ax4.bar(x + offsets[i], vals, width=width / len(algos) * 1.6,
                       color=colors[i], alpha=0.85, label=STYLE[algo]["label"])
        for b, v in zip(bars, vals):
            ax4.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(["mean V(s) = mean$_s$ max$_a$ Q", "peak  max Q(s,a)"])
    ax4.set_title("Value Estimates  (overestimation diagnostic)")
    ax4.set_ylabel("Estimated value")
    ax4.legend(loc="upper left")

    fig.suptitle(f"Q-Learning vs Double Q-Learning — {_title_suffix(config)}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "comparison.png")


# --------------------------------------------------------------------------- #
# 6. Convergence comparison
# --------------------------------------------------------------------------- #
def plot_convergence(runs: dict, summaries: dict, config: ExperimentConfig,
                     out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    window = config.success_window

    # Left: smoothed success curves with the convergence episode marked.
    for algo, data in runs.items():
        stats = data["stats"]
        ep = stats["episode"].to_numpy()
        style = STYLE[algo]
        rate = rolling_mean(stats["success"].to_numpy(), window)
        ax1.plot(ep, rate, color=style["color"], lw=2, label=style["label"])
        conv = summaries[algo].get("convergence_episode")
        if conv is not None:
            ax1.axvline(conv, color=style["color"], ls=":", lw=1.5)
            ax1.scatter([conv], [rate[conv]], color=style["color"], zorder=5)
    ax1.axhline(config.convergence_threshold, color="gray", ls="--", lw=1)
    ax1.set_title("Convergence of Success Rate")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Success Rate"); ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="lower right")

    # Right: episodes-to-threshold and training time side by side.
    algos = list(runs.keys())
    labels = [STYLE[a]["label"] for a in algos]
    colors = [STYLE[a]["color"] for a in algos]
    conv_eps = [summaries[a].get("convergence_episode") or np.nan for a in algos]
    times = [summaries[a].get("train_time_s", np.nan) for a in algos]

    x = np.arange(len(algos))
    b1 = ax2.bar(x - 0.2, conv_eps, width=0.4, color="#4c72b0", alpha=0.9,
                 label="Episodes to converge")
    ax2b = ax2.twinx()
    b2 = ax2b.bar(x + 0.2, times, width=0.4, color="#dd8452", alpha=0.9,
                  label="Training time (s)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel(f"Episodes to reach {config.convergence_threshold:.0%} success")
    ax2b.set_ylabel("Training time (s)")
    ax2b.grid(False)
    ax2.set_title("Convergence Speed & Cost")
    for xi, ce in zip(x, conv_eps):
        txt = "n/a" if np.isnan(ce) else f"{int(ce)}"
        ax2.text(xi - 0.2, (0 if np.isnan(ce) else ce), txt, ha="center", va="bottom")
    for xi, t in zip(x, times):
        if not np.isnan(t):
            ax2b.text(xi + 0.2, t, f"{t:.0f}s", ha="center", va="bottom")
    ax2.legend(handles=[b1, b2], loc="upper left")

    fig.suptitle(f"Convergence Comparison — {_title_suffix(config)}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out / "convergence_comparison.png")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate_all_plots(runs: dict, summaries: dict, config: ExperimentConfig,
                       plots_dir: str) -> list[Path]:
    """Generate the full set of figures for one experiment.

    ``runs`` maps ``algo -> {"stats": DataFrame, "agent": TabularAgent}``.
    ``summaries`` maps ``algo -> training-summary dict``.
    """
    out = ensure_dir(Path(plots_dir) / config.name)
    paths = [
        plot_reward_vs_episodes(runs, config, out),
        plot_success_rate(runs, config, out),
        plot_moving_average_reward(runs, config, out),
        plot_epsilon_decay(runs, config, out),
        plot_comparison(runs, config, out),
        plot_convergence(runs, summaries, config, out),
    ]
    return paths


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
