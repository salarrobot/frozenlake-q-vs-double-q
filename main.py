"""Command-line entry point for the FrozenLake RL project.

Examples
--------
    python main.py list                          # show available experiments
    python main.py all                           # train+eval+plot every experiment
    python main.py run --experiment 4x4_slippery # one experiment, both algorithms
    python main.py train --experiment 8x8_deterministic --algo q_learning
    python main.py evaluate --experiment 4x4_slippery
    python main.py plot --experiment 4x4_slippery
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from src.agents import AGENTS, load_trained_agent
from src.config import (
    ALGORITHMS,
    ExperimentConfig,
    config_by_name,
    default_configs,
)
from src.evaluate import evaluate
from src.record import record_agent_video
from src.train import run_training, training_summary
from src.utils import ensure_dir, format_time, save_json
from src.visualize import generate_all_plots

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
VIDEOS_DIR = "videos"


# --------------------------------------------------------------------------- #
# Per-experiment pipeline
# --------------------------------------------------------------------------- #
def run_experiment(config: ExperimentConfig, algos=None, do_plots=True,
                   record_video=False, verbose=True) -> pd.DataFrame:
    """Train, evaluate, plot and persist results for one experiment."""
    algos = algos or ALGORITHMS
    print(f"\n{'=' * 70}\nEXPERIMENT: {config.name}  "
          f"(episodes={config.n_episodes}, alpha={config.alpha}, "
          f"gamma={config.gamma})\n{'=' * 70}")

    exp_results = ensure_dir(Path(RESULTS_DIR) / config.name)
    save_json(config.to_dict(), exp_results / "config.json")

    runs: dict = {}
    summaries: dict = {}
    rows = []

    for algo in algos:
        print(f"\n--- Training {AGENTS[algo].name} ---")
        agent, stats, summary = run_training(config, algo, RESULTS_DIR, verbose)

        print(f"--- Evaluating {AGENTS[algo].name} over "
              f"{config.eval_episodes} greedy episodes ---")
        metrics = evaluate(agent, config)

        runs[algo] = {"stats": stats, "agent": agent}
        summaries[algo] = summary

        row = {
            "experiment": config.name,
            "map": config.map_name,
            "is_slippery": config.is_slippery,
            "algorithm": AGENTS[algo].name,
            "algo_key": algo,
            **metrics,
            **summary,
        }
        rows.append(row)
        _print_metrics(AGENTS[algo].name, metrics, summary)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(exp_results / "summary.csv", index=False)

    if do_plots:
        print("\n--- Generating plots ---")
        paths = generate_all_plots(runs, summaries, config, PLOTS_DIR)
        for p in paths:
            print(f"  saved {p}")

    if record_video:
        print("\n--- Recording demonstration videos ---")
        out_dir = ensure_dir(VIDEOS_DIR)
        for algo in algos:
            out = out_dir / f"{config.name}_{algo}"
            record_agent_video(runs[algo]["agent"], config, str(out), n_episodes=3)

    return summary_df


def _print_metrics(name: str, metrics: dict, summary: dict) -> None:
    conv = summary.get("convergence_episode")
    conv_s = "not reached" if conv is None else f"{conv} episodes"
    t = summary.get("train_time_s")
    time_s = format_time(t) if t is not None and not math.isnan(t) else "n/a"
    print(
        f"  {name}:\n"
        f"      success rate      : {metrics['success_rate']:.1%}\n"
        f"      avg reward        : {metrics['avg_reward']:.3f}\n"
        f"      avg episode length: {metrics['avg_length']:.2f}\n"
        f"      convergence       : {conv_s}\n"
        f"      training time     : {time_s}"
    )


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
def cmd_list(_args) -> None:
    print("Available experiments:")
    for cfg in default_configs():
        print(f"  - {cfg.name:<20} episodes={cfg.n_episodes:<6} "
              f"alpha={cfg.alpha} gamma={cfg.gamma} "
              f"eps_decay={cfg.epsilon_decay:.5f}")
    print("\nAlgorithms:", ", ".join(ALGORITHMS))


def cmd_all(args) -> None:
    all_summaries = []
    for cfg in default_configs():
        all_summaries.append(run_experiment(
            cfg, do_plots=not args.no_plots, record_video=args.video))
    combined = pd.concat(all_summaries, ignore_index=True)
    out = ensure_dir(RESULTS_DIR) / "summary_all.csv"
    combined.to_csv(out, index=False)
    print(f"\nAll experiments complete. Combined summary -> {out}")
    _print_combined(combined)


def cmd_run(args) -> None:
    cfg = config_by_name(args.experiment)
    if args.episodes:
        cfg = _override_episodes(cfg, args.episodes)
    run_experiment(cfg, do_plots=not args.no_plots, record_video=args.video)


def cmd_train(args) -> None:
    cfg = config_by_name(args.experiment)
    if args.episodes:
        cfg = _override_episodes(cfg, args.episodes)
    algos = [args.algo] if args.algo else ALGORITHMS
    for algo in algos:
        run_training(cfg, algo, RESULTS_DIR)


def cmd_evaluate(args) -> None:
    cfg = config_by_name(args.experiment)
    algos = [args.algo] if args.algo else ALGORITHMS
    exp_dir = Path(RESULTS_DIR) / cfg.name
    rows = []
    for algo in algos:
        qpath = exp_dir / f"{algo}_qtable.npz"
        if not qpath.exists():
            print(f"  [skip] no trained table at {qpath} — run training first.")
            continue
        agent = _load_agent(algo, cfg, qpath)
        metrics = evaluate(agent, cfg)
        rows.append({"algorithm": AGENTS[algo].name, **metrics})
        _print_metrics(AGENTS[algo].name, metrics, {"convergence_episode": None,
                                                    "train_time_s": float("nan")})
    if rows:
        print("\n", pd.DataFrame(rows).to_string(index=False))


def cmd_video(args) -> None:
    cfg = config_by_name(args.experiment)
    algos = [args.algo] if args.algo else ALGORITHMS
    exp_dir = Path(RESULTS_DIR) / cfg.name
    out_dir = ensure_dir(VIDEOS_DIR)
    for algo in algos:
        qpath = exp_dir / f"{algo}_qtable.npz"
        if not qpath.exists():
            print(f"  [skip] no trained table at {qpath} — train it first.")
            continue
        agent = _load_agent(algo, cfg, qpath)
        out = out_dir / f"{cfg.name}_{algo}"
        record_agent_video(agent, cfg, str(out), n_episodes=args.episodes,
                           fps=args.fps, seed=args.seed, video_format=args.format,
                           graphics=args.graphics)


def cmd_plot(args) -> None:
    cfg = config_by_name(args.experiment)
    exp_dir = Path(RESULTS_DIR) / cfg.name
    # Recover training times from a previously saved summary, if present.
    train_times = {}
    summary_path = exp_dir / "summary.csv"
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        train_times = dict(zip(prev["algo_key"], prev["train_time_s"]))
    runs, summaries = {}, {}
    for algo in ALGORITHMS:
        stats_path = exp_dir / f"{algo}_training_stats.csv"
        qpath = exp_dir / f"{algo}_qtable.npz"
        if not stats_path.exists() or not qpath.exists():
            print(f"  [skip] missing artifacts for {algo} — run training first.")
            continue
        stats = pd.read_csv(stats_path)
        agent = _load_agent(algo, cfg, qpath)
        runs[algo] = {"stats": stats, "agent": agent}
        summaries[algo] = training_summary(
            stats, cfg, train_times.get(algo, float("nan")))
    if runs:
        paths = generate_all_plots(runs, summaries, cfg, PLOTS_DIR)
        for p in paths:
            print(f"  saved {p}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _override_episodes(cfg: ExperimentConfig, episodes: int) -> ExperimentConfig:
    d = cfg.to_dict()
    d["n_episodes"] = episodes
    d["epsilon_decay"] = None  # recompute decay for the new horizon
    return ExperimentConfig(**d)


def _load_agent(algo: str, cfg: ExperimentConfig, qpath: Path):
    return load_trained_agent(algo, cfg.n_states, qpath)


def _print_combined(df: pd.DataFrame) -> None:
    cols = ["experiment", "algorithm", "success_rate", "avg_reward",
            "avg_length", "convergence_episode", "train_time_s"]
    view = df[cols].copy()
    view["success_rate"] = (view["success_rate"] * 100).round(1)
    view["avg_reward"] = view["avg_reward"].round(3)
    view["avg_length"] = view["avg_length"].round(1)
    view["train_time_s"] = view["train_time_s"].round(1)
    print("\nCombined results:\n")
    print(view.to_string(index=False))


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FrozenLake Q-Learning vs Double Q-Learning research toolkit.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available experiments").set_defaults(func=cmd_list)

    pa = sub.add_parser("all", help="run every experiment end-to-end")
    pa.add_argument("--no-plots", action="store_true")
    pa.add_argument("--video", action="store_true", help="also record demo videos")
    pa.set_defaults(func=cmd_all)

    pr = sub.add_parser("run", help="train+eval+plot a single experiment")
    pr.add_argument("--experiment", required=True)
    pr.add_argument("--episodes", type=int, default=None)
    pr.add_argument("--no-plots", action="store_true")
    pr.add_argument("--video", action="store_true", help="also record demo videos")
    pr.set_defaults(func=cmd_run)

    pt = sub.add_parser("train", help="train only")
    pt.add_argument("--experiment", required=True)
    pt.add_argument("--algo", choices=ALGORITHMS, default=None)
    pt.add_argument("--episodes", type=int, default=None)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate", help="evaluate a trained agent")
    pe.add_argument("--experiment", required=True)
    pe.add_argument("--algo", choices=ALGORITHMS, default=None)
    pe.set_defaults(func=cmd_evaluate)

    pp = sub.add_parser("plot", help="regenerate plots from saved stats")
    pp.add_argument("--experiment", required=True)
    pp.set_defaults(func=cmd_plot)

    pv = sub.add_parser("video", help="record a trained agent playing as MP4/GIF")
    pv.add_argument("--experiment", required=True)
    pv.add_argument("--algo", choices=ALGORITHMS, default=None)
    pv.add_argument("--episodes", type=int, default=3)
    pv.add_argument("--fps", type=int, default=3)
    pv.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    pv.add_argument("--graphics", choices=["auto", "original", "simple"],
                    default="auto", help="sprite art (original) or coloured grid (simple)")
    pv.add_argument("--seed", type=int, default=None)
    pv.set_defaults(func=cmd_video)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
