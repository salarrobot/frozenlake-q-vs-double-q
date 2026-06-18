"""Visual demonstration of a trained FrozenLake agent.

Loads a saved Q-table, runs the greedy policy and prints, for every step, the
current state, the chosen action, the running reward and finally the
success/failure result.  By default it renders the grid as text (``ansi``) so it
works anywhere; pass ``--render human`` to open a pygame window (requires
``pygame``).

Examples
--------
    python demo.py --experiment 4x4_slippery --algo double_q_learning
    python demo.py --experiment 8x8_deterministic --algo q_learning --episodes 3
    python demo.py --experiment 4x4_deterministic --algo q_learning --render human
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.agents import AGENTS, load_trained_agent
from src.config import config_by_name
from src.environment import ACTIONS, make_env, state_to_rowcol
from src.record import record_agent_video
from src.utils import ensure_dir

RESULTS_DIR = "results"
VIDEOS_DIR = "videos"


def load_agent(algo: str, config, qpath: Path):
    return load_trained_agent(algo, config.n_states, qpath)


def run_episode(agent, env, config, render: str, pause: float, ep_idx: int) -> bool:
    ncols = 4 if config.map_name == "4x4" else 8
    state, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    step = 0

    print(f"\n{'#' * 56}\n# Episode {ep_idx + 1}\n{'#' * 56}")
    _render(env, render)

    while not (terminated or truncated):
        action = agent.greedy_action(state)
        row, col = state_to_rowcol(state, ncols)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1

        print(f"  step {step:>3} | state {state:>2} (row {row}, col {col}) "
              f"| action {action} = {ACTIONS[action]:<5} "
              f"| reward {reward:.0f} | total {total_reward:.0f}")
        _render(env, render)
        state = next_state
        if pause:
            time.sleep(pause)

    success = total_reward > 0
    outcome = "REACHED GOAL [SUCCESS]" if success else (
        "TIMED OUT" if truncated else "FELL IN HOLE [FAIL]")
    print(f"  --> {outcome}  (steps={step}, reward={total_reward:.0f})")
    return success


def _render(env, render: str) -> None:
    if render == "ansi":
        frame = env.render()
        if frame:
            print("    " + str(frame).replace("\n", "\n    "))
    # "human"/"rgb_array" render to a window / buffer on env.step automatically.


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained FrozenLake agent.")
    parser.add_argument("--experiment", required=True,
                        help="e.g. 4x4_slippery, 8x8_deterministic")
    parser.add_argument("--algo", choices=list(AGENTS), default="double_q_learning")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", choices=["ansi", "human"], default="ansi")
    parser.add_argument("--pause", type=float, default=0.4,
                        help="seconds to wait between steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", action="store_true",
                        help="also save an MP4 of the run to videos/")
    args = parser.parse_args()

    config = config_by_name(args.experiment)
    qpath = Path(RESULTS_DIR) / config.name / f"{args.algo}_qtable.npz"
    if not qpath.exists():
        raise SystemExit(
            f"No trained model at {qpath}.\n"
            f"Train it first, e.g.:  python main.py run --experiment {args.experiment}")

    agent = load_agent(args.algo, config, qpath)

    if args.video:
        out = ensure_dir(VIDEOS_DIR) / f"{config.name}_{args.algo}"
        record_agent_video(agent, config, str(out), n_episodes=args.episodes,
                           seed=args.seed)
    render_mode = "ansi" if args.render == "ansi" else "human"
    env = make_env(config.map_name, config.is_slippery, render_mode=render_mode)
    seed = args.seed if args.seed is not None else config.seed + 99_000
    env.reset(seed=seed)
    env.action_space.seed(seed)

    print(f"Loaded {AGENTS[args.algo].name} for experiment '{config.name}'")
    print(f"Map: FrozenLake {config.map_name}, slippery={config.is_slippery}")

    successes = 0
    for ep in range(args.episodes):
        successes += run_episode(agent, env, config, args.render, args.pause, ep)

    env.close()
    print(f"\n{'=' * 56}")
    print(f"Demo complete: {successes}/{args.episodes} episodes reached the goal "
          f"({successes / args.episodes:.0%}).")


if __name__ == "__main__":
    main()
