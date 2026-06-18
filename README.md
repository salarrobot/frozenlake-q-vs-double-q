# FrozenLake: Q-Learning vs Double Q-Learning

A complete, from-scratch reinforcement-learning study of **tabular Q-Learning**
and **Double Q-Learning** on the [Gymnasium FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
environment. Both algorithms are implemented without any RL library
(no Stable-Baselines3, RLlib, etc.) — only `numpy` for the tables, `gymnasium`
for the environment, and `pandas`/`matplotlib` for analysis and plots.

The project trains agents on the **4×4** and **8×8** maps in both
**deterministic** (`is_slippery=False`) and **stochastic** (`is_slippery=True`)
regimes, evaluates the learned greedy policies over 1,000 test episodes, and
produces publication-quality figures comparing the two algorithms.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [The FrozenLake environment](#2-the-frozenlake-environment)
3. [Project structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Mathematical formulation: Q-Learning](#6-mathematical-formulation-q-learning)
7. [Mathematical formulation: Double Q-Learning](#7-mathematical-formulation-double-q-learning)
8. [Hyperparameters](#8-hyperparameters)
9. [Example results](#9-example-results)
10. [Generated plots & demo videos](#10-generated-plots--demo-videos)
11. [Analysis & discussion](#11-analysis--discussion)
12. [Reproducibility](#12-reproducibility)
13. [Future work](#13-future-work)
14. [References](#14-references)

---

## 1. Project overview

| Goal | Train an agent to reliably walk from **S**tart to **G**oal across a frozen lake without falling into a **H**ole. |
|------|------------------------------------------------------------------------------------------------|
| Algorithms | Tabular **Q-Learning** and **Double Q-Learning**, both written from scratch. |
| Maps | `4x4` (16 states) and `8x8` (64 states). |
| Dynamics | Deterministic (`is_slippery=False`) and stochastic (`is_slippery=True`). |
| Deliverables | Trained Q-tables, per-episode CSV statistics, six plot types per experiment, a visual demo, and this report. |

The central scientific question is **overestimation bias**: ordinary Q-Learning
uses the same values to *choose* and to *evaluate* the next action, which makes
its target a biased (upward) estimate of the true action value. Double
Q-Learning decouples selection from evaluation to remove that bias. This project
quantifies the effect, which is most visible in the **stochastic** settings.

---

## 2. The FrozenLake environment

FrozenLake is a grid-world. The agent starts at the top-left tile `S` and must
reach the bottom-right goal `G`. Tiles are either **F**rozen (safe), a **H**ole
(episode ends, reward 0), the **S**tart, or the **G**oal (episode ends,
reward +1).

```
4x4 map                 8x8 map
S F F F                 S F F F F F F F
F H F H                 F F F F F F F F
F F F H                 F F F H F F F F
H F F G                 F F F F F H F F
                        F F F H F F F F
                        F H H F F F H F
                        F H F F H F H F
                        F F F H F F F G
```

- **State space** — a single integer `0 … n_rows*n_cols-1` (16 for 4×4,
  64 for 8×8), the flattened grid index.
- **Action space** — 4 discrete actions: `0=LEFT, 1=DOWN, 2=RIGHT, 3=UP`.
- **Reward** — `+1` only on reaching the goal, `0` everywhere else (including
  falling into a hole). FrozenLake is therefore a **sparse-reward** task.
- **Episode termination** — reaching the goal or falling into a hole
  (`terminated`), or hitting the time limit (`truncated`, 100 steps on 4×4,
  200 on 8×8).
- **`is_slippery`** — when `True`, the ice is slippery: the intended action only
  succeeds with probability 1/3; with probability 1/3 each the agent instead
  slips to one of the two perpendicular directions. This turns the task into a
  genuinely stochastic MDP and is where the two algorithms differ most.

> **Bootstrapping note.** A time-limit *truncation* is **not** a true terminal
> state, so this project bootstraps the next-state value on truncation and only
> zeroes it on real *termination* (goal/hole). This is the correct treatment and
> avoids biasing values near the time horizon.

---

## 3. Project structure

```
Frozen Lake/
├── main.py                     # CLI: train / evaluate / plot / video / run / all / list
├── demo.py                     # Visual run of a trained agent (ansi / human / --video)
├── requirements.txt
├── README.md
├── src/
│   ├── config.py               # ExperimentConfig + named experiment factory
│   ├── environment.py          # FrozenLake factory, maps, action names
│   ├── train.py                # Training loop, per-episode statistics
│   ├── evaluate.py             # Greedy-policy evaluation over N episodes
│   ├── visualize.py            # The six publication-quality plot types
│   ├── record.py               # MP4/GIF recorder (original sprites or simple grid)
│   ├── utils.py                # Seeding, rolling mean, convergence metric, I/O
│   └── agents/
│       ├── base.py             # Shared ε-greedy agent (selection, decay, I/O)
│       ├── q_learning.py       # Q-Learning update rule
│       └── double_q_learning.py# Double Q-Learning update rule
├── results/                    # (generated) Q-tables, per-episode CSVs, summaries
│   └── <experiment>/
│       ├── config.json
│       ├── q_learning_training_stats.csv
│       ├── q_learning_qtable.npz
│       ├── double_q_learning_training_stats.csv
│       ├── double_q_learning_qtable.npz
│       └── summary.csv
├── plots/                      # (generated) six PNG figures per experiment
│   └── <experiment>/
│       ├── reward_vs_episodes.png
│       ├── success_rate_vs_episodes.png
│       ├── moving_average_reward.png
│       ├── epsilon_decay.png
│       ├── comparison.png
│       └── convergence_comparison.png
└── videos/                     # (generated) MP4 demos: <experiment>_<algo>.mp4
```

Experiment names follow `"{map}_{deterministic|slippery}"`, e.g.
`4x4_slippery`, `8x8_deterministic`.

---

## 4. Installation

Requires **Python 3.9+**.

```bash
# (optional) create an isolated environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

`pygame` is only needed if you want the demo's `--render human` window; the
default text (`ansi`) renderer needs nothing extra.

---

## 5. Usage

All commands are run from the project root.

### List experiments
```bash
python main.py list
```

### Train + evaluate + plot a single experiment (both algorithms)
```bash
python main.py run --experiment 4x4_slippery
```

### Run the full study (all 4 experiments, both algorithms)
```bash
python main.py all
```
This writes Q-tables, CSVs and plots for every experiment plus a combined
`results/summary_all.csv`.

### Train only (optionally one algorithm / custom episode count)
```bash
python main.py train --experiment 8x8_deterministic --algo q_learning
python main.py train --experiment 4x4_slippery --episodes 30000
```

### Evaluate a saved model over 1,000 greedy episodes
```bash
python main.py evaluate --experiment 4x4_slippery
```

### (Re)generate plots from saved statistics
```bash
python main.py plot --experiment 8x8_slippery
```

### Visual demonstration
```bash
# text rendering in the terminal (default)
python demo.py --experiment 4x4_slippery --algo double_q_learning --episodes 3

# pop-up window (requires pygame)
python demo.py --experiment 4x4_deterministic --algo q_learning --render human
```
The demo prints, for each step, the **current state**, the **chosen action**,
the **running reward**, and the final **success/failure** result.

### Video output
Record a trained agent playing as an **MP4** (or GIF). By default the video uses
the **original FrozenLake sprite artwork** (the elf, snowy ice, cracked-ice
holes and the gift-box goal) rendered off-screen via Gymnasium + pygame, with a
caption bar showing the episode, step, current state, chosen action, running
reward and the final outcome. Files are written to `videos/<experiment>_<algo>.mp4`.

```bash
# record both algorithms for one experiment (3 episodes each)
python main.py video --experiment 4x4_slippery

# one algorithm, more episodes, custom frame rate / format / seed
python main.py video --experiment 8x8_slippery --algo q_learning \
    --episodes 6 --fps 4 --format mp4 --seed 7

# choose the renderer explicitly
python main.py video --experiment 4x4_slippery --graphics original  # sprite art (default)
python main.py video --experiment 4x4_slippery --graphics simple    # coloured grid, no pygame

# also save a video straight from the demo script
python demo.py --experiment 4x4_slippery --algo double_q_learning --video

# record videos as part of the full pipeline
python main.py all --video
python main.py run --experiment 8x8_deterministic --video
```

**Renderers.** `--graphics original` (default) uses Gymnasium's bundled
FrozenLake sprites and needs `pygame`; rendering is fully off-screen (SDL
`dummy` driver) so no window pops up. `--graphics simple` uses a built-in Pillow
grid (coloured tiles + a disc marker) and needs no pygame. With `--graphics
auto` the recorder tries the sprites and silently falls back to the simple grid
if pygame is unavailable.

MP4 export uses the `imageio-ffmpeg` bundled binary, so it works without a
system-wide ffmpeg. If encoding ever fails, the recorder automatically falls
back to writing a `.gif`.

---

## 6. Mathematical formulation: Q-Learning

Q-Learning (Watkins, 1989) is an **off-policy, model-free** temporal-difference
control algorithm. It learns the optimal action-value function
$Q^*(s,a)$ — the expected discounted return of taking action $a$ in state $s$
and acting optimally thereafter — which satisfies the **Bellman optimality
equation**:

$$
Q^*(s,a) = \mathbb{E}\!\left[\, r + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s, a \right].
$$

The agent maintains a table $Q(s,a)$ and, after each transition
$(s, a, r, s')$, applies the **TD(0) update**:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\, \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD target}} - Q(s,a) \,\Big],
$$

where

- $\alpha \in (0,1]$ is the **learning rate**,
- $\gamma \in [0,1)$ is the **discount factor**,
- $r + \gamma \max_{a'} Q(s',a')$ is the **TD target**, and
- the bracketed quantity is the **TD error** $\delta$.

For a terminal transition the bootstrap term is dropped:
$\text{target} = r$.

**Exploration** uses an **ε-greedy** policy:

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \varepsilon,\\[2pt]
\arg\max_a Q(s,a), & \text{with probability } 1-\varepsilon,
\end{cases}
$$

with $\varepsilon$ annealed each episode by a geometric schedule
$\varepsilon \leftarrow \max(\varepsilon_{\min}, \varepsilon \cdot d)$, where the
decay $d$ is chosen so $\varepsilon$ reaches $\varepsilon_{\min}$ after a fixed
fraction of training.

**The overestimation problem.** Because the *same* table supplies both the
$\arg\max$ (which action) and the value of that action (the $\max$), and the
estimates are noisy, $\mathbb{E}[\max_a Q(s,a)] \ge \max_a \mathbb{E}[Q(s,a)]$
by Jensen's inequality. The target is therefore **positively biased**: noise is
systematically interpreted as signal. The bias grows with the number of actions
and with transition stochasticity.

---

## 7. Mathematical formulation: Double Q-Learning

Double Q-Learning (van Hasselt, 2010) removes the maximization bias by keeping
**two independent tables** $Q_1$ and $Q_2$ and **decoupling action selection
from action evaluation**.

On each step, choose one table to update (a fair coin flip). If $Q_1$ is chosen,
use $Q_1$ to *select* the greedy next action but $Q_2$ to *evaluate* it:

$$
a^{*} = \arg\max_{a'} Q_1(s', a'), \qquad
Q_1(s,a) \leftarrow Q_1(s,a) + \alpha\Big[\, r + \gamma\, Q_2\big(s', a^{*}\big) - Q_1(s,a) \,\Big].
$$

Symmetrically, if $Q_2$ is chosen:

$$
a^{*} = \arg\max_{a'} Q_2(s', a'), \qquad
Q_2(s,a) \leftarrow Q_2(s,a) + \alpha\Big[\, r + \gamma\, Q_1\big(s', a^{*}\big) - Q_2(s,a) \,\Big].
$$

(For a terminal transition the bootstrap term is again dropped.)

**Why this removes the bias.** Since $Q_1$ and $Q_2$ are trained on disjoint
update streams, they are statistically independent estimators of the same
values. Selecting with one and evaluating with the other gives, in expectation,

$$
\mathbb{E}\big[\, Q_2(s', a^{*}) \,\big] = \mathbb{E}\big[\, Q_2(s',\, \arg\max_{a'} Q_1(s',a')) \,\big] \;\le\; \max_{a'} \mathbb{E}\big[Q_2(s',a')\big],
$$

i.e. the target is an **unbiased** (in fact slightly conservative) estimate of
the chosen action's value, rather than the upward-biased plain $\max$. The
result is lower-variance, less inflated value estimates — particularly valuable
under stochastic transitions.

**Behaviour policy.** ε-greedy exploration and the final greedy policy use the
**sum** $Q_1(s,\cdot) + Q_2(s,\cdot)$ (equivalently their mean), so both
estimators contribute to decisions.

---

## 8. Hyperparameters

Defaults are chosen per map / dynamics by `src/config.py::make_config`.
Stochastic and larger maps get a smaller learning rate and a higher discount.

| Setting | 4×4 deterministic | 4×4 slippery | 8×8 deterministic | 8×8 slippery |
|---|---|---|---|---|
| Episodes | 10,000 | 20,000 | 50,000 | 50,000 |
| Learning rate α | 0.80 | 0.10 | 0.80 | 0.10 |
| Discount γ | 0.95 | 0.99 | 0.99 | 0.99 |
| ε start → min | 1.0 → 0.01 | 1.0 → 0.05 | 1.0 → 0.01 | 1.0 → 0.05 |
| ε decay reaches min by | 60% of episodes | 60% | 60% | 60% |
| Eval episodes | 1,000 | 1,000 | 1,000 | 1,000 |
| Convergence threshold | 90% | 40% | 90% | 40% |
| Seed | 42 | 42 | 42 | 42 |

The **convergence threshold** is the rolling (100-episode) success rate used to
define "converged"; it is lower for slippery maps because their *optimal*
success rate is well below 100%.

---

## 9. Example results

The numbers below come from the bundled reference run (`python main.py all`,
seed 42). They are fully reproducible and are saved to
`results/summary_all.csv` and `results/overestimation_diagnostics.csv`.

### 9.1 Evaluation over 1,000 greedy episodes

| Experiment | Algorithm | Success rate | Avg reward | Avg length (all) | Avg length (successful) |
|---|---|:--:|:--:|:--:|:--:|
| 4×4 deterministic | Q-Learning        | **100.0%** | 1.000 | 6.0  | 6.0  |
| 4×4 deterministic | Double Q-Learning | **100.0%** | 1.000 | 6.0  | 6.0  |
| 4×4 slippery      | Q-Learning        | 73.1% | 0.731 | 44.1 | 37.9 |
| 4×4 slippery      | Double Q-Learning | **73.3%** | 0.733 | 41.1 | 36.9 |
| 8×8 deterministic | Q-Learning        | **100.0%** | 1.000 | 14.0 | 14.0 |
| 8×8 deterministic | Double Q-Learning | **100.0%** | 1.000 | 14.0 | 14.0 |
| 8×8 slippery      | Q-Learning        | **51.6%** | 0.516 | 76.9 | 62.7 |
| 8×8 slippery      | Double Q-Learning | 44.9% | 0.449 | 81.6 | 67.3 |

Both algorithms **solve** the task: a perfect 100% on every deterministic map,
and a high success rate on the slippery maps — close to the *optimal achievable*
rate, since even a perfect policy slips into holes sometimes. (For the standard
slippery 4×4 map the optimal policy reaches the goal ≈74% of the time, which
both algorithms essentially match.)

### 9.2 Convergence speed and training cost

| Experiment | Algorithm | Episodes to converge | Training time |
|---|---|:--:|:--:|
| 4×4 deterministic | Q-Learning        | 2,878  | 1.8 s |
| 4×4 deterministic | Double Q-Learning | **2,658**  | 2.7 s |
| 4×4 slippery      | Q-Learning        | **7,485**  | 13.8 s |
| 4×4 slippery      | Double Q-Learning | 7,534  | 15.1 s |
| 8×8 deterministic | Q-Learning        | **14,580** | 22.7 s |
| 8×8 deterministic | Double Q-Learning | 15,926 | 32.1 s |
| 8×8 slippery      | Q-Learning        | **17,791** | 76.7 s |
| 8×8 slippery      | Double Q-Learning | 22,348 | 97.9 s |

*"Converge" = first episode at which the 100-episode rolling success rate
crosses the threshold (90% deterministic, 40% slippery).* Q-Learning generally
converges in **fewer episodes** and trains **faster**, because Double Q-Learning
updates each of its two tables only ~half the time and so needs more data per
table.

### 9.3 Overestimation diagnostic (the key result)

Mean state value $\overline{V} = \text{mean}_s \max_a Q(s,a)$ from the learned
tables. Lower (less inflated) is better when the comparison is against an
unbiased reference:

| Experiment | Q-Learning $\overline{V}$ | Double Q $\overline{V}$ | Reduction | Peak Q vs Double-Q |
|---|:--:|:--:|:--:|:--:|
| 4×4 deterministic | 0.602 | 0.602 | 0.0%  | 1.000 vs 1.000 |
| 4×4 slippery      | 0.382 | 0.355 | **7.1%**  | 0.908 vs 0.900 |
| 8×8 deterministic | 0.708 | 0.669 | 5.5%  | 1.000 vs 1.000 |
| 8×8 slippery      | 0.311 | 0.271 | **12.8%** | 0.891 vs 0.875 |

This is the clearest, most reproducible finding: **Double Q-Learning's value
estimates are systematically lower than Q-Learning's on the stochastic maps**
(7–13% lower mean value), exactly the overestimation reduction the algorithm is
designed to produce. On the fully deterministic 4×4 map there is no bias to
remove and the two agents converge to *identical* values; the small 8×8
deterministic gap reflects Double-Q still climbing toward the true values (its
split updates converge more slowly), not bias.

### 9.4 Honest caveat: bias reduction ≠ guaranteed higher reward

Lower value estimates did **not** always translate into higher success within a
fixed episode budget. On the large **8×8 slippery** map, plain Q-Learning
actually finished with a higher success rate (51.6% vs 44.9%): its optimistic
targets propagate the sparse reward faster, and because Double Q-Learning splits
its experience across two tables it effectively under-trains on the hardest map.
On the **4×4 slippery** map the two are a statistical tie (73.1% vs 73.3%). This
is the expected, nuanced picture — Double Q-Learning's advantage is *unbiased
values and stability*, which matters more as noise grows and given enough data,
not a free boost to reward on every problem. See [§11](#11-analysis--discussion).

---

## 10. Generated plots & demo videos

For every experiment, `python main.py run/all` writes six figures to
`plots/<experiment>/`. Each has a title, axis labels, a legend and grid lines.

| File | Plot |
|---|---|
| `reward_vs_episodes.png` | **Reward vs Training Episodes** — raw per-episode reward (faint) plus the 100-episode moving average for both algorithms. |
| `success_rate_vs_episodes.png` | **Success Rate vs Episodes** — rolling success rate with the convergence threshold marked. |
| `moving_average_reward.png` | **Moving Average Reward** — smoothed reward curves, head-to-head. |
| `epsilon_decay.png` | **Epsilon Decay Curve** — the ε schedule with ε_min marked. |
| `comparison.png` | **Q-Learning vs Double Q-Learning** — a 4-panel dashboard (reward, success rate, episode length, and a **value-estimate** overestimation diagnostic). |
| `convergence_comparison.png` | **Convergence Comparison** — smoothed success curves with convergence episodes marked, plus a bar chart of episodes-to-converge and training time. |

The `comparison.png` "Value Estimates" panel is the clearest single picture of
overestimation: Q-Learning's mean state value $\text{mean}_s\max_a Q(s,a)$ sits
**above** Double Q-Learning's on the slippery maps (see §9.3).

### 10.1 Figures

#### 4×4 slippery — Q-Learning vs Double Q-Learning
![comparison 4x4 slippery](plots/4x4_slippery/comparison.png)
![convergence 4x4 slippery](plots/4x4_slippery/convergence_comparison.png)

<details>
<summary>4×4 slippery — reward, success rate, moving average, epsilon decay</summary>

![reward](plots/4x4_slippery/reward_vs_episodes.png)
![success rate](plots/4x4_slippery/success_rate_vs_episodes.png)
![moving average reward](plots/4x4_slippery/moving_average_reward.png)
![epsilon decay](plots/4x4_slippery/epsilon_decay.png)
</details>

#### 8×8 slippery — Q-Learning vs Double Q-Learning
![comparison 8x8 slippery](plots/8x8_slippery/comparison.png)
![convergence 8x8 slippery](plots/8x8_slippery/convergence_comparison.png)

<details>
<summary>8×8 slippery — reward, success rate, moving average, epsilon decay</summary>

![reward](plots/8x8_slippery/reward_vs_episodes.png)
![success rate](plots/8x8_slippery/success_rate_vs_episodes.png)
![moving average reward](plots/8x8_slippery/moving_average_reward.png)
![epsilon decay](plots/8x8_slippery/epsilon_decay.png)
</details>

<details>
<summary>4×4 deterministic — all six figures</summary>

![comparison](plots/4x4_deterministic/comparison.png)
![convergence](plots/4x4_deterministic/convergence_comparison.png)
![reward](plots/4x4_deterministic/reward_vs_episodes.png)
![success rate](plots/4x4_deterministic/success_rate_vs_episodes.png)
![moving average reward](plots/4x4_deterministic/moving_average_reward.png)
![epsilon decay](plots/4x4_deterministic/epsilon_decay.png)
</details>

<details>
<summary>8×8 deterministic — all six figures</summary>

![comparison](plots/8x8_deterministic/comparison.png)
![convergence](plots/8x8_deterministic/convergence_comparison.png)
![reward](plots/8x8_deterministic/reward_vs_episodes.png)
![success rate](plots/8x8_deterministic/success_rate_vs_episodes.png)
![moving average reward](plots/8x8_deterministic/moving_average_reward.png)
![epsilon decay](plots/8x8_deterministic/epsilon_decay.png)
</details>

### 10.2 Demo videos (original FrozenLake graphics)

Short clips of the trained greedy policy, rendered with Gymnasium's original
FrozenLake sprites (the elf, ice, cracked-ice holes and the gift-box goal).
The animations below are GIF previews; the full-length, higher-quality MP4s are
in [`videos/`](videos/).

| 4×4 deterministic | 4×4 slippery |
|:--:|:--:|
| ![4x4 deterministic demo](videos/preview_4x4_deterministic.gif) | ![4x4 slippery demo](videos/preview_4x4_slippery.gif) |
| **8×8 deterministic** | **8×8 slippery** |
| ![8x8 deterministic demo](videos/preview_8x8_deterministic.gif) | ![8x8 slippery demo](videos/preview_8x8_slippery.gif) |

Full MP4 recordings (3 episodes each, both algorithms):

| Experiment | Q-Learning | Double Q-Learning |
|---|---|---|
| 4×4 deterministic | [mp4](videos/4x4_deterministic_q_learning.mp4) | [mp4](videos/4x4_deterministic_double_q_learning.mp4) |
| 4×4 slippery | [mp4](videos/4x4_slippery_q_learning.mp4) | [mp4](videos/4x4_slippery_double_q_learning.mp4) |
| 8×8 deterministic | [mp4](videos/8x8_deterministic_q_learning.mp4) | [mp4](videos/8x8_deterministic_double_q_learning.mp4) |
| 8×8 slippery | [mp4](videos/8x8_slippery_q_learning.mp4) | [mp4](videos/8x8_slippery_double_q_learning.mp4) |

> GitHub renders the GIFs inline; relative-path MP4s open in the browser's video
> player when clicked. Regenerate everything with `python main.py video
> --experiment <name>`.

---

## 11. Analysis & discussion

### How Q-Learning works
Q-Learning bootstraps the optimal value of each state-action pair from its own
current estimate of the best next action. It is off-policy (it learns the greedy
policy while exploring with ε-greedy) and, in the tabular case with sufficient
exploration and a suitable step-size schedule, is guaranteed to converge to
$Q^*$. On FrozenLake the sparse +1 reward must propagate backwards from the
goal one step at a time, which is why a high enough discount γ and many episodes
are needed — more so on the 8×8 map where optimal paths are long.

### How Double Q-Learning reduces overestimation bias
The single $\max$ in the Q-Learning target couples "which action looks best"
with "how good is it," so estimation noise is rectified into a persistent upward
bias (Jensen's inequality). Double Q-Learning breaks that coupling: one table
picks the action, an independent table scores it. The scoring table has no
reason to be biased high for the action the *other* table happened to
over-rate, so the target becomes unbiased/slightly conservative. The visible
consequences are **lower peak Q-values** and **smoother value estimates**.

### Learning-behaviour differences (as observed here)
- **Deterministic maps.** With no transition noise there is little
  overestimation to correct, so both algorithms learn the optimal policy and
  reach 100% success. Q-Learning converges in *fewer* episodes and *faster*,
  because it updates a single table every step whereas Double Q-Learning updates
  each table only ~half the time (≈twice the data needed per table). On 4×4 both
  converge to identical values; on 8×8 Double-Q's values are still slightly
  below the converged Q-Learning values at 50k episodes — under-convergence, not
  bias.
- **Stochastic (slippery) maps.** Here the bias is real and measurable: Double
  Q-Learning's mean value estimate is **7–13% lower** than Q-Learning's
  (§9.3) — the overestimation reduction it is designed for. However, with a
  *constant* learning rate and a *single seed*, both success-rate curves are
  noisy and oscillate around their plateaus; the reduced bias did **not** yield
  a higher success rate within the episode budget (4×4: a tie; 8×8: Q-Learning
  ahead, because splitting experience across two tables under-trains Double-Q on
  the largest map). The robust, reproducible effect of Double Q-Learning in this
  study is therefore **lower, more realistic value estimates**, not a guaranteed
  reward gain — consistent with the theory, which promises unbiasedness rather
  than universally better control performance.

### Convergence characteristics
Convergence speed is reported as the first episode at which the 100-episode
rolling success rate crosses the threshold. Deterministic settings converge
quickly and cleanly; slippery settings converge slowly to a *plateau* below
100% (the optimal achievable success rate under slip), and the constant
learning rate leaves residual variance around that plateau.

### Impact of stochastic transitions
Slip turns a short, certain path into a distribution over outcomes: the optimal
policy must steer *away* from holes even at the cost of detours, and even the
optimal policy fails sometimes. This (a) lowers the achievable success rate,
(b) lengthens episodes, (c) slows convergence, and (d) amplifies the
overestimation bias that Double Q-Learning is designed to fight.

### Strengths and weaknesses

| | **Q-Learning** | **Double Q-Learning** |
|---|---|---|
| **Strengths** | Simple; one table; fast, data-efficient convergence; optimal in deterministic / low-noise settings. | Unbiased targets; lower, more realistic Q-values; more stable learning under stochasticity. |
| **Weaknesses** | Overestimates action values under noise; learning curves can oscillate in stochastic environments. | Two tables (≈2× memory); each table sees half the updates, so it can need more episodes to converge. |
| **Best suited to** | Deterministic or near-deterministic problems where bias is negligible. | Noisy / stochastic environments where overestimation hurts. |

---

## 12. Reproducibility

- **Seeds.** A fixed seed (default 42) seeds the agent RNG (ε-greedy action
  choice and the Double-Q coin flip), the environment transition RNG, and the
  action space. Evaluation uses a separate, fixed seed offset.
- **Saved artifacts.** Q-tables are saved as `.npz` (`q_learning_qtable.npz`
  stores `Q`; `double_q_learning_qtable.npz` stores `Q1`, `Q2`). Per-episode
  statistics (`reward, length, epsilon, success`) are saved as CSV. Each
  experiment's `config.json` records every hyperparameter.
- **Plots & CSVs are written automatically** by `run`/`all`.
- **Determinism caveat.** Greedy ties are broken with the seeded RNG, so results
  are reproducible run-to-run for a given seed and library version.

---

## 13. Future work

- **Decaying / state-visit learning rates** (e.g. $\alpha_t \propto 1/n(s,a)$)
  to drive the slippery-map success rate to its true optimum and shrink residual
  variance.
- **Expected SARSA / Double Expected SARSA** baselines for a fuller bias study.
- **Function approximation** (Deep Q-Network and Double DQN) to scale beyond
  tabular states and connect to the modern literature.
- **Hyperparameter sweeps** (α, γ, ε schedules) with confidence intervals over
  multiple seeds, instead of a single seed.
- **Random map generation** (`generate_random_map`) to test generalization
  across layouts rather than the two fixed maps.

---

## 14. References

- C. J. C. H. Watkins, *Learning from Delayed Rewards*, PhD thesis, 1989.
- H. van Hasselt, "Double Q-learning," *NeurIPS*, 2010.
- H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double
  Q-learning," *AAAI*, 2016.
- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*,
  2nd ed., MIT Press, 2018.
- Gymnasium FrozenLake documentation:
  <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>
