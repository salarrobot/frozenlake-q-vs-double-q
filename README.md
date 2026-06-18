# FrozenLake: Q-Learning vs Double Q-Learning

A complete, from-scratch reinforcement-learning study of **tabular Q-Learning**
and **Double Q-Learning** on the [Gymnasium FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
environment. Both algorithms are implemented without any RL library
(no Stable-Baselines3, RLlib, etc.) — only `numpy` for the tables, `gymnasium`
for the environment, and `pandas`/`matplotlib` for analysis and plots.

The project trains agents on the **4×4** and **8×8** maps in both
**deterministic** (`is_slippery=False`) and **stochastic** (`is_slippery=True`)
regimes, evaluates the learned greedy policies over 1,000 test episodes, and
produces publication-quality figures and demo videos comparing the two
algorithms. The central question is **overestimation bias**: ordinary
Q-Learning uses the same values to choose and to evaluate the next action, while
Double Q-Learning decouples the two — which measurably lowers its value
estimates on the stochastic maps.

---

## Plots

For every experiment, six figures are written to `plots/<experiment>/`. Each has
a title, axis labels, a legend and grid lines.

### 4×4 slippery — Q-Learning vs Double Q-Learning
![comparison 4x4 slippery](plots/4x4_slippery/comparison.png)
![convergence 4x4 slippery](plots/4x4_slippery/convergence_comparison.png)

<details>
<summary>4×4 slippery — reward, success rate, moving average, epsilon decay</summary>

![reward](plots/4x4_slippery/reward_vs_episodes.png)
![success rate](plots/4x4_slippery/success_rate_vs_episodes.png)
![moving average reward](plots/4x4_slippery/moving_average_reward.png)
![epsilon decay](plots/4x4_slippery/epsilon_decay.png)
</details>

### 8×8 slippery — Q-Learning vs Double Q-Learning
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

---

## Demo videos

Short clips of the trained greedy policy, rendered with Gymnasium's original
FrozenLake sprites (the elf, ice, cracked-ice holes and the gift-box goal).
The animations below are GIF previews; the full-length MP4s are in
[`videos/`](videos/).

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
