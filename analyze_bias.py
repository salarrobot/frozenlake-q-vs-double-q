"""Quantify the overestimation diagnostic from saved Q-tables.

For each experiment, report mean state-value V(s)=max_a Q(s,a) and peak Q for
both algorithms.  Q-Learning's values are expected to sit above Double
Q-Learning's, most visibly on the stochastic (slippery) maps.
"""

from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = ["4x4_deterministic", "4x4_slippery",
               "8x8_deterministic", "8x8_slippery"]


def q_table(path: Path) -> np.ndarray:
    with np.load(path) as d:
        if "Q" in d:
            return d["Q"]
        return 0.5 * (d["Q1"] + d["Q2"])


rows = []
for exp in EXPERIMENTS:
    base = Path("results") / exp
    q = q_table(base / "q_learning_qtable.npz")
    dq = q_table(base / "double_q_learning_qtable.npz")
    rows.append({
        "experiment": exp,
        "Q mean V(s)": q.max(axis=1).mean(),
        "DoubleQ mean V(s)": dq.max(axis=1).mean(),
        "Q peak": q.max(),
        "DoubleQ peak": dq.max(),
        "mean V(s) reduction %": 100 * (q.max(axis=1).mean() - dq.max(axis=1).mean())
        / max(q.max(axis=1).mean(), 1e-9),
    })

df = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
print(df.to_string(index=False))
df.to_csv("results/overestimation_diagnostics.csv", index=False)
print("\nsaved results/overestimation_diagnostics.csv")
