"""Shared utilities: reproducibility, smoothing, metrics and I/O helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int) -> None:
    """Seed Python's ``random`` and the legacy NumPy global RNG.

    Per-run randomness (action selection, table choice in Double Q-Learning)
    uses a dedicated ``numpy.random.Generator`` seeded separately so that the
    behaviour of each experiment is fully deterministic and independent.
    """
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Smoothing / metrics
# --------------------------------------------------------------------------- #
def rolling_mean(values, window: int) -> np.ndarray:
    """Trailing rolling mean with the same length as ``values``.

    ``min_periods=1`` means the first few entries average over whatever data is
    available, so the returned array lines up one-to-one with the episode axis.
    """
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


def episodes_to_threshold(successes, window: int, threshold: float) -> int | None:
    """First episode at which the rolling success rate reaches ``threshold``.

    Returns ``None`` when the threshold is never reached (no convergence).
    This is the project's "convergence speed" metric.
    """
    smoothed = rolling_mean(successes, window)
    # Only count once the rolling window is actually full, otherwise a single
    # early lucky success could trivially cross a low threshold.
    idx = np.flatnonzero(smoothed >= threshold)
    idx = idx[idx >= window - 1]
    return int(idx[0]) if idx.size else None


def format_time(seconds: float) -> str:
    """Human readable duration, e.g. ``2m 03.4s``."""
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{int(minutes)}m {secs:04.1f}s"
    return f"{secs:.2f}s"


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)
