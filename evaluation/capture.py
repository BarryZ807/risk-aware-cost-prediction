"""
CostCapture@k: of all spending in the population, the share that belongs to the people a model flags.

A care team can reach only a fixed number of people. This metric asks how much of the total
cost sits in the group a model would send them to.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def weighted_capture_at_k(y_cost: np.ndarray, score: np.ndarray, w: np.ndarray, k_pct: float) -> float:
    """Select the top ceil(N * k%) rows by score, then return
    sum(w * y) over the selected rows divided by sum(w * y) over all rows.

    Selection is by row count, because capacity is a head count. Spending is survey-weighted.
    Ties are broken by original order, so the result is deterministic.
    """
    n = len(score)
    n_sel = max(1, int(np.ceil(n * k_pct / 100.0)))
    order = np.argsort(-score, kind="stable")[:n_sel]
    denom = float((w * y_cost).sum())
    if denom <= 0:
        return float("nan")
    return float((w[order] * y_cost[order]).sum() / denom)


def metric_capture_at_k(k_pct: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """Return capture@k as a function of (y_cost, score, w), for use in the bootstrap."""
    label_pct = int(k_pct) if float(k_pct).is_integer() else k_pct

    def _fn(y, s, w):
        return weighted_capture_at_k(y, s, w, k_pct)

    _fn.__name__ = f"Capture@{label_pct}%"
    return _fn
