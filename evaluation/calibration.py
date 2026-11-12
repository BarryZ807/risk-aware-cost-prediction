"""Survey-weighted calibration error and decision-curve net benefit for the risk heads."""

from __future__ import annotations

from typing import Optional

import numpy as np


def weighted_ece(y_true: np.ndarray, prob: np.ndarray, w: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins, weighted by w."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    w = np.asarray(w, dtype=float)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total_w = float(np.sum(w))
    if total_w <= 0:
        return float("nan")

    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i < int(n_bins) - 1:
            mask = (prob >= lo) & (prob < hi)
        else:
            mask = (prob >= lo) & (prob <= hi)
        w_bin = float(np.sum(w[mask]))
        if w_bin <= 0:
            continue
        acc = float(np.sum(w[mask] * y_true[mask]) / w_bin)
        conf = float(np.sum(w[mask] * prob[mask]) / w_bin)
        ece += (w_bin / total_w) * abs(acc - conf)
    return float(ece)


def decision_curve_net_benefit(
    y_true: np.ndarray, prob: np.ndarray, w: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return an array with columns: threshold, net benefit of the model, treat-all, treat-none."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    w = np.asarray(w, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)
    thresholds = np.asarray(thresholds, dtype=float)

    n = float(np.sum(w))
    if n <= 0:
        out = np.zeros((len(thresholds), 4), dtype=float)
        out[:, 0] = thresholds
        out[:, 1:] = np.nan
        return out

    treat_none = np.zeros_like(thresholds)
    prev = float(np.sum(w * y_true) / n)
    treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))

    nb = []
    for pt in thresholds:
        pred = prob >= pt
        tp = float(np.sum(w[pred] * y_true[pred]))
        fp = float(np.sum(w[pred] * (1 - y_true[pred])))
        nb.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    return np.stack([thresholds, np.asarray(nb, dtype=float), treat_all, treat_none], axis=1)
