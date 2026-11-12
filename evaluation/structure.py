"""Audits of structural coherence: do the model's outputs ever contradict each other?"""

from __future__ import annotations

from typing import Dict

import numpy as np


def quantile_crossings(pred_q: np.ndarray) -> Dict[str, float]:
    """Count adjacent quantile pairs that are out of order. pred_q has shape [N, Q], increasing in tau."""
    pred_q = np.asarray(pred_q, dtype=float)
    pairs = pred_q[:, :-1] > pred_q[:, 1:]
    return {"pairs": int(pairs.size), "crossings": int(pairs.sum()), "rate": float(pairs.mean()) if pairs.size else 0.0}


def risk_order_violations(risk_prob: np.ndarray) -> Dict[str, float]:
    """Count nested-risk pairs that are out of order.

    risk_prob has shape [N, K], with columns from the widest group to the narrowest
    (for example top 10%, top 5%, top 1%), so each row should be non-increasing.
    """
    risk_prob = np.asarray(risk_prob, dtype=float)
    pairs = risk_prob[:, :-1] < risk_prob[:, 1:]
    return {"pairs": int(pairs.size), "violations": int(pairs.sum()), "rate": float(pairs.mean()) if pairs.size else 0.0}
