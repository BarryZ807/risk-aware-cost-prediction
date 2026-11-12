"""
Paired, person-clustered bootstrap for comparing a model against a baseline on the same test set.

The unit of resampling is the person, not the row: one person can contribute several
person-years, and those rows are not independent. Each resample draws persons with
replacement and applies the multiplicity to the survey weight, so the estimate stays a
weighted-population estimate. Model and baseline are scored on the same resample, which is
what makes the comparison paired. One resample is shared across all metrics in an iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 123

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


@dataclass
class ComparisonInputs:
    """Aligned per-row arrays for one model-against-baseline comparison."""

    label: str                   # e.g. "model vs calibrated logistic"
    person_id: np.ndarray        # shape [N], the cluster key
    y_cost: np.ndarray           # shape [N], realised cost in the following year
    w: np.ndarray                # shape [N], person-year survey weight
    score_model: np.ndarray      # shape [N], model score (seed-averaged if several seeds)
    score_baseline: np.ndarray   # shape [N], baseline score
    n_seeds_model: int = 1

    def __post_init__(self) -> None:
        n = len(self.person_id)
        for name in ("y_cost", "w", "score_model", "score_baseline"):
            assert len(getattr(self, name)) == n, f"length mismatch on {name}"


def paired_cluster_bootstrap(
    inputs: ComparisonInputs,
    metric_fns: Dict[str, Metric],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, object]]:
    """Return one row per metric: point estimates, mean difference, standard error,
    95% percentile interval, a two-sided p-value and the share of resamples where the model wins."""
    rng = np.random.default_rng(seed)
    persons = np.unique(inputs.person_id)
    n_persons = len(persons)
    person_to_rows = {pid: np.where(inputs.person_id == pid)[0] for pid in persons}

    model_point = {m: fn(inputs.y_cost, inputs.score_model, inputs.w) for m, fn in metric_fns.items()}
    base_point = {m: fn(inputs.y_cost, inputs.score_baseline, inputs.w) for m, fn in metric_fns.items()}

    deltas: Dict[str, List[float]] = {m: [] for m in metric_fns}
    for _ in range(n_boot):
        sampled = rng.choice(persons, size=n_persons, replace=True)
        unique_pids, counts = np.unique(sampled, return_counts=True)
        idx_chunks: List[np.ndarray] = []
        mult_chunks: List[np.ndarray] = []
        for pid, cnt in zip(unique_pids, counts):
            rows = person_to_rows[pid]
            idx_chunks.append(rows)
            mult_chunks.append(np.full(len(rows), cnt, dtype=np.float64))
        idx = np.concatenate(idx_chunks)
        mult = np.concatenate(mult_chunks)

        y_b = inputs.y_cost[idx]
        w_b = inputs.w[idx] * mult
        sm_b = inputs.score_model[idx]
        sb_b = inputs.score_baseline[idx]

        for m, fn in metric_fns.items():
            deltas[m].append(fn(y_b, sm_b, w_b) - fn(y_b, sb_b, w_b))

    rows: List[Dict[str, object]] = []
    for m in metric_fns:
        arr = np.asarray(deltas[m], dtype=np.float64)
        rows.append(
            {
                "comparison": inputs.label,
                "metric": m,
                "model": model_point[m],
                "baseline": base_point[m],
                "mean_delta": float(arr.mean()),
                "se": float(arr.std(ddof=1)),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5)),
                # two-sided: twice the smaller tail beyond zero, capped at 1
                "p_value": float(min(1.0, 2.0 * min((arr >= 0).mean(), (arr <= 0).mean()))),
                "p_superior": float((arr > 0).mean()),
                "n_test": int(len(inputs.person_id)),
                "n_seeds_model": int(inputs.n_seeds_model),
                "n_bootstrap": int(n_boot),
            }
        )
    return rows
