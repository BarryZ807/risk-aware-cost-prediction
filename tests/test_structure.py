import numpy as np

from evaluation.structure import quantile_crossings, risk_order_violations


def test_ordered_outputs_pass_both_audits():
    rng = np.random.default_rng(0)
    # Rows sorted in increasing order stand in for well-ordered quantile predictions.
    q = np.sort(rng.normal(size=(512, 7)), axis=1)
    assert quantile_crossings(q) == {"pairs": 512 * 6, "crossings": 0, "rate": 0.0}
    # Nested risks listed from the widest group to the narrowest are non-increasing in every row.
    p = np.sort(rng.uniform(size=(512, 3)), axis=1)[:, ::-1]
    assert risk_order_violations(p)["violations"] == 0


def test_audits_detect_a_planted_violation():
    q = np.array([[1.0, 2.0, 3.0], [1.0, 0.5, 3.0]])
    assert quantile_crossings(q)["crossings"] == 1
    p = np.array([[0.5, 0.3, 0.1], [0.2, 0.3, 0.1]])
    assert risk_order_violations(p)["violations"] == 1


def test_rates_are_shares_of_adjacent_pairs():
    q = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    out = quantile_crossings(q)
    assert out["pairs"] == 4 and out["crossings"] == 2 and out["rate"] == 0.5
