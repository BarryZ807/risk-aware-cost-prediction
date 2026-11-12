import numpy as np

from evaluation.calibration import decision_curve_net_benefit, weighted_ece
from evaluation.capture import metric_capture_at_k, weighted_capture_at_k
from evaluation.paired_bootstrap import ComparisonInputs, paired_cluster_bootstrap


def test_capture_on_a_worked_example():
    y = np.array([100.0, 50.0, 30.0, 20.0])
    w = np.ones(4)
    score = np.array([0.9, 0.1, 0.8, 0.2])
    # top 50% of 4 rows = 2 rows: the ones with cost 100 and 30
    assert np.isclose(weighted_capture_at_k(y, score, w, 50), 130 / 200)


def test_capture_uses_survey_weights_for_spending():
    y = np.array([100.0, 100.0])
    w = np.array([3.0, 1.0])
    score = np.array([0.9, 0.1])
    assert np.isclose(weighted_capture_at_k(y, score, w, 50), 0.75)


def _toy(seed=0, n_persons=300):
    rng = np.random.default_rng(seed)
    person = np.repeat(np.arange(n_persons), 2)  # two person-years each
    y = rng.lognormal(mean=7.0, sigma=1.5, size=person.size)
    w = rng.uniform(0.5, 2.0, size=person.size)
    return ComparisonInputs(
        label="oracle vs noise",
        person_id=person,
        y_cost=y,
        w=w,
        score_model=y.copy(),  # ranks cost perfectly
        score_baseline=rng.normal(size=person.size),  # ranks at random
    )


def test_bootstrap_separates_a_good_model_from_noise():
    rows = paired_cluster_bootstrap(_toy(), {"Capture@5%": metric_capture_at_k(5)}, n_boot=300, seed=1)
    row = rows[0]
    assert row["mean_delta"] > 0 and row["ci_lo"] > 0
    assert row["p_superior"] > 0.99 and row["p_value"] < 0.05


def test_bootstrap_is_reproducible_and_null_when_scores_match():
    a = paired_cluster_bootstrap(_toy(), {"Capture@5%": metric_capture_at_k(5)}, n_boot=100, seed=7)
    b = paired_cluster_bootstrap(_toy(), {"Capture@5%": metric_capture_at_k(5)}, n_boot=100, seed=7)
    assert a == b
    same = _toy()
    same.score_baseline = same.score_model.copy()
    row = paired_cluster_bootstrap(same, {"Capture@5%": metric_capture_at_k(5)}, n_boot=100, seed=7)[0]
    assert row["mean_delta"] == 0.0 and row["p_value"] == 1.0


def test_ece_is_zero_for_perfect_calibration_and_dca_matches_treat_all_at_zero_threshold():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    w = np.ones(4)
    assert weighted_ece(y, np.array([0.0, 0.0, 1.0, 1.0]), w) == 0.0
    out = decision_curve_net_benefit(y, np.array([0.1, 0.2, 0.8, 0.9]), w, thresholds=np.array([0.05]))
    assert np.isclose(out[0, 1], out[0, 2])  # below every score, the model treats everyone
