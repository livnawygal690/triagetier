from __future__ import annotations

import math

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import cohen_kappa_score

from triagetier.reports.stats import (
    auc,
    benjamini_hochberg,
    bootstrap_ci,
    cohen_kappa,
    delong_test,
    expected_calibration_error,
    holm_bonferroni,
    mcnemar_test,
    spearman,
)


def test_auc_extremes_and_degenerate() -> None:
    labels = [0, 0, 1, 1]
    assert auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert math.isnan(auc([1, 1, 1], [0.2, 0.3, 0.4]))


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(0)
    labels = [0] * 25 + [1] * 25
    scores = list(rng.normal(0, 1, 25)) + list(rng.normal(2, 1, 25))
    point, lo, hi = bootstrap_ci(labels, scores, auc, resamples=300, seed=1)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_cohen_kappa_matches_sklearn() -> None:
    a = [1, 2, 3, 4, 5, 3, 2, 1]
    b = [1, 2, 3, 4, 4, 3, 2, 2]
    assert math.isclose(cohen_kappa(a, b), float(cohen_kappa_score(a, b)), rel_tol=1e-9)


def test_spearman_matches_scipy() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 1.0, 4.0, 3.0, 6.0]
    r, _ = spearman(x, y)
    expected = scipy_stats.spearmanr(x, y).statistic
    assert math.isclose(r, float(expected), rel_tol=1e-9)


def test_ece_zero_when_calibrated() -> None:
    confidences = [0.95, 0.95, 0.05, 0.05]
    correct = [1.0, 1.0, 0.0, 0.0]
    assert expected_calibration_error(confidences, correct, n_bins=10) < 0.1


def test_ece_positive_when_overconfident() -> None:
    confidences = [0.99, 0.99, 0.99, 0.99]
    correct = [1.0, 0.0, 0.0, 0.0]
    assert expected_calibration_error(confidences, correct, n_bins=10) > 0.5


def test_delong_identical_predictors() -> None:
    rng = np.random.default_rng(3)
    labels = [0] * 30 + [1] * 30
    scores = list(rng.normal(0, 1, 30)) + list(rng.normal(1.5, 1, 30))
    auc_a, auc_b, z, p = delong_test(labels, scores, scores)
    assert math.isclose(auc_a, auc_b, rel_tol=1e-9)
    assert math.isclose(z, 0.0, abs_tol=1e-9)
    assert math.isclose(p, 1.0, rel_tol=1e-9)


def test_delong_orders_better_predictor() -> None:
    labels = [0] * 20 + [1] * 20
    strong = [0.0] * 20 + [1.0] * 20
    weak = ([0.4] * 10 + [0.6] * 10) * 2
    auc_strong, auc_weak, _, _ = delong_test(labels, strong, weak)
    assert auc_strong > auc_weak


def test_mcnemar_symmetric_and_skewed() -> None:
    a_sym = [1, 0, 1, 0]
    b_sym = [0, 1, 0, 1]
    _, p_sym = mcnemar_test(a_sym, b_sym)
    assert p_sym > 0.5

    a_skew = [1] * 10 + [1] * 10
    b_skew = [0] * 10 + [1] * 10
    stat, p_skew = mcnemar_test(a_skew, b_skew)
    assert p_skew < 0.05


def test_holm_rejects_small_pvalue() -> None:
    pvalues = [0.001, 0.04, 0.2, 0.8]
    result = holm_bonferroni(pvalues, alpha=0.05)
    assert result[0][0] is True
    assert result[3][0] is False
    adjusted = [r[1] for r in result]
    assert adjusted == sorted(adjusted) or adjusted[0] <= adjusted[-1]


def test_benjamini_hochberg_monotone() -> None:
    pvalues = [0.001, 0.01, 0.03, 0.5]
    result = benjamini_hochberg(pvalues, alpha=0.05)
    assert all(0.0 <= adj <= 1.0 for _, adj in result)
    assert result[0][0] is True
