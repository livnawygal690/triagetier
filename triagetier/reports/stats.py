from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def _as_float(values: Sequence[float]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    y = _as_float(labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, _as_float(scores)))


def bootstrap_ci(
    labels: Sequence[float],
    scores: Sequence[float],
    metric: Callable[..., float] = auc,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    y = _as_float(labels)
    s = _as_float(scores)
    rng = np.random.default_rng(seed)
    point = metric(y, s)
    estimates: list[float] = []
    n = len(y)
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        value = metric(y[idx], s[idx])
        if not np.isnan(value):
            estimates.append(value)
    if not estimates:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(estimates, 100 * alpha / 2))
    hi = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return point, lo, hi


def cohen_kappa(rater_a: Sequence[float], rater_b: Sequence[float]) -> float:
    return float(cohen_kappa_score(_as_float(rater_a), _as_float(rater_b)))


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    result = stats.spearmanr(_as_float(x), _as_float(y))
    return float(result.statistic), float(result.pvalue)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10
) -> float:
    conf = _as_float(confidences)
    hit = _as_float(correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    error = 0.0
    for i in range(n_bins):
        lower = edges[i] if i > 0 else -np.inf
        mask = (conf > lower) & (conf <= edges[i + 1])
        if mask.any():
            error += mask.sum() / total * abs(hit[mask].mean() - conf[mask].mean())
    return float(error)


def _midrank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _fast_delong(
    sorted_preds: NDArray[np.float64], positive_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    m = positive_count
    n = sorted_preds.shape[1] - m
    positive = sorted_preds[:, :m]
    negative = sorted_preds[:, m:]
    k = sorted_preds.shape[0]
    tx = np.vstack([_midrank(positive[r]) for r in range(k)])
    ty = np.vstack([_midrank(negative[r]) for r in range(k)])
    tz = np.vstack([_midrank(sorted_preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    labels: Sequence[float], scores_a: Sequence[float], scores_b: Sequence[float]
) -> tuple[float, float, float, float]:
    y = _as_float(labels)
    order = np.argsort(-y)
    positive_count = int((y == 1).sum())
    preds = np.vstack((_as_float(scores_a), _as_float(scores_b)))[:, order]
    aucs, cov = _fast_delong(preds, positive_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ cov @ contrast.T)[0, 0])
    if variance <= 0.0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), z, p


def mcnemar_test(correct_a: Sequence[float], correct_b: Sequence[float]) -> tuple[float, float]:
    a = _as_float(correct_a).astype(bool)
    b = _as_float(correct_b).astype(bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    discordant = only_a + only_b
    if discordant == 0:
        return 0.0, 1.0
    statistic = (abs(only_a - only_b) - 1.0) ** 2 / discordant
    p = float(stats.chi2.sf(statistic, df=1))
    return float(statistic), p


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[tuple[bool, float]]:
    p = _as_float(pvalues)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return [(bool(adjusted[i] <= alpha), float(adjusted[i])) for i in range(m)]


def benjamini_hochberg(
    pvalues: Sequence[float], alpha: float = 0.05
) -> list[tuple[bool, float]]:
    p = _as_float(pvalues)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * m / (rank + 1))
        adjusted[idx] = min(running, 1.0)
    return [(bool(adjusted[i] <= alpha), float(adjusted[i])) for i in range(m)]
