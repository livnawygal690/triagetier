from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from triagetier.reports.stats import auc, bootstrap_ci


@dataclass(frozen=True)
class OrganAUC:
    organ: str
    auc: float
    ci_low: float
    ci_high: float


def per_organ_auc(
    labels: Mapping[str, Sequence[float]],
    scores: Mapping[str, Sequence[float]],
    seed: int = 0,
) -> list[OrganAUC]:
    results: list[OrganAUC] = []
    for organ in labels:
        point, lo, hi = bootstrap_ci(labels[organ], scores[organ], auc, seed=seed)
        results.append(OrganAUC(organ, point, lo, hi))
    return results


def mean_organ_auc(results: Sequence[OrganAUC]) -> float:
    values = [r.auc for r in results if not np.isnan(r.auc)]
    return float(np.mean(values)) if values else float("nan")


@dataclass(frozen=True)
class ReasoningMetrics:
    detection_precision: float
    grading_accuracy: float
    chain_completeness: float


def agent_reasoning_metrics(
    pred_present: Sequence[float],
    true_present: Sequence[float],
    pred_grade: Sequence[float],
    true_grade: Sequence[float],
    chain_complete: Sequence[float],
) -> ReasoningMetrics:
    pp = np.asarray(pred_present, dtype=bool)
    tp = np.asarray(true_present, dtype=bool)
    predicted_positive = int(np.sum(pp))
    precision = (
        float(np.sum(pp & tp) / predicted_positive) if predicted_positive else float("nan")
    )

    pg = np.asarray(pred_grade, dtype=np.float64)
    tg = np.asarray(true_grade, dtype=np.float64)
    grading = float(np.mean(pg == tg)) if len(pg) else float("nan")

    completeness = float(np.mean(np.asarray(chain_complete, dtype=np.float64)))
    return ReasoningMetrics(precision, grading, completeness)
