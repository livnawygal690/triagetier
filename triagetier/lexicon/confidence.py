from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from triagetier.ir.records import ConfidenceReport


def term_probability(logprobs: Sequence[float]) -> float:
    if not logprobs:
        return 1.0
    return float(np.exp(np.mean(np.asarray(logprobs, dtype=np.float64))))


def window_agreement(presence_per_window: Sequence[bool]) -> float:
    if not presence_per_window:
        return 1.0
    votes = np.asarray(presence_per_window, dtype=np.float64)
    majority = votes.mean() >= 0.5
    return float((votes == majority).mean())


def confidence_score(logprobs: Sequence[float], windows: Sequence[bool]) -> float:
    blended = 0.5 * term_probability(logprobs) + 0.5 * window_agreement(windows)
    return float(np.clip(blended, 0.0, 1.0))


def aggregate(per_agent: Mapping[str, float], threshold: float) -> ConfidenceReport:
    score = min(per_agent.values()) if per_agent else 1.0
    return ConfidenceReport(
        score=float(score),
        flagged_for_review=bool(score < threshold),
        per_agent=dict(per_agent),
    )
