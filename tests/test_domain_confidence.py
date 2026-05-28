from __future__ import annotations

import math

from triagetier.lexicon.confidence import (
    aggregate,
    confidence_score,
    term_probability,
    window_agreement,
)


def test_term_probability_bounds() -> None:
    assert term_probability([]) == 1.0
    assert term_probability([0.0]) == 1.0
    assert math.isclose(term_probability([-1.0]), math.exp(-1.0), rel_tol=1e-6)


def test_window_agreement() -> None:
    assert window_agreement([True, True, True]) == 1.0
    assert window_agreement([]) == 1.0
    assert 0.0 <= window_agreement([True, False, True]) <= 1.0


def test_confidence_score_in_unit_interval() -> None:
    for logprobs in ([-5.0], [-0.01], []):
        value = confidence_score(logprobs, [True, True])
        assert 0.0 <= value <= 1.0


def test_aggregate_flags_below_threshold() -> None:
    flagged = aggregate({"detection": 0.2, "qa": 0.9}, 0.3)
    assert flagged.flagged_for_review is True
    assert flagged.score == 0.2

    clear = aggregate({"detection": 0.9, "qa": 0.95}, 0.3)
    assert clear.flagged_for_review is False
