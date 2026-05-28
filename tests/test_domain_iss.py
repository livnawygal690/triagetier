from __future__ import annotations

import pytest

from triagetier.ir.records import AISGrade
from triagetier.lexicon.graph import KnowledgeGraph
from triagetier.lexicon.iss import aggregate_iss, baker_score


def _grade(organ: str, ais: int, region: str = "abdomen") -> AISGrade:
    return AISGrade(organ=organ, region=region, aast_grade=ais, ais=ais, justification="")


def test_baker_matches_worked_example() -> None:
    assert baker_score([2, 1]) == 5


def test_baker_uses_only_top_three() -> None:
    assert baker_score([3, 2, 2, 1]) == 9 + 4 + 4


def test_baker_ignores_zero_and_empty() -> None:
    assert baker_score([]) == 0
    assert baker_score([0, 0]) == 0


def test_baker_caps_at_three_terms() -> None:
    assert baker_score([4, 4, 4, 4]) == 48


@pytest.mark.parametrize("extra", [0, 1, 2, 3])
def test_iss_monotonic_in_added_injury(graph: KnowledgeGraph, extra: int) -> None:
    base = baker_score([3, 1])
    grown = baker_score([3, 1, extra])
    assert grown >= base


def test_aggregate_iss_marks_partial(graph: KnowledgeGraph) -> None:
    grades = [_grade("liver", 2), _grade("spleen", 1)]
    result = aggregate_iss(grades, graph)
    assert result.score == 5
    assert result.is_partial is True
    assert result.contributions[0][2] == 2
