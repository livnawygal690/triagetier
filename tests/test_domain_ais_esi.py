from __future__ import annotations

import pytest

from triagetier.lexicon.ais import normalize_grade, organ_ais
from triagetier.lexicon.esi import esi_level, vitals_unstable
from triagetier.lexicon.graph import KnowledgeGraph

STABLE = {"systolic_bp": 120.0, "heart_rate": 84.0}


@pytest.mark.parametrize(
    "raw,expected",
    [("II", "II"), ("2", "II"), ("grade iii", "III"), ("V", "V"), ("none", None), ("0", None)],
)
def test_normalize_grade(raw: str, expected: str | None) -> None:
    assert normalize_grade(raw) == expected


def test_organ_ais_uses_graph(graph: KnowledgeGraph) -> None:
    assert organ_ais("liver", "II", graph) == 2
    assert organ_ais("spleen", "I", graph) == 1
    assert organ_ais("kidney", "V", graph) == 5
    assert organ_ais("liver", None, graph) == 0


def test_esi_boundary_at_iss_five(graph: KnowledgeGraph) -> None:
    thresholds = graph.esi_thresholds()
    assert esi_level(5, 2, STABLE, thresholds)[0] == 3
    assert esi_level(4, 2, STABLE, thresholds)[0] == 4


def test_esi_high_risk_and_lifesaving(graph: KnowledgeGraph) -> None:
    thresholds = graph.esi_thresholds()
    assert esi_level(17, 3, STABLE, thresholds)[0] == 2
    assert esi_level(0, 5, STABLE, thresholds)[0] == 1
    assert esi_level(0, 4, STABLE, thresholds)[0] == 2


def test_esi_floor_and_instability(graph: KnowledgeGraph) -> None:
    thresholds = graph.esi_thresholds()
    assert esi_level(0, 0, STABLE, thresholds)[0] == 5
    assert esi_level(4, 2, {"systolic_bp": 80.0}, thresholds)[0] == 2


def test_vitals_unstable(graph: KnowledgeGraph) -> None:
    thresholds = graph.esi_thresholds()
    assert vitals_unstable({"systolic_bp": 80.0}, thresholds) is True
    assert vitals_unstable({"heart_rate": 130.0}, thresholds) is True
    assert vitals_unstable(STABLE, thresholds) is False
