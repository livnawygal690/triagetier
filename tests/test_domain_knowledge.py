from __future__ import annotations

import pytest

from triagetier.lexicon.graph import KnowledgeGraph


def test_region_structure(graph: KnowledgeGraph) -> None:
    assert len(graph.regions) == 6
    assert len(graph.grade_order) == 5
    assert graph.imaged_regions() == ("abdomen", "head_neck")


def test_organ_membership(graph: KnowledgeGraph) -> None:
    assert graph.has_organ("liver")
    assert graph.has_organ("spleen")
    assert graph.has_organ("kidney")
    assert not graph.has_organ("bowel")


def test_region_mapping(graph: KnowledgeGraph) -> None:
    assert graph.region_of("liver") == "abdomen"
    assert graph.region_of("subdural") == "head_neck"
    with pytest.raises(KeyError):
        graph.region_of("unknown_finding")


def test_aast_grade_to_ais(graph: KnowledgeGraph) -> None:
    assert graph.aast_to_ais("liver", "II") == 2
    assert graph.aast_to_ais("kidney", "IV") == 4


def test_esi_thresholds_loaded(graph: KnowledgeGraph) -> None:
    thresholds = graph.esi_thresholds()
    assert thresholds.high_risk_min_iss == 16
    assert thresholds.multi_resource_min_iss == 5
    assert thresholds.shock_systolic_bp == 90.0
