from __future__ import annotations

from triagetier.ir.records import AISGrade, OrganFinding
from triagetier.lexicon.graph import KnowledgeGraph
from triagetier.passes.qa import QualityAssurance

STABLE = {"systolic_bp": 122.0, "heart_rate": 80.0}


def _finding(organ: str, present: bool, severity: str) -> OrganFinding:
    return OrganFinding(organ, present, "location", severity, "raw")


def _grade(organ: str, value: int) -> AISGrade:
    return AISGrade(organ, "abdomen", value, value, "")


def test_consistency_flags_grade_without_descriptor(graph: KnowledgeGraph) -> None:
    qa = QualityAssurance(graph)
    findings = [_finding("spleen", True, "mild contusion")]
    issues = qa.consistency_issues(findings, [_grade("spleen", 3)])
    assert issues


def test_consistency_passes_with_descriptor(graph: KnowledgeGraph) -> None:
    qa = QualityAssurance(graph)
    findings = [_finding("spleen", True, "deep laceration with active bleeding")]
    assert qa.consistency_issues(findings, [_grade("spleen", 3)]) == []


def test_plausibility_detects_named_rules(graph: KnowledgeGraph) -> None:
    qa = QualityAssurance(graph)
    kidney_issue = qa.plausibility_issues([_grade("kidney", 5)], STABLE, False)
    assert any("kidney" in issue for issue in kidney_issue)
    extrav_issue = qa.plausibility_issues([], {"heart_rate": 80.0}, True)
    assert any("extravasation" in issue for issue in extrav_issue)


def test_reconcile_caps_inconsistent_grade(graph: KnowledgeGraph) -> None:
    qa = QualityAssurance(graph)
    findings = [_finding("spleen", True, "mild contusion")]
    adjusted = qa.reconcile(findings, [_grade("spleen", 3)])
    assert adjusted[0].aast_grade == 2
    assert adjusted[0].ais == 2
