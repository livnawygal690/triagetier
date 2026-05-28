from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triagetier.ir.records import AISGrade, OrganFinding
from triagetier.passes.qa import QualityAssurance


@dataclass(frozen=True)
class ReActResult:
    grades: tuple[AISGrade, ...]
    iterations: int
    resolved: bool
    issues: tuple[str, ...]


def react_loop(
    findings: Sequence[OrganFinding],
    grades: Sequence[AISGrade],
    qa: QualityAssurance,
    vitals: Mapping[str, float],
    active_extravasation: bool,
    max_iter: int = 3,
) -> ReActResult:
    current = list(grades)
    issues = qa.check(findings, current, vitals, active_extravasation)
    iterations = 1
    while issues and iterations < max_iter:
        if not qa.consistency_issues(findings, current):
            break
        current = qa.reconcile(findings, current)
        iterations += 1
        issues = qa.check(findings, current, vitals, active_extravasation)
    return ReActResult(tuple(current), iterations, not issues, tuple(issues))
