from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triagetier.ir.records import AISGrade, OrganFinding
from triagetier.lexicon.ais import organ_ais
from triagetier.lexicon.esi import vitals_unstable
from triagetier.lexicon.graph import KnowledgeGraph

_SEVERE_TERMS = (
    "laceration",
    "disruption",
    "discontinuity",
    "devascular",
    "shattered",
    "rupture",
    "active bleeding",
    "extravasation",
)


def _finding_for(findings: Sequence[OrganFinding], organ: str) -> OrganFinding | None:
    for finding in findings:
        if finding.organ == organ:
            return finding
    return None


@dataclass
class QualityAssurance:
    graph: KnowledgeGraph

    def consistency_issues(
        self, findings: Sequence[OrganFinding], grades: Sequence[AISGrade]
    ) -> list[str]:
        issues: list[str] = []
        for grade in grades:
            if grade.aast_grade < 3:
                continue
            finding = _finding_for(findings, grade.organ)
            phrase = finding.severity_phrase.lower() if finding else ""
            if not any(term in phrase for term in _SEVERE_TERMS):
                issues.append(
                    f"{grade.organ} graded {grade.aast_grade} but detection lacks a "
                    "structural-injury descriptor"
                )
        return issues

    def plausibility_issues(
        self,
        grades: Sequence[AISGrade],
        vitals: Mapping[str, float],
        active_extravasation: bool,
    ) -> list[str]:
        thresholds = self.graph.esi_thresholds()
        unstable = vitals_unstable(vitals, thresholds)
        have_hr = "heart_rate" in vitals
        have_bp = "systolic_bp" in vitals
        by_organ = {g.organ: g for g in grades}

        def grade_int(organ: str) -> int:
            return by_organ[organ].aast_grade if organ in by_organ else 0

        high = [g for g in grades if g.aast_grade >= 4]
        grade5 = [g for g in grades if g.aast_grade >= 5]
        issues: list[str] = []

        if grade_int("kidney") >= 5 and have_bp and not unstable:
            issues.append("grade V kidney injury with non-shock blood pressure")
        if (
            active_extravasation
            and have_hr
            and vitals["heart_rate"] <= thresholds.tachycardia_hr
        ):
            issues.append("active extravasation without compensatory tachycardia")
        if grade5 and have_bp and have_hr and not unstable:
            issues.append("grade V solid-organ injury with fully stable vitals")
        if len(high) >= 2 and not unstable and (have_bp or have_hr):
            issues.append("multiple grade IV+ injuries without haemodynamic compromise")
        if (
            grade_int("spleen") >= 5
            and have_hr
            and vitals["heart_rate"] <= thresholds.tachycardia_hr
        ):
            issues.append("shattered spleen without tachycardia")
        if (
            grade_int("liver") >= 5
            and have_bp
            and vitals["systolic_bp"] >= thresholds.shock_systolic_bp
        ):
            issues.append("grade V liver injury without hypotension")
        if (
            grade_int("kidney") >= 4
            and grade_int("kidney") == grade_int("spleen") >= 4
            and not unstable
        ):
            issues.append("bilateral-pattern high-grade injury without instability")
        iss_like = sum(
            min(g.ais, 5) ** 2 for g in sorted(grades, key=lambda x: x.ais, reverse=True)[:3]
        )
        if iss_like >= 25 and have_bp and have_hr and not unstable:
            issues.append("severe aggregate injury burden with stable vitals")
        if active_extravasation and not any(g.aast_grade > 0 for g in grades):
            issues.append("active extravasation without any graded organ injury")
        if have_hr and vitals["heart_rate"] < 40.0 and high:
            issues.append("high-grade injury with implausible bradycardia")
        if grade_int("liver") >= 4 and grade_int("spleen") >= 4 and not unstable:
            issues.append("combined hepatic and splenic high-grade injury without shock")
        if unstable and not grades and not active_extravasation:
            issues.append("haemodynamic instability without any detected injury source")
        return issues

    def check(
        self,
        findings: Sequence[OrganFinding],
        grades: Sequence[AISGrade],
        vitals: Mapping[str, float],
        active_extravasation: bool,
    ) -> list[str]:
        return self.consistency_issues(findings, grades) + self.plausibility_issues(
            grades, vitals, active_extravasation
        )

    def reconcile(
        self, findings: Sequence[OrganFinding], grades: Sequence[AISGrade]
    ) -> list[AISGrade]:
        flagged = {
            issue.split(" ", 1)[0] for issue in self.consistency_issues(findings, grades)
        }
        adjusted: list[AISGrade] = []
        for grade in grades:
            if grade.organ in flagged and grade.aast_grade > 2:
                capped = organ_ais(grade.organ, "II", self.graph)
                adjusted.append(
                    AISGrade(
                        organ=grade.organ,
                        region=grade.region,
                        aast_grade=2,
                        ais=capped,
                        justification=grade.justification + " (reconciled to detection)",
                    )
                )
            else:
                adjusted.append(grade)
        return adjusted
