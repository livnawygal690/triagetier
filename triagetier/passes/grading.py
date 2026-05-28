from __future__ import annotations

from dataclasses import dataclass

from triagetier.ir.records import AISGrade, OrganFinding
from triagetier.lexicon.ais import grade_to_int, normalize_grade, organ_ais
from triagetier.lexicon.graph import KnowledgeGraph
from triagetier.oracles.base import VLMBackend
from triagetier.passes.base import parse_fields


@dataclass
class GradingAgent:
    backend: VLMBackend
    graph: KnowledgeGraph

    def gradable(self, finding: OrganFinding) -> bool:
        return finding.present and self.graph.has_organ(finding.organ)

    def _rubric(self, organ: str) -> str:
        grades = self.graph.aast["organs"][organ]
        lines = [f"{g}: {grades[g]['imaging']}" for g in self.graph.grade_order]
        return "\n".join(lines)

    def prompt(self, finding: OrganFinding) -> str:
        return (
            f"role=grading organ={finding.organ}\n"
            f"Finding: {finding.severity_phrase} at {finding.location}.\n"
            f"AAST rubric for {finding.organ}:\n{self._rubric(finding.organ)}\n"
            "Select the highest grade whose imaging criteria are all satisfied. "
            "Respond with fields aast_grade, justification."
        )

    def grade(self, finding: OrganFinding) -> AISGrade:
        generation = self.backend.generate(self.prompt(finding), ())
        fields = parse_fields(generation.text)
        grade = normalize_grade(fields.get("aast_grade", "none"))
        ais = organ_ais(finding.organ, grade, self.graph)
        return AISGrade(
            organ=finding.organ,
            region=self.graph.region_of(finding.organ),
            aast_grade=0 if grade is None else grade_to_int(grade),
            ais=ais,
            justification=fields.get("justification", finding.severity_phrase),
        )
