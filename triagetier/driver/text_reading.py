from __future__ import annotations

import re

from triagetier.ir.records import AISGrade
from triagetier.lexicon.ais import grade_to_int, normalize_grade, organ_ais
from triagetier.lexicon.graph import KnowledgeGraph

_GRADE_PATTERN = re.compile(
    r"(liver|spleen|kidney)\D{0,24}?grade\s+(v|iv|iii|ii|i|[1-5])", re.IGNORECASE
)
_EXTRAV = re.compile(r"active\s+(extravasation|bleeding|haemorrhage|hemorrhage)", re.IGNORECASE)


def parse_clinical_grades(text: str, graph: KnowledgeGraph) -> list[AISGrade]:
    grades: list[AISGrade] = []
    seen: set[str] = set()
    for match in _GRADE_PATTERN.finditer(text):
        organ = match.group(1).lower()
        if organ in seen:
            continue
        seen.add(organ)
        grade = normalize_grade(match.group(2))
        if grade is None:
            continue
        grades.append(
            AISGrade(
                organ=organ,
                region=graph.region_of(organ),
                aast_grade=grade_to_int(grade),
                ais=organ_ais(organ, grade, graph),
                justification="documented in clinical note",
            )
        )
    return grades


def mentions_active_extravasation(text: str) -> bool:
    return bool(_EXTRAV.search(text))
