from __future__ import annotations

from triagetier.lexicon.graph import KnowledgeGraph

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
_ROMAN_BY_INT = {v: k for k, v in _ROMAN.items()}


def normalize_grade(token: str) -> str | None:
    raw = token.strip().upper().replace("GRADE", "").strip()
    if not raw or raw in {"0", "NONE", "ABSENT", "N/A"}:
        return None
    if raw in _ROMAN:
        return raw
    if raw.isdigit():
        value = int(raw)
        return _ROMAN_BY_INT.get(value)
    return None


def grade_to_int(grade: str) -> int:
    return _ROMAN[grade]


def organ_ais(organ: str, grade: str | None, graph: KnowledgeGraph) -> int:
    if grade is None:
        return 0
    if not graph.has_organ(organ):
        return grade_to_int(grade)
    return graph.aast_to_ais(organ, grade)
