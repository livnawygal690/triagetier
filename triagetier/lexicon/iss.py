from __future__ import annotations

from collections.abc import Sequence

from triagetier.ir.records import AISGrade, PartialISS
from triagetier.lexicon.graph import KnowledgeGraph

_FULL_BODY_REGIONS = 6


def baker_score(ais_values: Sequence[int]) -> int:
    top = sorted((v for v in ais_values if v > 0), reverse=True)[:3]
    return sum(v * v for v in top)


def aggregate_iss(grades: Sequence[AISGrade], graph: KnowledgeGraph) -> PartialISS:
    scored = [g for g in grades if g.ais > 0]
    ranked = sorted(scored, key=lambda g: g.ais, reverse=True)[:3]
    contributions = tuple((g.organ, g.region, g.ais) for g in ranked)
    score = baker_score([g.ais for g in scored])
    imaged = graph.imaged_regions()
    assessed = {g.region for g in grades} | set(imaged)
    is_partial = len(assessed) < _FULL_BODY_REGIONS
    return PartialISS(
        score=score,
        contributions=contributions,
        imaged_regions=tuple(sorted(imaged)),
        is_partial=is_partial,
    )
