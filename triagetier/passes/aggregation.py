from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from triagetier.ir.records import AISGrade, PartialISS
from triagetier.lexicon.graph import KnowledgeGraph
from triagetier.lexicon.iss import aggregate_iss


@dataclass
class AggregationAgent:
    graph: KnowledgeGraph

    def aggregate(self, grades: Sequence[AISGrade]) -> PartialISS:
        return aggregate_iss(grades, self.graph)
