from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triagetier.ir.records import AISGrade, PartialISS, TriageDecision
from triagetier.lexicon.esi import esi_level
from triagetier.lexicon.graph import KnowledgeGraph
from triagetier.oracles.base import VLMBackend
from triagetier.passes.base import parse_fields


@dataclass
class TriageAgent:
    backend: VLMBackend
    graph: KnowledgeGraph

    def _rationale(self, esi: int, drivers: Sequence[str], context: Mapping[str, str]) -> str:
        mechanism = context.get("mechanism", "unspecified mechanism")
        generation = self.backend.generate(
            f"role=triage esi={esi}\nMechanism: {mechanism}. Drivers: {', '.join(drivers)}.",
            (),
        )
        note = parse_fields(generation.text).get("rationale", generation.text.strip())
        return note or f"ESI {esi} from {', '.join(drivers)}"

    def decide(
        self,
        iss: PartialISS,
        grades: Sequence[AISGrade],
        vitals: Mapping[str, float],
        context: Mapping[str, str],
        active_extravasation: bool,
    ) -> TriageDecision:
        max_ais = max((g.ais for g in grades), default=0)
        esi, drivers = esi_level(iss.score, max_ais, vitals, self.graph.esi_thresholds())
        driver_list = list(drivers)
        if active_extravasation and esi > 2:
            esi = 2
            driver_list.append("active extravasation")
        rationale = self._rationale(esi, driver_list, context)
        return TriageDecision(esi=esi, rationale=rationale, drivers=tuple(driver_list))
