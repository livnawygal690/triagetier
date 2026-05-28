from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triagetier.lexicon import loader


@dataclass(frozen=True)
class ESIThresholds:
    lifesaving_min_ais: int
    high_risk_min_ais: int
    high_risk_min_iss: int
    multi_resource_min_iss: int
    single_resource_min_iss: int
    shock_systolic_bp: float
    tachycardia_hr: float


@dataclass(frozen=True)
class KnowledgeGraph:
    aast: dict[str, Any]
    iss_regions: dict[str, Any]
    esi: dict[str, Any]

    @property
    def grade_order(self) -> list[str]:
        order: list[str] = self.aast["grade_order"]
        return order

    @property
    def regions(self) -> list[str]:
        regs: list[str] = self.iss_regions["regions"]
        return regs

    def has_organ(self, organ: str) -> bool:
        return organ in self.aast["organs"]

    def grade_criteria(self, organ: str, grade: str) -> str:
        text: str = self.aast["organs"][organ][grade]["imaging"]
        return text

    def aast_to_ais(self, organ: str, grade: str) -> int:
        ais: int = self.aast["organs"][organ][grade]["ais"]
        return ais

    def region_of(self, finding: str) -> str:
        mapping: dict[str, str] = self.iss_regions["finding_to_region"]
        if finding not in mapping:
            raise KeyError(f"no ISS region mapping for finding '{finding}'")
        return mapping[finding]

    def imaged_regions(self) -> tuple[str, ...]:
        return tuple(self.iss_regions["imaged_by"].keys())

    def esi_thresholds(self) -> ESIThresholds:
        decision = self.esi["decision"]
        physiology = self.esi["physiology"]
        return ESIThresholds(
            lifesaving_min_ais=int(decision["lifesaving_min_ais"]),
            high_risk_min_ais=int(decision["high_risk_min_ais"]),
            high_risk_min_iss=int(decision["high_risk_min_iss"]),
            multi_resource_min_iss=int(decision["multi_resource_min_iss"]),
            single_resource_min_iss=int(decision["single_resource_min_iss"]),
            shock_systolic_bp=float(physiology["shock_systolic_bp"]),
            tachycardia_hr=float(physiology["tachycardia_hr"]),
        )


def load_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        aast=loader.load_aast(),
        iss_regions=loader.load_iss_regions(),
        esi=loader.load_esi(),
    )
