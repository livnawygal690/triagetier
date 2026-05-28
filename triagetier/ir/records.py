from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Modality = Literal["image", "text"]
SOLID_ORGANS: tuple[str, ...] = ("liver", "spleen", "kidney")
ABDOMINAL_FINDINGS: tuple[str, ...] = ("liver", "spleen", "kidney", "bowel", "extravasation")


@dataclass(frozen=True)
class Case:
    case_id: str
    modality: Modality
    organs: tuple[str, ...]
    volume: NDArray[np.float32] | None = None
    spacing: tuple[float, float, float] | None = None
    clinical_text: str = ""
    vitals: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OrganFinding:
    organ: str
    present: bool
    location: str
    severity_phrase: str
    raw_text: str
    term_logprobs: tuple[float, ...] = ()


@dataclass(frozen=True)
class AISGrade:
    organ: str
    region: str
    aast_grade: int
    ais: int
    justification: str


@dataclass(frozen=True)
class PartialISS:
    score: int
    contributions: tuple[tuple[str, str, int], ...]
    imaged_regions: tuple[str, ...]
    is_partial: bool


@dataclass(frozen=True)
class TriageDecision:
    esi: int
    rationale: str
    drivers: tuple[str, ...]


@dataclass(frozen=True)
class ConfidenceReport:
    score: float
    flagged_for_review: bool
    per_agent: Mapping[str, float]


@dataclass(frozen=True)
class QAOutcome:
    iterations: int
    resolved: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    findings: tuple[OrganFinding, ...]
    grades: tuple[AISGrade, ...]
    iss: PartialISS
    triage: TriageDecision
    confidence: ConfidenceReport
    qa: QAOutcome
    seed: int


def presence_vector(findings: Sequence[OrganFinding]) -> dict[str, int]:
    return {f.organ: int(f.present) for f in findings}
