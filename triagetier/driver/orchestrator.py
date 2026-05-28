from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from triagetier.driver.react import react_loop
from triagetier.driver.text_reading import mentions_active_extravasation, parse_clinical_grades
from triagetier.host.config import PipelineConfig
from triagetier.host.seed import set_seed
from triagetier.ir.records import (
    AISGrade,
    Case,
    CaseResult,
    ConfidenceReport,
    OrganFinding,
    PartialISS,
    QAOutcome,
    TriageDecision,
)
from triagetier.lexicon import confidence
from triagetier.lexicon.graph import KnowledgeGraph, load_graph
from triagetier.lowering.localize import AxialExtent, localize_by_name
from triagetier.lowering.sampling import percentiles_for, select_slices
from triagetier.lowering.windowing import (
    ABDOMINAL_WINDOWS,
    BRAIN,
    HEAD_WINDOWS,
    SOFT_TISSUE,
    CTWindow,
    tile_windows,
)
from triagetier.oracles.base import VLMBackend
from triagetier.passes.aggregation import AggregationAgent
from triagetier.passes.detection import DetectionAgent
from triagetier.passes.grading import GradingAgent
from triagetier.passes.qa import QualityAssurance
from triagetier.passes.triage import TriageAgent


@dataclass(frozen=True)
class _Toggles:
    slice_selection: bool
    multi_window: bool
    qa: bool
    knowledge_graph: bool
    k_solid: int


def _effective(cfg: PipelineConfig) -> _Toggles:
    if cfg.single_pass:
        return _Toggles(False, False, False, False, 1)
    return _Toggles(
        cfg.use_slice_selection,
        cfg.multi_window,
        cfg.use_qa,
        cfg.use_knowledge_graph,
        cfg.k_solid,
    )


class Orchestrator:
    def __init__(
        self, backend: VLMBackend, cfg: PipelineConfig, graph: KnowledgeGraph | None = None
    ):
        self.backend = backend
        self.cfg = cfg
        self.graph = graph or load_graph()
        self.detection = DetectionAgent(backend)
        self.grading = GradingAgent(backend, self.graph)
        self.aggregation = AggregationAgent(self.graph)
        self.triage = TriageAgent(backend, self.graph)
        self.qa = QualityAssurance(self.graph)

    def run_case(self, case: Case, seed: int) -> CaseResult:
        set_seed(seed)
        if case.modality == "text":
            return self._run_text(case, seed)
        return self._run_image(case, seed)

    def _windows(self, case: Case, toggles: _Toggles) -> tuple[CTWindow, ...]:
        head = case.context.get("anatomy") == "head"
        full = HEAD_WINDOWS if head else ABDOMINAL_WINDOWS
        if toggles.multi_window:
            return full
        return (BRAIN,) if head else (SOFT_TISSUE,)

    def _extent(
        self, volume: NDArray[np.float32], organ: str, toggles: _Toggles
    ) -> AxialExtent:
        if toggles.slice_selection:
            return localize_by_name(volume, organ)
        return AxialExtent(0, volume.shape[0] - 1, found=False)

    def _slice_indices(
        self,
        volume: NDArray[np.float32],
        organ: str,
        toggles: _Toggles,
        rng: np.random.Generator,
    ) -> list[int]:
        depth = volume.shape[0]
        k = toggles.k_solid if organ != "bowel" else self.cfg.k_bowel
        strategy = "random" if not toggles.slice_selection else self.cfg.slice_strategy
        if strategy == "random":
            return sorted({int(z) for z in rng.integers(0, depth, size=min(k, depth))})
        if strategy == "max_entropy":
            variance = volume.reshape(depth, -1).var(axis=1)
            return sorted(int(z) for z in np.argsort(variance)[-k:])
        extent = self._extent(volume, organ, toggles)
        if strategy == "uniform":
            return select_slices(extent, tuple(np.linspace(5, 95, k)))
        return select_slices(extent, percentiles_for(organ), rng, self.cfg.jitter_fraction)

    def _organ_images(
        self, case: Case, organ: str, toggles: _Toggles, rng: np.random.Generator
    ) -> list[NDArray[np.uint8]]:
        volume = case.volume
        if volume is None:
            return []
        windows = self._windows(case, toggles)
        return [
            tile_windows(volume[z], windows)
            for z in self._slice_indices(volume, organ, toggles, rng)
        ]

    def _run_image(self, case: Case, seed: int) -> CaseResult:
        toggles = _effective(self.cfg)
        rng = np.random.default_rng(seed)
        findings: list[OrganFinding] = []
        for organ in case.organs:
            images = self._organ_images(case, organ, toggles, rng)
            findings.append(self.detection.detect(organ, images))

        grades = [self.grading.grade(f) for f in findings if self.grading.gradable(f)]
        extravasation = any(f.organ == "extravasation" and f.present for f in findings)

        if toggles.qa:
            outcome = react_loop(
                findings, grades, self.qa, case.vitals, extravasation, self.cfg.max_qa_iter
            )
            grades = list(outcome.grades)
            qa = QAOutcome(outcome.iterations, outcome.resolved, outcome.issues)
        else:
            qa = QAOutcome(0, True, ())

        return self._assemble(case, seed, tuple(findings), grades, extravasation, qa)

    def _run_text(self, case: Case, seed: int) -> CaseResult:
        grades = parse_clinical_grades(case.clinical_text, self.graph)
        extravasation = mentions_active_extravasation(case.clinical_text)
        qa = QAOutcome(0, True, ())
        if self.cfg.use_qa:
            outcome = react_loop(
                (), grades, self.qa, case.vitals, extravasation, self.cfg.max_qa_iter
            )
            grades = list(outcome.grades)
            qa = QAOutcome(outcome.iterations, outcome.resolved, outcome.issues)
        return self._assemble(case, seed, (), grades, extravasation, qa)

    def _assemble(
        self,
        case: Case,
        seed: int,
        findings: tuple[OrganFinding, ...],
        grades: Sequence[AISGrade],
        extravasation: bool,
        qa: QAOutcome,
    ) -> CaseResult:
        iss: PartialISS = self.aggregation.aggregate(grades)
        triage: TriageDecision = self.triage.decide(
            iss, grades, case.vitals, case.context, extravasation
        )
        report: ConfidenceReport = self._confidence(findings, grades, qa)
        return CaseResult(
            case_id=case.case_id,
            findings=findings,
            grades=tuple(grades),
            iss=iss,
            triage=triage,
            confidence=report,
            qa=qa,
            seed=seed,
        )

    def _confidence(
        self, findings: Sequence[OrganFinding], grades: Sequence[AISGrade], qa: QAOutcome
    ) -> ConfidenceReport:
        present = [f for f in findings if f.present]
        detection_scores = [
            confidence.confidence_score(f.term_logprobs, [f.present, f.present, f.present])
            for f in present
        ]
        per_agent = {
            "detection": min(detection_scores) if detection_scores else 1.0,
            "grading": 0.6 if any("reconciled" in g.justification for g in grades) else 1.0,
            "qa": 1.0 if qa.resolved else 0.2,
        }
        return confidence.aggregate(per_agent, self.cfg.confidence_threshold)
