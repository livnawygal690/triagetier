from __future__ import annotations

from triagetier.driver.orchestrator import Orchestrator
from triagetier.frontends.synthetic import synthetic_cases, synthetic_text_cases
from triagetier.host.config import PipelineConfig
from triagetier.oracles.mock import MockVLM, ScriptEntry


def test_full_pipeline_invariants() -> None:
    orchestrator = Orchestrator(MockVLM(), PipelineConfig(jitter_fraction=0.0))
    case = synthetic_cases(1, 0)[0]
    result = orchestrator.run_case(case, 0)
    assert 1 <= result.triage.esi <= 5
    assert result.iss.score >= 0
    assert result.qa.iterations <= 3
    assert len(result.findings) == len(case.organs)
    assert all(g.ais >= 0 for g in result.grades)


def test_low_confidence_flag_on_persistent_implausibility() -> None:
    script = {
        "kidney": ScriptEntry(
            True, "renal hilum", "shattered kidney with devascularization", "V"
        ),
        "liver": ScriptEntry(False),
        "spleen": ScriptEntry(False),
        "bowel": ScriptEntry(False),
        "extravasation": ScriptEntry(False),
    }
    orchestrator = Orchestrator(MockVLM(script=script), PipelineConfig(jitter_fraction=0.0))
    result = orchestrator.run_case(synthetic_cases(1, 0)[0], 0)
    assert result.qa.resolved is False
    assert result.confidence.flagged_for_review is True


def test_text_mode_runs() -> None:
    orchestrator = Orchestrator(MockVLM(), PipelineConfig())
    result = orchestrator.run_case(synthetic_text_cases(1, 0)[0], 0)
    assert 1 <= result.triage.esi <= 5
    assert result.findings == ()


def test_qa_disabled_skips_loop() -> None:
    orchestrator = Orchestrator(MockVLM(), PipelineConfig(use_qa=False, jitter_fraction=0.0))
    result = orchestrator.run_case(synthetic_cases(1, 0)[0], 0)
    assert result.qa.iterations == 0
