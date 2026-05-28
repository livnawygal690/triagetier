from __future__ import annotations

from triagetier.driver.orchestrator import Orchestrator
from triagetier.frontends.synthetic import synthetic_cases
from triagetier.host.config import PipelineConfig
from triagetier.oracles.mock import MockVLM, ScriptEntry


def test_hepatic_two_splenic_one_gives_iss_five_esi_three() -> None:
    orchestrator = Orchestrator(MockVLM(), PipelineConfig(jitter_fraction=0.0))
    result = orchestrator.run_case(synthetic_cases(1, 0)[0], 0)
    assert result.iss.score == 5
    assert result.triage.esi == 3
    assert result.iss.is_partial is True


def test_single_abdominal_ais_two_gives_iss_four_esi_four() -> None:
    script = {
        "liver": ScriptEntry(True, "right hepatic lobe", "linear laceration", "II"),
        "spleen": ScriptEntry(False),
        "kidney": ScriptEntry(False),
        "bowel": ScriptEntry(False),
        "extravasation": ScriptEntry(False),
    }
    orchestrator = Orchestrator(MockVLM(script=script), PipelineConfig(jitter_fraction=0.0))
    result = orchestrator.run_case(synthetic_cases(1, 0)[0], 0)
    assert result.iss.score == 4
    assert result.triage.esi == 4
