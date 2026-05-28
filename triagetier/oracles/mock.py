from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from triagetier.oracles.base import Generation

_ROLE = re.compile(r"role=([a-z_]+)")
_ORGAN = re.compile(r"organ=([a-z_]+)")


@dataclass(frozen=True)
class ScriptEntry:
    present: bool
    location: str = "not visualized"
    severity: str = "no abnormality"
    grade: str | None = None
    key_logprob: float = -0.04


def _default_script() -> dict[str, ScriptEntry]:
    return {
        "liver": ScriptEntry(True, "right hepatic lobe", "linear parenchymal laceration", "II"),
        "spleen": ScriptEntry(True, "splenic surface", "subcapsular hematoma", "I"),
        "kidney": ScriptEntry(False),
        "bowel": ScriptEntry(False),
        "extravasation": ScriptEntry(False),
    }


@dataclass
class MockVLM:
    script: Mapping[str, ScriptEntry] = field(default_factory=_default_script)
    triage_note: str = "mechanism integrated with imaged severity profile"
    seed: int = 0
    name: str = "mock-vlm"

    def generate(self, prompt: str, images: Sequence[Any] = ()) -> Generation:
        role_match = _ROLE.search(prompt)
        role = role_match.group(1) if role_match else "detection"
        organ_match = _ORGAN.search(prompt)
        organ = organ_match.group(1) if organ_match else ""

        if role == "detection":
            return self._detection(organ)
        if role == "grading":
            return self._grading(organ)
        if role == "triage":
            return Generation(self.triage_note, (-0.03,))
        return Generation("", ())

    def _detection(self, organ: str) -> Generation:
        entry = self.script.get(organ, ScriptEntry(False))
        present = "yes" if entry.present else "no"
        text = (
            f"present: {present}\n"
            f"location: {entry.location}\n"
            f"severity: {entry.severity}"
        )
        logprob = entry.key_logprob if entry.present else -0.02
        return Generation(text, (logprob, logprob))

    def _grading(self, organ: str) -> Generation:
        entry = self.script.get(organ, ScriptEntry(False))
        grade = entry.grade if entry.grade is not None else "none"
        text = f"aast_grade: {grade}\njustification: {entry.severity}"
        return Generation(text, (entry.key_logprob,))
