from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from triagetier.ir.records import OrganFinding
from triagetier.oracles.base import VLMBackend
from triagetier.passes.base import is_positive, parse_fields

_CUES = {
    "liver": "parenchymal laceration depth and subcapsular hematoma in the liver window",
    "spleen": "subcapsular hematoma and trabecular laceration on the soft-tissue window",
    "kidney": "perirenal hematoma and collecting-system involvement",
    "bowel": "wall thickening, free fluid and mesenteric stranding across the cavity",
    "extravasation": "focal contrast pooling indicating active bleeding",
}


@dataclass
class DetectionAgent:
    backend: VLMBackend

    def prompt(self, organ: str) -> str:
        cue = _CUES.get(organ, "trauma-relevant abnormality")
        return (
            f"role=detection organ={organ}\n"
            f"Examine the tiled multi-window axial slices for {organ} injury. "
            f"Attend to {cue}. "
            "Respond with fields present, location, severity."
        )

    def detect(self, organ: str, images: Sequence[Any]) -> OrganFinding:
        generation = self.backend.generate(self.prompt(organ), images)
        fields = parse_fields(generation.text)
        present = is_positive(fields.get("present", "no"))
        return OrganFinding(
            organ=organ,
            present=present,
            location=fields.get("location", "unspecified"),
            severity_phrase=fields.get("severity", ""),
            raw_text=generation.text,
            term_logprobs=generation.token_logprobs,
        )
