from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from triagetier.ir.records import ABDOMINAL_FINDINGS, Case


def synthetic_volume(seed: int, depth: int = 24, size: int = 48) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    volume = rng.normal(-40.0, 8.0, size=(depth, size, size)).astype(np.float32)
    lo, hi = depth // 3, 2 * depth // 3
    volume[lo:hi, 12:36, 12:36] = 60.0
    return volume


def synthetic_cases(n: int = 6, seed: int = 0) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    for i in range(n):
        injured = bool(i % 2)
        esi = 3 if injured else 4
        labels: dict[str, float] = {organ: 0.0 for organ in ABDOMINAL_FINDINGS}
        if injured:
            labels["liver"] = 1.0
            labels["spleen"] = 1.0
        labels["esi"] = float(esi)
        cases.append(
            Case(
                case_id=f"synthetic-{i:03d}",
                modality="image",
                organs=ABDOMINAL_FINDINGS,
                volume=synthetic_volume(seed + i),
                spacing=(5.0, 1.0, 1.0),
                vitals={
                    "systolic_bp": float(110 + int(rng.integers(0, 20))),
                    "heart_rate": 88.0,
                },
                context={"anatomy": "abdomen", "mechanism": "motor vehicle collision"},
                labels=labels,
            )
        )
    return cases


def synthetic_text_cases(n: int = 6, seed: int = 0) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    templates = [
        ("blunt abdominal trauma with spleen grade III laceration", 2),
        ("minor fall, no acute injury documented", 5),
        ("liver grade II laceration after collision", 3),
        ("laceration of forearm, stable", 4),
    ]
    for i in range(n):
        note, esi = templates[i % len(templates)]
        cases.append(
            Case(
                case_id=f"mimic-syn-{i:03d}",
                modality="text",
                organs=(),
                clinical_text=note,
                vitals={
                    "systolic_bp": float(120 - 5 * (i % 4)),
                    "heart_rate": float(90 + 4 * (i % 5)),
                },
                context={"mechanism": "trauma", "chief_complaint": "injury"},
                labels={"esi": float(esi)},
            )
        )
    _ = rng
    return cases
