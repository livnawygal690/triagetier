from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triagetier.ir.records import Case

_TRAUMA_ICD = re.compile(r"^[ST]\d", re.IGNORECASE)
_EXCLUDED = (
    "T20",
    "T21",
    "T22",
    "T23",
    "T24",
    "T25",
    "T26",
    "T27",
    "T28",
    "T29",
    "T30",
    "T31",
    "T32",
    "T36",
    "T40",
    "T50",
    "T65",
    "T80",
    "T81",
    "T82",
    "T83",
    "T84",
    "T85",
    "T86",
    "T87",
    "T88",
)
_KEYWORDS = ("injury", "trauma", "fall", "collision", "assault", "laceration")


@dataclass
class MimicEdDataset:
    root: Path
    n: int = 2000
    edstays: str = "edstays.csv"
    triage: str = "triage.csv"
    diagnosis: str = "diagnosis.csv"
    exclude_under: int = 18
    _rng_field: int = field(default=0, repr=False)

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(
                f"MIMIC-IV-ED file {name} not found at {path}; credentialed PhysioNet "
                "access (mimic-iv-ed v2.2) is required"
            )
        return path

    def _is_trauma(self, code: str) -> bool:
        upper = code.upper().replace(".", "")
        if not _TRAUMA_ICD.match(upper):
            return False
        return not any(upper.startswith(prefix) for prefix in _EXCLUDED)

    def cases(self) -> list[Case]:
        diagnoses = self._group_diagnoses()
        triage_rows = self._index(self.triage, "stay_id")
        out: list[Case] = []
        for row in self._read(self.edstays):
            stay = row.get("stay_id", "")
            codes = diagnoses.get(stay, [])
            note = " ".join(codes)
            if not any(self._is_trauma(code) for code in codes) and not any(
                key in note.lower() for key in _KEYWORDS
            ):
                continue
            triage = triage_rows.get(stay, {})
            esi = triage.get("acuity")
            if not esi:
                continue
            out.append(
                Case(
                    case_id=stay,
                    modality="text",
                    organs=(),
                    clinical_text=triage.get("chiefcomplaint", "") + " " + note,
                    vitals=_vitals(triage),
                    context={
                        "mechanism": "trauma",
                        "chief_complaint": triage.get("chiefcomplaint", ""),
                    },
                    labels={"esi": float(esi)},
                )
            )
            if len(out) >= self.n:
                break
        return out

    def _group_diagnoses(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for row in self._read(self.diagnosis):
            grouped.setdefault(row.get("stay_id", ""), []).append(row.get("icd_code", ""))
        return grouped

    def _index(self, name: str, key: str) -> dict[str, dict[str, Any]]:
        return {row.get(key, ""): row for row in self._read(name)}

    def _read(self, name: str) -> list[dict[str, Any]]:
        with self._path(name).open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def _vitals(triage: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    if triage.get("sbp"):
        out["systolic_bp"] = float(triage["sbp"])
    if triage.get("heartrate"):
        out["heart_rate"] = float(triage["heartrate"])
    return out
