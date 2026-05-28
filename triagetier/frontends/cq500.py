from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from triagetier.ir.records import Case

ICH_SUBTYPES = (
    "intraparenchymal",
    "intraventricular",
    "subdural",
    "extradural",
    "subarachnoid",
    "calvarial_fracture",
)


@dataclass
class CQ500Dataset:
    root: Path
    manifest: str = "reads.csv"

    def _manifest_path(self) -> Path:
        path = self.root / self.manifest
        if not path.exists():
            raise FileNotFoundError(
                f"CQ500 reads file not found at {path}; obtain the dataset from the "
                "qure.ai head-CT release and provide a manifest with a series_path column "
                "plus the three-radiologist consensus labels"
            )
        return path

    def _load_volume(self, series_path: Path) -> NDArray[np.float32]:
        import nibabel as nib

        return np.asarray(nib.load(str(series_path)).get_fdata(), dtype=np.float32)

    def cases(self) -> list[Case]:
        out: list[Case] = []
        for row in self._read_rows():
            series = self.root / row["series_path"]
            labels = {name: float(row.get(name, 0.0)) for name in ICH_SUBTYPES}
            labels["ich"] = float(any(labels[name] > 0 for name in ICH_SUBTYPES[:-1]))
            out.append(
                Case(
                    case_id=row.get("name", series.stem),
                    modality="image",
                    organs=ICH_SUBTYPES,
                    volume=self._load_volume(series),
                    context={
                        "anatomy": "head",
                        "mechanism": row.get("mechanism", "head trauma"),
                    },
                    labels=labels,
                )
            )
        return out

    def _read_rows(self) -> list[dict[str, Any]]:
        with self._manifest_path().open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
