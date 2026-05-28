from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from triagetier.ir.records import ABDOMINAL_FINDINGS, Case

_LABEL_COLUMNS = ("liver", "spleen", "kidney", "bowel", "extravasation")


@dataclass
class RaticDataset:
    root: Path
    split: str = "test"
    manifest: str = "test_manifest.csv"

    def _manifest_path(self) -> Path:
        path = self.root / self.manifest
        if not path.exists():
            raise FileNotFoundError(
                f"RATIC manifest not found at {path}; download the RSNA 2023 Abdominal "
                "Trauma Detection data and build a manifest with columns "
                "series_path,liver,spleen,kidney,bowel,extravasation"
            )
        return path

    def _load_volume(self, series_path: Path) -> NDArray[np.float32]:
        import nibabel as nib

        image = nib.load(str(series_path))
        return np.asarray(image.get_fdata(), dtype=np.float32)

    def cases(self) -> list[Case]:
        rows = self._read_rows()
        out: list[Case] = []
        for row in rows:
            series = self.root / row["series_path"]
            labels = {organ: float(row.get(organ, 0.0)) for organ in _LABEL_COLUMNS}
            out.append(
                Case(
                    case_id=row.get("study_id", series.stem),
                    modality="image",
                    organs=ABDOMINAL_FINDINGS,
                    volume=self._load_volume(series),
                    spacing=None,
                    context={
                        "anatomy": "abdomen",
                        "mechanism": row.get("mechanism", "blunt trauma"),
                    },
                    labels=labels,
                )
            )
        return out

    def _read_rows(self) -> list[dict[str, Any]]:
        with self._manifest_path().open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
