from __future__ import annotations

from pathlib import Path
from typing import Any

from triagetier.frontends.cq500 import CQ500Dataset
from triagetier.frontends.mimic_iv_ed import MimicEdDataset
from triagetier.frontends.ratic import RaticDataset
from triagetier.frontends.synthetic import synthetic_cases, synthetic_text_cases
from triagetier.ir.records import Case


def build_dataset(data_cfg: dict[str, Any]) -> list[Case]:
    kind = str(data_cfg.get("kind", data_cfg.get("name", "synthetic")))
    if kind == "synthetic":
        return synthetic_cases(int(data_cfg.get("n", 6)), int(data_cfg.get("seed", 0)))
    if kind == "synthetic_text":
        return synthetic_text_cases(int(data_cfg.get("n", 6)), int(data_cfg.get("seed", 0)))
    root = Path(str(data_cfg.get("root", "data/raw")))
    if kind == "ratic":
        return RaticDataset(root, split=str(data_cfg.get("split", "test"))).cases()
    if kind == "cq500":
        return CQ500Dataset(root).cases()
    if kind == "mimic_iv_ed":
        return MimicEdDataset(root, n=int(data_cfg.get("n", 2000))).cases()
    raise ValueError(f"unknown dataset kind '{kind}'")
