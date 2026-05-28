from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CTWindow:
    name: str
    width: float
    level: float


SOFT_TISSUE = CTWindow("soft_tissue", 400.0, 50.0)
LIVER = CTWindow("liver", 150.0, 30.0)
BONE = CTWindow("bone", 1800.0, 400.0)
BRAIN = CTWindow("brain", 80.0, 40.0)
SUBDURAL = CTWindow("subdural", 150.0, 125.0)

ABDOMINAL_WINDOWS: tuple[CTWindow, ...] = (SOFT_TISSUE, LIVER, BONE)
HEAD_WINDOWS: tuple[CTWindow, ...] = (BRAIN, SUBDURAL, BONE)


def apply_window(slice_hu: NDArray[np.float32], window: CTWindow) -> NDArray[np.uint8]:
    low = window.level - window.width / 2.0
    scaled = (slice_hu - low) / window.width * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def tile_windows(
    slice_hu: NDArray[np.float32], windows: Sequence[CTWindow]
) -> NDArray[np.uint8]:
    panels = [apply_window(slice_hu, w) for w in windows]
    return np.concatenate(panels, axis=1)
