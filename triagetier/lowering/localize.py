from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

ORGAN_HU_BANDS: dict[str, tuple[float, float]] = {
    "liver": (40.0, 80.0),
    "spleen": (45.0, 65.0),
    "kidney": (30.0, 60.0),
}
MIN_COMPONENT_AREA = 500


@dataclass(frozen=True)
class AxialExtent:
    z_start: int
    z_end: int
    found: bool


def _largest_component_area(mask_2d: NDArray[np.bool_]) -> int:
    labels, count = ndimage.label(mask_2d)
    if count == 0:
        return 0
    sizes = ndimage.sum(mask_2d, labels, index=range(1, count + 1))
    return int(np.max(sizes))


def localize_organ(
    volume_hu: NDArray[np.float32],
    hu_low: float,
    hu_high: float,
    min_area: int = MIN_COMPONENT_AREA,
) -> AxialExtent:
    in_band = (volume_hu >= hu_low) & (volume_hu <= hu_high)
    qualifying = np.array(
        [_largest_component_area(in_band[z]) >= min_area for z in range(in_band.shape[0])]
    )
    if not qualifying.any():
        return AxialExtent(0, volume_hu.shape[0] - 1, found=False)
    indices = np.flatnonzero(qualifying)
    return AxialExtent(int(indices[0]), int(indices[-1]), found=True)


def localize_by_name(volume_hu: NDArray[np.float32], organ: str) -> AxialExtent:
    if organ not in ORGAN_HU_BANDS:
        return AxialExtent(0, volume_hu.shape[0] - 1, found=False)
    low, high = ORGAN_HU_BANDS[organ]
    return localize_organ(volume_hu, low, high)
