from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from triagetier.lowering.localize import AxialExtent

SOLID_PERCENTILES: tuple[float, ...] = (10.0, 25.0, 40.0, 50.0, 60.0, 75.0, 90.0)
BOWEL_PERCENTILES: tuple[float, ...] = tuple(float(p) for p in range(5, 100, 10))


def percentiles_for(organ: str) -> tuple[float, ...]:
    return BOWEL_PERCENTILES if organ == "bowel" else SOLID_PERCENTILES


def select_slices(
    extent: AxialExtent,
    percentiles: Sequence[float],
    rng: np.random.Generator | None = None,
    jitter_fraction: float = 0.0,
) -> list[int]:
    span = extent.z_end - extent.z_start
    if span <= 0:
        return [extent.z_start]
    chosen: list[int] = []
    for pct in percentiles:
        offset = pct / 100.0
        if rng is not None and jitter_fraction > 0.0:
            offset += float(rng.uniform(-jitter_fraction, jitter_fraction))
        offset = min(max(offset, 0.0), 1.0)
        chosen.append(extent.z_start + int(round(offset * span)))
    ordered = sorted(set(min(max(z, extent.z_start), extent.z_end) for z in chosen))
    return ordered


def select_for_organ(
    extent: AxialExtent,
    organ: str,
    rng: np.random.Generator | None = None,
    jitter_fraction: float = 0.0,
) -> list[int]:
    return select_slices(extent, percentiles_for(organ), rng, jitter_fraction)
