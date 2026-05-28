from __future__ import annotations

import numpy as np

from triagetier.lowering.localize import AxialExtent, localize_organ
from triagetier.lowering.sampling import (
    BOWEL_PERCENTILES,
    SOLID_PERCENTILES,
    select_for_organ,
    select_slices,
)
from triagetier.lowering.windowing import (
    ABDOMINAL_WINDOWS,
    SOFT_TISSUE,
    apply_window,
    tile_windows,
)


def test_window_center_maps_to_mid_gray() -> None:
    slice_hu = np.full((4, 4), SOFT_TISSUE.level, dtype=np.float32)
    out = apply_window(slice_hu, SOFT_TISSUE)
    assert out.dtype == np.uint8
    assert int(out[0, 0]) == 127


def test_window_clips_to_bounds() -> None:
    slice_hu = np.array([[-5000.0, 5000.0]], dtype=np.float32)
    out = apply_window(slice_hu, SOFT_TISSUE)
    assert int(out[0, 0]) == 0
    assert int(out[0, 1]) == 255


def test_tile_concatenates_three_panels() -> None:
    slice_hu = np.zeros((16, 20), dtype=np.float32)
    tiled = tile_windows(slice_hu, ABDOMINAL_WINDOWS)
    assert tiled.shape == (16, 20 * 3)


def test_localize_finds_band_blob() -> None:
    volume = np.full((10, 30, 30), -40.0, dtype=np.float32)
    volume[3:7, 3:27, 3:27] = 60.0
    extent = localize_organ(volume, 40.0, 80.0)
    assert extent.found is True
    assert extent.z_start == 3
    assert extent.z_end == 6


def test_localize_handles_absent_organ() -> None:
    volume = np.full((8, 12, 12), -40.0, dtype=np.float32)
    extent = localize_organ(volume, 40.0, 80.0)
    assert extent.found is False
    assert extent.z_start == 0
    assert extent.z_end == 7


def test_slice_selection_counts_and_range() -> None:
    extent = AxialExtent(0, 100, found=True)
    indices = select_slices(extent, SOLID_PERCENTILES)
    assert len(indices) <= len(SOLID_PERCENTILES)
    assert all(0 <= z <= 100 for z in indices)
    assert indices == sorted(indices)


def test_bowel_uses_wider_sampling() -> None:
    extent = AxialExtent(0, 200, found=True)
    indices = select_for_organ(extent, "bowel")
    assert len(BOWEL_PERCENTILES) == 10
    assert len(indices) <= 10
