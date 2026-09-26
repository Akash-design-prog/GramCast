"""Tests for imd_gridded_validation.py's grid-transform and aggregation logic - the orientation-sensitive
parts, since this is exactly the bug class (south/north row order) that has bitten this project twice
already (terrain, then nearly again during the ERA5 integration).

Run: python -m pytest ml/model/tests/test_imd_gridded_validation.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from imd_gridded_validation import (
    aggregate_to_imd_grid,
    get_fine_grid_south_ascending_transform,
    get_imd_south_ascending_transform,
)


def test_fine_grid_transform_row0_is_south():
    transform = get_fine_grid_south_ascending_transform()
    _, lat_row0 = rasterio.transform.xy(transform, 0, 0)
    _, lat_row34 = rasterio.transform.xy(transform, 34, 0)
    assert lat_row0 < lat_row34, "row 0 must be the southernmost row (lower latitude)"
    assert lat_row0 == pytest.approx(17.725, abs=0.01)  # known CHIRPS south edge cell centre


def test_imd_transform_row0_is_south():
    lat = np.array([17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5])  # real IMD lat array
    lon = np.array([73.0, 73.25, 73.5, 73.75, 74.0, 74.25, 74.5, 74.75, 75.0, 75.25, 75.5])
    transform, h, w = get_imd_south_ascending_transform(lat, lon)
    assert h == 8 and w == 11
    _, lat_row0 = rasterio.transform.xy(transform, 0, 0)
    _, lat_row7 = rasterio.transform.xy(transform, 7, 0)
    assert lat_row0 == pytest.approx(17.75)
    assert lat_row7 == pytest.approx(19.5)


def test_aggregate_preserves_south_to_north_gradient():
    """The exact regression test for the real diagnostic run during development: a synthetic
    south-to-north gradient (low values at row 0, high at row 34) must aggregate to low values at the
    destination's southernmost row and high values at its northernmost row - not flipped."""
    src_transform = get_fine_grid_south_ascending_transform()
    lat = np.array([17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5])
    lon = np.array([73.0, 73.25, 73.5, 73.75, 74.0, 74.25, 74.5, 74.75, 75.0, 75.25, 75.5])
    dst_transform, dst_h, dst_w = get_imd_south_ascending_transform(lat, lon)

    fine = np.zeros((35, 50), dtype="float32")
    for r in range(35):
        fine[r, :] = r  # south (row 0) = 0 (lowest), north (row 34) = 34 (highest)

    result = aggregate_to_imd_grid(fine, src_transform, dst_transform, (dst_h, dst_w))
    assert result[0].mean() < result[-1].mean(), "aggregation must preserve south=low, north=high - got the opposite (flip bug)"
    assert np.all(np.diff(result.mean(axis=1)) > 0), "aggregated rows must be monotonically increasing south to north"


def test_aggregate_preserves_east_west_gradient():
    """Same idea but for the column (longitude) axis, since a west/east flip is a different bug class
    that a north/south-only check wouldn't catch."""
    src_transform = get_fine_grid_south_ascending_transform()
    lat = np.array([17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5])
    lon = np.array([73.0, 73.25, 73.5, 73.75, 74.0, 74.25, 74.5, 74.75, 75.0, 75.25, 75.5])
    dst_transform, dst_h, dst_w = get_imd_south_ascending_transform(lat, lon)

    fine = np.zeros((35, 50), dtype="float32")
    for c in range(50):
        fine[:, c] = c  # west (col 0) = 0 (lowest), east (col 49) = 49 (highest)

    result = aggregate_to_imd_grid(fine, src_transform, dst_transform, (dst_h, dst_w))
    assert result[:, 0].mean() < result[:, -1].mean(), "aggregation must preserve west=low, east=high"
