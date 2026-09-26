"""Unit tests for resample_era5_to_training_grid.py's helper functions, in isolation with synthetic
data - the real-data integration is covered separately by full_pipeline_qc.py's structural orientation
check (run against the actual 1981 ERA5 file).

Run: python -m pytest data_pipeline/align/tests/test_resample_era5.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resample_era5_to_training_grid import (
    _fill_nan_nearest,
    _pad_source_for_bilinear,
    get_era5_source_transform,
    reproject_to_chirps,
)


# ---------------------------------------------------------------------------
# get_era5_source_transform: orientation - the exact bug class already found once for terrain
# ---------------------------------------------------------------------------

def test_source_transform_north_edge_for_descending_lat():
    """ERA5's real convention: lat[0] is the NORTHERNMOST value (descending)."""
    lat = np.array([19.6, 19.5, 19.4])  # descending
    lon = np.array([73.0, 73.1, 73.2])
    transform = get_era5_source_transform(lat, lon)
    # north edge should be lat[0] + half a cell = 19.6 + 0.05 = 19.65
    assert transform.f == pytest.approx(19.65)


def test_source_transform_north_edge_for_ascending_lat():
    """If ever given ascending lat (not ERA5's actual convention, but the function shouldn't
    silently mishandle it) - north edge should still be the larger value + half a cell."""
    lat = np.array([17.7, 17.8, 17.9])  # ascending
    lon = np.array([73.0, 73.1, 73.2])
    transform = get_era5_source_transform(lat, lon)
    assert transform.f == pytest.approx(17.95)


# ---------------------------------------------------------------------------
# _fill_nan_nearest
# ---------------------------------------------------------------------------

def test_fill_nan_nearest_no_nan_is_noop():
    arr = np.arange(12, dtype="float32").reshape(3, 4)
    result = _fill_nan_nearest(arr)
    assert np.array_equal(result, arr)


def test_fill_nan_nearest_fills_with_nearest_valid_value():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]], dtype="float32")
    result = _fill_nan_nearest(arr)
    assert not np.isnan(result).any()
    # the centre NaN's nearest neighbours are all equidistant - just confirm it picked SOME real value, not 0/garbage
    assert result[1, 1] in [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]


def test_fill_nan_nearest_edge_cell():
    """Corner NaN (only diagonal neighbours) - the real ERA5 case is a corner/edge cluster, not an
    isolated centre cell, so test that shape specifically."""
    arr = np.array([[np.nan, np.nan, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype="float32")
    result = _fill_nan_nearest(arr)
    assert not np.isnan(result).any()
    assert result[0, 0] == pytest.approx(5.0)  # nearest valid cell to (0,0) is (1,1)=5.0, distance sqrt(2) vs others further


def test_fill_nan_nearest_multiple_disconnected_clusters():
    """Two separate NaN islands, not just one - each must fill from ITS nearest valid neighbour, not
    accidentally bleed into the other cluster's fill value."""
    arr = np.array(
        [
            [1.0, np.nan, 1.0, 1.0, 9.0],
            [1.0, 1.0, 1.0, np.nan, 9.0],
            [1.0, 1.0, 1.0, 1.0, 9.0],
        ],
        dtype="float32",
    )
    result = _fill_nan_nearest(arr)
    assert not np.isnan(result).any()
    assert result[0, 1] == pytest.approx(1.0)  # surrounded by 1.0s, nowhere near the 9.0 column
    assert result[1, 3] == pytest.approx(9.0)  # (1,3)'s nearest valid neighbour is (0,4)=9.0 or (1,4)=9.0, both distance ~1.4/1.0, closer than any 1.0


def test_fill_nan_nearest_ring_shape():
    """A ring of NaN surrounding a single valid centre cell - nearest-neighbour fill must reach OUT from
    the ring's own outer boundary, not get confused by the enclosed valid cell being 'inside'."""
    arr = np.full((5, 5), np.nan, dtype="float32")
    arr[2, 2] = 100.0  # lone valid cell, surrounded by the NaN ring
    arr[0, :] = 1.0  # valid border on all 4 sides
    arr[-1, :] = 1.0
    arr[:, 0] = 1.0
    arr[:, -1] = 1.0
    result = _fill_nan_nearest(arr)
    assert not np.isnan(result).any()
    assert result[2, 2] == 100.0  # the one originally-valid cell must be untouched
    # a ring cell adjacent to the border (e.g. (1,2)) should fill from the nearby border value (1.0),
    # not reach all the way to the distant centre (100.0)
    assert result[1, 2] == pytest.approx(1.0)


def test_fill_nan_nearest_mostly_nan_still_works():
    """Only one valid cell in a large array - every NaN cell's nearest (only) neighbour is that one cell."""
    arr = np.full((10, 10), np.nan, dtype="float32")
    arr[0, 0] = 42.0
    result = _fill_nan_nearest(arr)
    assert not np.isnan(result).any()
    assert np.all(result == 42.0)


def test_fill_nan_nearest_all_nan_raises():
    arr = np.full((3, 3), np.nan, dtype="float32")
    with pytest.raises(ValueError, match="entire source array is NaN"):
        _fill_nan_nearest(arr)


def test_fill_nan_nearest_preserves_non_nan_values_exactly():
    arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype="float32")
    result = _fill_nan_nearest(arr)
    assert result[0, 0] == 1.0
    assert result[1, 0] == 3.0
    assert result[1, 1] == 4.0


# ---------------------------------------------------------------------------
# _pad_source_for_bilinear: shape and geometric consistency
# ---------------------------------------------------------------------------

def test_pad_source_increases_shape_by_two():
    arr = np.ones((5, 7), dtype="float32")
    transform = rasterio.transform.from_origin(73.0, 19.6, 0.1, 0.1)
    padded, padded_transform = _pad_source_for_bilinear(arr, transform)
    assert padded.shape == (7, 9)


def test_pad_source_shifts_origin_by_one_cell():
    res = 0.1
    transform = rasterio.transform.from_origin(73.0, 19.6, res, res)
    arr = np.ones((5, 7), dtype="float32")
    _, padded_transform = _pad_source_for_bilinear(arr, transform)
    assert padded_transform.c == pytest.approx(73.0 - res)  # west shifted further west
    assert padded_transform.f == pytest.approx(19.6 + res)  # north shifted further north


def test_pad_source_edge_replicates_values():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    transform = rasterio.transform.from_origin(73.0, 19.6, 0.1, 0.1)
    padded, _ = _pad_source_for_bilinear(arr, transform)
    # corner of padded array should replicate the nearest original corner value
    assert padded[0, 0] == 1.0
    assert padded[-1, -1] == 4.0


# ---------------------------------------------------------------------------
# reproject_to_chirps: end-to-end on synthetic data - confirms no NaN leaks through for an
# interior-covering destination grid, and that NaN source cells get filled before reprojection
# rather than propagating (this is the exact real bug found and fixed against the actual ERA5 data).
# ---------------------------------------------------------------------------

def test_reproject_fills_source_nan_before_projecting():
    src = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, np.nan, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]], dtype="float32")
    src_transform = rasterio.transform.from_origin(73.0, 19.7, 0.1, 0.1)
    dst_transform = rasterio.transform.from_origin(73.05, 19.65, 0.05, 0.05)
    result = reproject_to_chirps(src, src_transform, dst_transform, (4, 6))
    assert not np.isnan(result).any(), "NaN source cell should have been filled before reprojection, not propagated"
    # since every source cell is ~10 (after filling the one NaN with a neighbour that's also 10), output should be uniformly ~10
    assert np.allclose(result, 10.0, atol=0.5)


def test_reproject_no_nan_at_domain_interior():
    """Regression test for the real bug: destination pixels well inside the source domain must never
    come back NaN, even before any padding/filling - this specific synthetic case reproduces the
    original failure mode (values present everywhere, checking pure geometry, not the NaN-fill fix)."""
    rng = np.random.default_rng(42)
    src = rng.uniform(280, 300, size=(20, 26)).astype("float32")
    src_transform = rasterio.transform.from_origin(72.95, 19.65, 0.1, 0.1)
    dst_transform = rasterio.transform.from_origin(72.9998, 19.5998, 0.05, 0.05)
    result = reproject_to_chirps(src, src_transform, dst_transform, (38, 50))
    assert not np.isnan(result).any()
