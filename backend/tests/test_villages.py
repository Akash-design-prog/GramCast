"""Tests for backend/villages.py - the point-in-polygon lookup and the panchayat-value zonal-stats
wiring. Real data (village boundaries), since there's no reasonable synthetic substitute for 2,003 real
polygons - skipped if the boundaries file isn't present.

Run: python -m pytest backend/tests/test_villages.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from villages import BOUNDARIES_PATH, VillageIndex, panchayat_value

pytestmark = pytest.mark.skipif(not BOUNDARIES_PATH.exists(), reason="village boundaries file not present")


@pytest.fixture(scope="module")
def village_index():
    return VillageIndex()


def test_find_known_point_matches_within_not_fallback(village_index):
    """Regression test for the real bug found during development: predicate='contains' was backwards
    (a point can never 'contain' a polygon) and silently fell through to the nearest-centroid fallback
    for every single lookup, even ones squarely inside a village. Lonavala's own municipal-council
    polygon must be found via true point-in-polygon containment, not the fallback."""
    result = village_index.find(18.75, 73.41)
    assert result["matched_by"] == "within"
    assert "Lonavala" in result["name"]


def test_find_pune_city(village_index):
    result = village_index.find(18.52, 73.85)
    assert result["matched_by"] == "within"
    assert "Pune" in result["name"]


def test_find_far_outside_district_uses_fallback(village_index):
    result = village_index.find(25.0, 80.0)  # nowhere near Pune district
    assert result["matched_by"] == "nearest_centroid_fallback"
    assert result["name"] is not None  # still returns a real answer, not an error


def test_panchayat_value_monotonic_with_synthetic_grid(village_index):
    """A synthetic all-constant grid must produce that same constant for a real village's mean/p10/p50/p90/max."""
    grid = np.full((35, 50), 7.5, dtype="float32")
    result = panchayat_value(grid, "Pune (M Corp.)", "Pune City", "test-date")
    assert result["mean_mm"] == pytest.approx(7.5)
    assert result["p10_mm"] == pytest.approx(7.5)
    assert result["p90_mm"] == pytest.approx(7.5)
    assert result["max_mm"] == pytest.approx(7.5)


def test_stress_100_random_real_village_centroids_all_match_within(village_index):
    """Full-scale stress test: every village's own centroid must resolve via true point-in-polygon
    containment ('within'), not the fallback - across 100 random real villages, not just 2 hand-picked
    ones. A regression here would mean the predicate bug (or something like it) came back."""
    sample = village_index.villages.sample(n=100, random_state=7)
    for _, row in sample.iterrows():
        centroid = row.geometry.centroid
        result = village_index.find(centroid.y, centroid.x)
        assert result["matched_by"] == "within", f"{row['NAME']} centroid did not match via containment"


def test_panchayat_value_unknown_village_raises():
    grid = np.zeros((35, 50), dtype="float32")
    with pytest.raises(ValueError, match="not found"):
        panchayat_value(grid, "Not A Real Village", "Nowhere", "test-date")
