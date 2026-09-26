"""Tests for backend/villages.py - the point-in-polygon lookup and the panchayat-value zonal-stats
wiring. Real data (village boundaries), since there's no reasonable synthetic substitute for 2,003 real
polygons - skipped if the boundaries file isn't present.

Run: python -m pytest backend/tests/test_villages.py -v
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from villages import BOUNDARIES_PATH, VillageIndex, bulk_village_stats, panchayat_value, panchayat_value_by_index

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


def test_find_row_index_points_back_at_the_same_village(village_index):
    """row_index must be a real positional index into village_index.villages that actually resolves back
    to the same village find() itself just returned - this is the whole basis for the O(1) lookup path
    (panchayat_value_by_index), so a wrong row_index would silently return some OTHER village's rainfall."""
    result = village_index.find(18.75, 73.41)
    row = village_index.villages.iloc[result["row_index"]]
    assert row["NAME"] == result["name"]
    assert row["SUB_DIST"] == result["sub_district"]


def test_find_row_index_stress_100_random_real_villages(village_index):
    """Full-scale check: for 100 random real villages, the row_index returned by find() at each
    village's own centroid must point back at that exact village, not a neighbor sharing similar
    geometry-index proximity."""
    sample = village_index.villages.sample(n=100, random_state=13)
    for _, expected_row in sample.iterrows():
        centroid = expected_row.geometry.centroid
        result = village_index.find(centroid.y, centroid.x)
        found_row = village_index.villages.iloc[result["row_index"]]
        assert found_row["NAME"] == expected_row["NAME"]
        assert found_row["SUB_DIST"] == expected_row["SUB_DIST"]


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


# ---------------------------------------------------------------------------
# panchayat_value_by_index - the O(1) single-village lookup that replaced panchayat_value() on the
# /forecast hot path (which was O(all 2003 villages) per call, 4x per request, 3x more for the
# frontend's 3-day sparkline - ~24,000 whole-district stat computations to answer one village's forecast)
# ---------------------------------------------------------------------------

def test_panchayat_value_by_index_matches_panchayat_value_exactly(village_index):
    """The fast path must return bit-for-bit the same answer as the slow, already-verified path for a
    real village, on a real non-constant grid - not just on trivial constant-grid synthetic data."""
    rng = np.random.default_rng(21)
    grid = rng.uniform(0, 60, size=(35, 50)).astype("float32")

    row = village_index.villages.iloc[500]
    slow = panchayat_value(grid, row["NAME"], row["SUB_DIST"], "test-date")
    fast = panchayat_value_by_index(grid, 500)

    for key in ("mean_mm", "p10_mm", "p50_mm", "p90_mm", "max_mm"):
        assert fast[key] == pytest.approx(slow[key])


def test_panchayat_value_by_index_stress_100_random_villages_match_slow_path(village_index):
    """Full-scale equivalence check across 100 random real villages, not one hand-picked row - a mask
    misalignment (off-by-one row order drift, wrong village_index vs zonal_stats row correspondence)
    would only show up on some rows, not necessarily the first one tried.

    Excludes villages whose NAME+SUB_DIST is duplicated elsewhere in the file (134 rows are - e.g. two
    entirely separate real villages are both 'Pimpri' in 'Mulshi' taluka, rows 646 and 1924). For those,
    panchayat_value()'s name-based lookup is itself ambiguous (grabs whichever duplicate comes first in
    the DataFrame) and isn't a valid ground truth to compare the row-index-based fast path against - this
    was caught here directly (a real mismatch on row 1924, traced to exactly this ambiguity, not a bug in
    the fast path)."""
    rng = np.random.default_rng(23)
    grid = rng.uniform(0, 60, size=(35, 50)).astype("float32")
    unambiguous = village_index.villages[~village_index.villages.duplicated(subset=["NAME", "SUB_DIST"], keep=False)]

    for row_index in rng.choice(unambiguous.index.to_numpy(), size=100, replace=False):
        row = village_index.villages.iloc[row_index]
        slow = panchayat_value(grid, row["NAME"], row["SUB_DIST"], "test-date")
        fast = panchayat_value_by_index(grid, int(row_index))
        assert fast["mean_mm"] == pytest.approx(slow["mean_mm"]), f"mismatch at row {row_index} ({row['NAME']})"
        assert fast["p50_mm"] == pytest.approx(slow["p50_mm"]), f"mismatch at row {row_index} ({row['NAME']})"


def test_panchayat_value_by_index_distinguishes_real_duplicate_named_villages(village_index):
    """Regression test for the ambiguity found above: two real, distinct villages ('Pimpri', rows 646
    and 1924, both in Mulshi taluka) must resolve to DIFFERENT values on a non-constant grid when looked
    up by their own row_index - proving the fast path actually disambiguates what the name-based path
    cannot, not just that it's faster."""
    rng = np.random.default_rng(31)
    grid = rng.uniform(0, 60, size=(35, 50)).astype("float32")
    row_646 = village_index.villages.iloc[646]
    row_1924 = village_index.villages.iloc[1924]
    assert row_646["NAME"] == row_1924["NAME"] == "Pimpri"
    assert row_646["SUB_DIST"] == row_1924["SUB_DIST"] == "Mulshi"

    value_646 = panchayat_value_by_index(grid, 646)
    value_1924 = panchayat_value_by_index(grid, 1924)
    assert value_646["mean_mm"] != pytest.approx(value_1924["mean_mm"]), (
        "two distinct real villages sharing a name+sub_district resolved to the same value - "
        "row_index isn't actually disambiguating them"
    )


def test_panchayat_value_by_index_first_and_last_row_do_not_crash(village_index):
    """Edge cases: row 0 and the last row (off-by-one boundary conditions for the mask array index)."""
    grid = np.full((35, 50), 10.0, dtype="float32")
    n = len(village_index.villages)
    assert panchayat_value_by_index(grid, 0)["mean_mm"] == pytest.approx(10.0)
    assert panchayat_value_by_index(grid, n - 1)["mean_mm"] == pytest.approx(10.0)


def test_panchayat_value_by_index_all_nan_grid_raises():
    """A village whose only pixels are NaN (e.g. a real coverage gap) must raise a clear error, not
    silently return a NaN or crash on an empty-array percentile call."""
    grid = np.full((35, 50), np.nan, dtype="float32")
    with pytest.raises(ValueError, match="no pixel coverage"):
        panchayat_value_by_index(grid, 0)


def test_panchayat_value_by_index_is_independent_of_village_count(village_index):
    """The actual complexity claim, proven by direct timing rather than asserted by inspection: looking
    up one village must not take meaningfully longer than looking up another, and must be dramatically
    faster than compute_village_stats()'s O(all 2003 villages) path for the same grid. This is what
    'O(1) instead of O(n)' means in practice for a /forecast request."""
    from zonal_stats import compute_village_stats

    rng = np.random.default_rng(29)
    grid = rng.uniform(0, 60, size=(35, 50)).astype("float32")

    # Warm the mask cache first so the timing below measures the actual per-call cost, not one-time setup.
    panchayat_value_by_index(grid, 0)
    compute_village_stats(grid, "warmup")

    t0 = time.perf_counter()
    for row_index in range(200):
        panchayat_value_by_index(grid, row_index)
    fast_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_village_stats(grid, "test-date")
    full_scan_elapsed = time.perf_counter() - t0

    per_call_fast = fast_elapsed / 200
    # Looking up 200 individual villages the fast way must still be faster than ONE full-district scan -
    # a generous bound (the real speedup is roughly 2003x per call), chosen to be robust to machine noise
    # while still failing hard if panchayat_value_by_index regressed back to an O(n) implementation.
    assert fast_elapsed < full_scan_elapsed, (
        f"200 single-village lookups ({fast_elapsed:.4f}s) should be faster than one full 2003-village "
        f"scan ({full_scan_elapsed:.4f}s) - per-call: {per_call_fast:.6f}s"
    )


# ---------------------------------------------------------------------------
# bulk_village_stats (the /forecast/map endpoint's data source)
# ---------------------------------------------------------------------------

def _synthetic_pred(block_val, p10_val, p50_val, p90_val):
    return {
        "block_value": np.full((35, 50), block_val, dtype="float32"),
        "p10": np.full((35, 50), p10_val, dtype="float32"),
        "p50": np.full((35, 50), p50_val, dtype="float32"),
        "p90": np.full((35, 50), p90_val, dtype="float32"),
    }


def test_bulk_village_stats_covers_every_village(village_index):
    """Row count must match the full boundaries file exactly - a lossy merge (duplicate keys, a left
    join dropping rows) would silently shrink the district for every consumer of this endpoint."""
    pred = _synthetic_pred(1.0, 2.0, 3.0, 4.0)
    result = bulk_village_stats(pred)
    assert len(result) == len(village_index.villages)


def test_bulk_village_stats_survives_real_duplicate_name_subdist_pairs(village_index):
    """Regression test for the real bug found during development: NAME+SUB_DIST is NOT a unique key in
    the real boundaries file (98 duplicate pairs, e.g. several 'Pune (M Corp.)' polygons sharing one
    CEN_2001 code) - a key-based merge on that pair silently exploded 2003 rows into 52.8 million via
    an accidental many-to-many join. Confirms the real file still has duplicates (so this test would
    actually catch a regression back to key-based merging) and that the row count stays exact anyway."""
    dup_count = village_index.villages.duplicated(subset=["NAME", "SUB_DIST"]).sum()
    assert dup_count > 0, "boundaries file no longer has duplicate NAME+SUB_DIST pairs - test's premise changed"

    pred = _synthetic_pred(1.0, 2.0, 3.0, 4.0)
    result = bulk_village_stats(pred)
    assert len(result) == len(village_index.villages)


def test_bulk_village_stats_synthetic_constant_grid(village_index):
    """A constant-valued grid must produce that same constant for every village's every column."""
    pred = _synthetic_pred(1.0, 2.0, 3.0, 4.0)
    result = bulk_village_stats(pred)
    assert result["block_mm"].dropna().unique() == pytest.approx([1.0])
    assert result["p10_mm"].dropna().unique() == pytest.approx([2.0])
    assert result["p50_mm"].dropna().unique() == pytest.approx([3.0])
    assert result["p90_mm"].dropna().unique() == pytest.approx([4.0])


def test_bulk_village_stats_quantiles_stay_ordered_with_real_checkpoint_shaped_noise(village_index):
    """Non-constant grids (closer to real model output) must still preserve p10 <= p50 <= p90 per
    village once aggregated - a monotonicity break here would mean the merge mismatched rows across
    the four separate zonal-stats calls (block/p10/p50/p90 each re-reads the boundaries file)."""
    rng = np.random.default_rng(11)
    base = rng.uniform(0, 40, size=(35, 50)).astype("float32")
    pred = {
        "block_value": base,
        "p10": base * 0.6,
        "p50": base,
        "p90": base * 1.4,
    }
    result = bulk_village_stats(pred)
    covered = result.dropna(subset=["p10_mm", "p50_mm", "p90_mm"])
    assert len(covered) > 0
    assert (covered["p10_mm"] <= covered["p50_mm"] + 1e-6).all()
    assert (covered["p50_mm"] <= covered["p90_mm"] + 1e-6).all()


def test_bulk_village_stats_geometry_present_for_every_row(village_index):
    """Every row must carry real geometry - the GeoJSON endpoint has nothing to draw otherwise."""
    pred = _synthetic_pred(1.0, 2.0, 3.0, 4.0)
    result = bulk_village_stats(pred)
    assert result.geometry.notna().all()
    assert (result.geometry.area > 0).all()
