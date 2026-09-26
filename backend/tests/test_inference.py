"""Tests for backend/inference.py against the real dataset and a real checkpoint - this module has no
meaningful behavior to test with synthetic data (it IS the data-loading + model-loading glue), so these
are integration tests, skipped if the real files aren't present (e.g. a fresh checkout without the
training-pairs data downloaded).

Run: python -m pytest backend/tests/test_inference.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DEFAULT_CHECKPOINT, GramCastInference, TRAINING_PAIRS_DIR

pytestmark = pytest.mark.skipif(
    not (TRAINING_PAIRS_DIR / "era5_humidity_wind.npz").exists() or not DEFAULT_CHECKPOINT.exists(),
    reason="real training-pairs dataset or default checkpoint not present",
)


@pytest.fixture(scope="module")
def inference():
    return GramCastInference()


def test_loads_and_has_dates(inference):
    dates = inference.available_dates()
    assert len(dates) == 5582
    assert "2023-07-15" in dates


def test_predict_day_shapes_and_monotonicity(inference):
    result = inference.predict_day("2023-07-15")
    for key in ["p10", "p50", "p90", "block_value"]:
        assert result[key].shape == (35, 50)
        assert not np.isnan(result[key]).any()
    assert (result["p10"] <= result["p50"] + 1e-4).all()  # tiny float slack, not a real violation
    assert (result["p50"] <= result["p90"] + 1e-4).all()


def test_predict_day_unknown_date_raises():
    inf = GramCastInference()
    with pytest.raises(ValueError, match="not found"):
        inf.predict_day("1900-01-01")


def test_predict_day_block_value_is_flat_within_blocks(inference):
    """block_value must be uniform within each 5x5 coarse block - that's the whole point of contrasting
    it against the sharpened panchayat-level output. A non-flat block_value would mean the 'block vs
    panchayat' demo comparison is comparing two sharpened things, not the intended flat-vs-sharp."""
    result = inference.predict_day("2023-07-15")
    block = result["block_value"]
    for r0 in range(0, 35, 5):
        for c0 in range(0, 50, 5):
            patch = block[r0 : r0 + 5, c0 : c0 + 5]
            assert np.allclose(patch, patch[0, 0]), f"block at ({r0},{c0}) is not flat"


def test_stress_100_random_dates_no_crashes(inference):
    """Full-scale stress test, same discipline as the data pipeline/ML stress tests - 100 random dates
    spanning the whole dataset (not just one hand-picked day), checking shape/NaN/monotonicity every time."""
    rng = np.random.default_rng(42)
    sample_dates = rng.choice(inference.available_dates(), size=100, replace=False)
    for d in sample_dates:
        result = inference.predict_day(d)
        for key in ["p10", "p50", "p90"]:
            assert result[key].shape == (35, 50)
            assert not np.isnan(result[key]).any()
        assert (result["p10"] <= result["p50"] + 1e-4).all()
        assert (result["p50"] <= result["p90"] + 1e-4).all()


def test_different_dates_give_different_predictions(inference):
    """Sanity check that the model is actually conditioning on the day's real input, not returning a
    cached/constant output regardless of date."""
    a = inference.predict_day("2023-07-15")
    b = inference.predict_day("2023-08-01")
    assert not np.allclose(a["p50"], b["p50"]), "two different real days produced identical predictions"
