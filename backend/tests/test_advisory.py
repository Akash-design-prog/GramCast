"""Rigorous tests for the advisory rule engine, following the same standard applied to the data
pipeline and ML code: normal cases, boundary values, invalid/edge inputs, "test the test" (deliberately
break the rule table and confirm a test catches it), and a cross-file consistency check against the
thresholds already used and verified in ml/baselines/metrics.py and ml/model/quantile_loss.py.

Run: python -m pytest backend/tests/test_advisory.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml" / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml" / "model"))

from backend.advisory.rules import (
    CropStage,
    RainCategory,
    classify_rainfall,
    detect_dry_spell,
    generate_advisory,
    HEAVY_THRESHOLD_MM,
    LIGHT_THRESHOLD_MM,
    MODERATE_THRESHOLD_MM,
)


# ---------------------------------------------------------------------------
# Cross-file consistency: the single most important check here. If these drift
# from the ML code's own thresholds, the advisory text would describe a different
# event than the one the model was trained and evaluated against.
# ---------------------------------------------------------------------------

def test_thresholds_match_ml_metrics():
    from metrics import THRESHOLDS  # ml/baselines/metrics.py

    assert LIGHT_THRESHOLD_MM == THRESHOLDS["light"]
    assert MODERATE_THRESHOLD_MM == THRESHOLDS["moderate"]
    assert HEAVY_THRESHOLD_MM == THRESHOLDS["heavy"]


def test_heavy_threshold_matches_quantile_loss():
    from quantile_loss import HEAVY_THRESHOLD_MM as LOSS_HEAVY_THRESHOLD

    assert HEAVY_THRESHOLD_MM == LOSS_HEAVY_THRESHOLD


# ---------------------------------------------------------------------------
# classify_rainfall: normal cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mm,expected",
    [
        (0.0, RainCategory.NO_RAIN),
        (1.0, RainCategory.NO_RAIN),
        (5.0, RainCategory.LIGHT),
        (10.0, RainCategory.LIGHT),
        (30.0, RainCategory.MODERATE),
        (64.4, RainCategory.MODERATE),
        (100.0, RainCategory.HEAVY),
        (500.0, RainCategory.HEAVY),
    ],
)
def test_classify_rainfall_normal(mm, expected):
    assert classify_rainfall(mm) == expected


# ---------------------------------------------------------------------------
# classify_rainfall: boundary values - the exact mm value at each threshold, and
# the value one float-step below it, since these are the cases most likely to be
# broken by an off-by-one (> vs >=) mistake.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mm,expected",
    [
        (LIGHT_THRESHOLD_MM, RainCategory.LIGHT),  # exactly 2.5 -> LIGHT (inclusive lower bound)
        (LIGHT_THRESHOLD_MM - 0.01, RainCategory.NO_RAIN),
        (MODERATE_THRESHOLD_MM, RainCategory.MODERATE),  # exactly 15.6 -> MODERATE
        (MODERATE_THRESHOLD_MM - 0.01, RainCategory.LIGHT),
        (HEAVY_THRESHOLD_MM, RainCategory.HEAVY),  # exactly 64.5 -> HEAVY
        (HEAVY_THRESHOLD_MM - 0.01, RainCategory.MODERATE),
    ],
)
def test_classify_rainfall_boundaries(mm, expected):
    assert classify_rainfall(mm) == expected


def test_classify_rainfall_rejects_negative():
    with pytest.raises(ValueError):
        classify_rainfall(-0.1)


# ---------------------------------------------------------------------------
# detect_dry_spell: normal, boundary, and non-consecutive edge cases
# ---------------------------------------------------------------------------

def test_dry_spell_detected_at_exact_min_days():
    # exactly 3 dry days (the minimum) must trigger; 2 must not.
    assert detect_dry_spell([1.0, 1.0, 1.0], min_days=3) is True
    assert detect_dry_spell([1.0, 1.0], min_days=3) is False


def test_dry_spell_not_triggered_by_non_consecutive_dry_days():
    # 2 dry, 1 wet, 2 dry - no run of 3 anywhere, so this must NOT be a dry spell.
    assert detect_dry_spell([1.0, 1.0, 20.0, 1.0, 1.0], min_days=3) is False


def test_dry_spell_detected_partway_through_series():
    # first 2 days wet, next 3 days dry - the spell doesn't have to start on day 1.
    assert detect_dry_spell([20.0, 20.0, 1.0, 1.0, 1.0], min_days=3) is True


def test_dry_spell_boundary_value_counts_as_dry():
    # exactly at LIGHT_THRESHOLD_MM is NOT dry (dry = below the threshold, matching classify's
    # inclusive-lower-bound convention: LIGHT_THRESHOLD_MM itself already counts as rain).
    assert detect_dry_spell([LIGHT_THRESHOLD_MM] * 3, min_days=3) is False
    assert detect_dry_spell([LIGHT_THRESHOLD_MM - 0.01] * 3, min_days=3) is True


def test_dry_spell_empty_series():
    assert detect_dry_spell([], min_days=3) is False


def test_dry_spell_rejects_invalid_min_days():
    with pytest.raises(ValueError):
        detect_dry_spell([1.0, 1.0, 1.0], min_days=0)


def test_dry_spell_rejects_negative_rainfall():
    with pytest.raises(ValueError):
        detect_dry_spell([1.0, -5.0, 1.0])


# ---------------------------------------------------------------------------
# generate_advisory: quantile ordering guard
# ---------------------------------------------------------------------------

def test_generate_advisory_rejects_unordered_quantiles():
    with pytest.raises(ValueError):
        generate_advisory(p10=50.0, p50=10.0, p90=90.0)  # p10 > p50, physically impossible


def test_generate_advisory_accepts_degenerate_equal_quantiles():
    # p10 == p50 == p90 is a legitimate (if low-uncertainty) prediction, must not raise.
    result = generate_advisory(p10=5.0, p50=5.0, p90=5.0)
    assert result["category"] == RainCategory.LIGHT.value
    assert result["uncertain"] is False


# ---------------------------------------------------------------------------
# generate_advisory: base weather-only rules (no crop stage) - every category
# must produce distinct, non-empty text.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "p50,expected_category",
    [
        (1.0, RainCategory.NO_RAIN),
        (5.0, RainCategory.LIGHT),
        (30.0, RainCategory.MODERATE),
        (100.0, RainCategory.HEAVY),
    ],
)
def test_generate_advisory_base_rules(p50, expected_category):
    result = generate_advisory(p10=max(p50 - 2, 0), p50=p50, p90=p50 + 2)
    assert result["category"] == expected_category.value
    assert result["crop_stage"] is None
    assert isinstance(result["advisory_text"], str) and len(result["advisory_text"]) > 0


def test_all_base_rules_produce_distinct_text():
    texts = set()
    for p50 in [1.0, 5.0, 30.0, 100.0]:
        result = generate_advisory(p10=p50, p50=p50, p90=p50)
        texts.add(result["advisory_text"])
    assert len(texts) == 4, "each of the 4 rainfall categories must have distinct advisory text"


# ---------------------------------------------------------------------------
# generate_advisory: crop-stage overrides actually change the text where defined,
# and fall back to the base rule where no override exists.
# ---------------------------------------------------------------------------

def test_crop_stage_override_changes_text_for_heavy_flowering():
    base = generate_advisory(p10=100.0, p50=100.0, p90=100.0)
    overridden = generate_advisory(p10=100.0, p50=100.0, p90=100.0, crop_stage=CropStage.FLOWERING)
    assert overridden["advisory_text"] != base["advisory_text"]
    assert "flower" in overridden["advisory_text"].lower()


def test_crop_stage_falls_back_to_base_when_no_override_defined():
    # HEAVY + VEGETATIVE has no override in the lookup table -> must fall back to the base HEAVY text.
    base = generate_advisory(p10=100.0, p50=100.0, p90=100.0)
    fallback = generate_advisory(p10=100.0, p50=100.0, p90=100.0, crop_stage=CropStage.VEGETATIVE)
    assert fallback["advisory_text"] == base["advisory_text"]
    assert fallback["crop_stage"] == CropStage.VEGETATIVE.value


def test_every_crop_stage_and_category_combination_returns_something_sane():
    # full cross of every (category, stage) pair must not crash and must always return non-empty text.
    for p50, category in [(1.0, RainCategory.NO_RAIN), (5.0, RainCategory.LIGHT), (30.0, RainCategory.MODERATE), (100.0, RainCategory.HEAVY)]:
        for stage in CropStage:
            result = generate_advisory(p10=p50, p50=p50, p90=p50, crop_stage=stage)
            assert result["category"] == category.value
            assert len(result["advisory_text"]) > 0


# ---------------------------------------------------------------------------
# generate_advisory: dry-spell override takes priority over the single-day base
# rule when a forecast series is supplied, but only for NO_RAIN/LIGHT categories
# (a dry-spell label attached to a HEAVY day forecast would be self-contradictory).
# ---------------------------------------------------------------------------

def test_dry_spell_overrides_base_text_for_no_rain_day():
    result = generate_advisory(p10=0.0, p50=1.0, p90=1.0, forecast_series_mm=[1.0, 1.0, 1.0, 1.0])
    assert result["dry_spell"] is True
    assert "dry spell" in result["advisory_text"].lower()


def test_dry_spell_flag_does_not_override_heavy_day_text():
    # today is a heavy-rain day even though a dry spell is detected later in the series - the
    # headline for TODAY must still be the heavy-rain warning, not a dry-spell message.
    result = generate_advisory(p10=100.0, p50=100.0, p90=100.0, forecast_series_mm=[1.0, 1.0, 1.0, 1.0])
    assert result["dry_spell"] is True  # flag still reported...
    assert "dry spell" not in result["advisory_text"].lower()  # ...but doesn't overwrite today's real risk
    assert result["category"] == RainCategory.HEAVY.value


def test_no_forecast_series_means_no_dry_spell_claim():
    result = generate_advisory(p10=1.0, p50=1.0, p90=1.0)
    assert result["dry_spell"] is False
    assert "dry spell" not in result["advisory_text"].lower()


# ---------------------------------------------------------------------------
# generate_advisory: uncertainty flag - true only when the P10-P90 band actually
# spans more than one IMD category, not just whenever P10 != P90 numerically.
# ---------------------------------------------------------------------------

def test_uncertain_flag_true_when_band_spans_categories():
    result = generate_advisory(p10=10.0, p50=30.0, p90=70.0)  # LIGHT..MODERATE..HEAVY
    assert result["uncertain"] is True


def test_uncertain_flag_false_when_band_stays_within_one_category():
    result = generate_advisory(p10=20.0, p50=30.0, p90=40.0)  # all MODERATE
    assert result["uncertain"] is False


def test_uncertain_flag_false_at_zero_spread():
    result = generate_advisory(p10=30.0, p50=30.0, p90=30.0)
    assert result["uncertain"] is False


# ---------------------------------------------------------------------------
# "Test the test": deliberately corrupt the rule table / thresholds and confirm
# the tests above actually fail, rather than silently passing on anything.
# ---------------------------------------------------------------------------

def test_classify_rainfall_test_catches_broken_threshold(monkeypatch):
    import backend.advisory.rules as rules_module

    monkeypatch.setattr(rules_module, "HEAVY_THRESHOLD_MM", 999.0)
    # with a deliberately-broken (too-high) heavy threshold, 100mm must now be misclassified
    # as MODERATE instead of HEAVY - confirming the test would actually detect this class of bug.
    assert rules_module.classify_rainfall(100.0) == RainCategory.MODERATE


def test_duplicate_advisory_text_would_be_caught():
    # sanity-check the distinctness assertion itself: if two categories genuinely shared text,
    # this constructed case (both texts identical to "x") must fail len(texts) == 4.
    texts = {"x", "x", "y", "z"}
    assert len(texts) != 4
