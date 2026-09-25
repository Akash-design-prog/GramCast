"""Advisory rule engine: turns a rainfall forecast (with uncertainty) into a short farmer-facing
action, per the project guide's section 8 ("Rule-based advisory text tied to crop stage, not just
raw numbers") and section 8's own scope note: keep it to 3-5 weather rules, refine with a *small*
crop-stage lookup table, do not build a crop-specific system for the pilot.

IMD rainfall category thresholds (2.5 / 15.6 / 64.5 mm/day) MUST match ml/baselines/metrics.py's
THRESHOLDS and ml/model/quantile_loss.py's HEAVY_THRESHOLD_MM - the cross-file consistency test in
backend/tests/test_advisory.py checks this directly, since a silent drift here would mean the
advisory text disagrees with what the model itself was trained/evaluated to detect.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence

# Mirrors ml/baselines/metrics.py THRESHOLDS and ml/model/quantile_loss.py HEAVY_THRESHOLD_MM.
LIGHT_THRESHOLD_MM = 2.5
MODERATE_THRESHOLD_MM = 15.6
HEAVY_THRESHOLD_MM = 64.5

DRY_SPELL_MIN_DAYS = 3  # consecutive forecast days below LIGHT_THRESHOLD_MM


class RainCategory(str, Enum):
    NO_RAIN = "no_rain"  # < 2.5 mm/day
    LIGHT = "light"  # 2.5 - 15.5 mm/day
    MODERATE = "moderate"  # 15.6 - 64.4 mm/day
    HEAVY = "heavy"  # >= 64.5 mm/day


class CropStage(str, Enum):
    SOWING = "sowing"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    HARVEST = "harvest"


def classify_rainfall(mm: float) -> RainCategory:
    """IMD-threshold classification. Boundaries are inclusive at the lower edge (>=), matching the
    >= semantics ml/baselines/metrics.py already uses for event detection."""
    if mm < 0:
        raise ValueError(f"rainfall cannot be negative, got {mm}")
    if mm >= HEAVY_THRESHOLD_MM:
        return RainCategory.HEAVY
    if mm >= MODERATE_THRESHOLD_MM:
        return RainCategory.MODERATE
    if mm >= LIGHT_THRESHOLD_MM:
        return RainCategory.LIGHT
    return RainCategory.NO_RAIN


# The 3-5 base weather rules (guide section 8's own example table), keyed by category.
_BASE_ADVISORY = {
    RainCategory.HEAVY: "Heavy rain likely. Delay spraying and fertiliser application; clear field drainage; secure harvested produce.",
    RainCategory.MODERATE: "Moderate rain likely. Good window for sowing or transplanting where soil allows.",
    RainCategory.LIGHT: "Light rain likely. Generally safe for field work; check conditions before spraying.",
    RainCategory.NO_RAIN: "Little or no rain expected today. Normal field operations; irrigate if soil is dry.",
}

_DRY_SPELL_ADVISORY = "Dry spell expected over the next several days. Plan irrigation; good time for harvesting or drying."

# Small crop-stage refinement table (guide: "a small lookup table ... is enough to demonstrate the
# idea", explicitly not a full crop-specific system). Only overrides where the action genuinely
# changes with crop stage; every other (category, stage) combo falls back to _BASE_ADVISORY.
_CROP_STAGE_OVERRIDES = {
    (RainCategory.HEAVY, CropStage.FLOWERING): "Heavy rain likely during flowering - risk of flower/pod drop. Delay spraying; provide drainage to protect the crop.",
    (RainCategory.HEAVY, CropStage.HARVEST): "Heavy rain likely with crop ready for harvest - prioritise harvesting and securing produce before rain arrives if there is lead time; otherwise protect stored produce from moisture.",
    (RainCategory.MODERATE, CropStage.SOWING): "Moderate rain likely - favourable moisture window for sowing or transplanting.",
    (RainCategory.NO_RAIN, CropStage.SOWING): "Little or no rain expected - soil may be too dry for sowing; irrigate first or wait for rain.",
    (RainCategory.NO_RAIN, CropStage.HARVEST): "Dry conditions expected - good window for harvesting and drying the crop.",
}


def detect_dry_spell(daily_forecast_mm: Sequence[float], min_days: int = DRY_SPELL_MIN_DAYS) -> bool:
    """True if there is any run of >= min_days consecutive days below LIGHT_THRESHOLD_MM anywhere
    in the forecast series. Not just "are all days dry" - a dry spell can start partway through."""
    if min_days < 1:
        raise ValueError(f"min_days must be >= 1, got {min_days}")
    run = 0
    for mm in daily_forecast_mm:
        if mm < 0:
            raise ValueError(f"rainfall cannot be negative, got {mm}")
        if mm < LIGHT_THRESHOLD_MM:
            run += 1
            if run >= min_days:
                return True
        else:
            run = 0
    return False


def generate_advisory(
    p10: float,
    p50: float,
    p90: float,
    crop_stage: Optional[CropStage] = None,
    forecast_series_mm: Optional[Sequence[float]] = None,
) -> dict:
    """Build the farmer-facing advisory for one panchayat/day.

    p10/p50/p90: the quantile forecast for this cell (mm/day), p50 drives the headline category.
    crop_stage: optional refinement; falls back to the base weather-only rule when None or when
      no override exists for this (category, stage) pair.
    forecast_series_mm: optional multi-day forecast (this day onward) used only to detect a dry
      spell; a single day's p50 alone can never justify "expected over the next several days".
    """
    if not (p10 <= p50 <= p90):
        raise ValueError(f"expected p10 <= p50 <= p90, got p10={p10}, p50={p50}, p90={p90}")

    category = classify_rainfall(p50)
    p10_category = classify_rainfall(p10)
    p90_category = classify_rainfall(p90)

    dry_spell = False
    if forecast_series_mm is not None:
        dry_spell = detect_dry_spell(forecast_series_mm)

    if dry_spell and category in (RainCategory.NO_RAIN, RainCategory.LIGHT):
        text = _DRY_SPELL_ADVISORY
    elif crop_stage is not None and (category, crop_stage) in _CROP_STAGE_OVERRIDES:
        text = _CROP_STAGE_OVERRIDES[(category, crop_stage)]
    else:
        text = _BASE_ADVISORY[category]

    return {
        "category": category.value,
        "advisory_text": text,
        "dry_spell": dry_spell,
        "crop_stage": crop_stage.value if crop_stage is not None else None,
        "uncertain": p10_category != p90_category,
    }
