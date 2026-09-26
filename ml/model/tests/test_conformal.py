"""Tests for ml/conformal.py's pure-math functions (nonconformity scoring, the finite-sample-correct
conformal quantile, and coverage computation) - synthetic data, no real dataset/model needed.

Run: python -m pytest ml/model/tests/test_conformal.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conformal import conformal_quantile, coverage, nonconformity_scores


# ---------------------------------------------------------------------------
# conformal_quantile: the finite-sample-correct level calculation
# ---------------------------------------------------------------------------

def test_conformal_quantile_matches_manual_case():
    # n=9, alpha=0.2 -> level = ceil(10*0.8)/9 = ceil(8)/9 = 8/9
    scores = np.arange(1, 10, dtype="float32")  # 1..9
    q = conformal_quantile(scores, alpha=0.2)
    assert q == pytest.approx(np.quantile(scores, 8 / 9))


def test_conformal_quantile_clips_level_at_one():
    # very small n with strict alpha can push the level past 1.0 - must clip, not error
    scores = np.array([1.0, 2.0], dtype="float32")
    q = conformal_quantile(scores, alpha=0.01)
    assert q == pytest.approx(np.quantile(scores, 1.0))
    assert q == pytest.approx(2.0)


def test_conformal_quantile_all_identical_scores():
    scores = np.full(50, 3.0, dtype="float32")
    assert conformal_quantile(scores, alpha=0.2) == pytest.approx(3.0)


def test_conformal_quantile_can_be_negative():
    """A well-calibrated (or over-wide) band should be able to produce a negative q_hat, meaning the
    band could be narrowed - conformal correction isn't only ever a widening operation."""
    scores = np.array([-5.0, -4.0, -3.0, -2.0, -1.0], dtype="float32")
    q = conformal_quantile(scores, alpha=0.5)
    assert q < 0


# ---------------------------------------------------------------------------
# nonconformity_scores
# ---------------------------------------------------------------------------

def test_nonconformity_score_zero_when_inside_band():
    p10 = np.array([[0.0, 1.0]])
    p90 = np.array([[10.0, 5.0]])
    truth = np.array([[5.0, 3.0]])
    mask = np.array([True, True])
    scores = nonconformity_scores(p10, p90, truth, mask)
    assert (scores <= 0).all()


def test_nonconformity_score_positive_when_outside_band():
    p10 = np.array([[5.0]])
    p90 = np.array([[10.0]])
    truth = np.array([[20.0]])  # above p90 by 10
    mask = np.array([True])
    scores = nonconformity_scores(p10, p90, truth, mask)
    assert scores[0] == pytest.approx(10.0)


def test_nonconformity_score_below_p10():
    p10 = np.array([[5.0]])
    p90 = np.array([[10.0]])
    truth = np.array([[1.0]])  # below p10 by 4
    mask = np.array([True])
    scores = nonconformity_scores(p10, p90, truth, mask)
    assert scores[0] == pytest.approx(4.0)


def test_nonconformity_score_respects_mask():
    p10 = np.array([[0.0, 0.0]])
    p90 = np.array([[1.0, 1.0]])
    truth = np.array([[100.0, 0.5]])  # first pixel wildly outside band, second inside
    mask = np.array([False, True])  # only the second (inside) pixel is valid
    scores = nonconformity_scores(p10, p90, truth, mask)
    assert scores.size == 1
    assert scores[0] <= 0


def test_nonconformity_wet_only_excludes_dry_days():
    # shapes: (T, H, W) truth/p10/p90, (H, W) mask - matches real usage (fine_val is (T,35,50), valid_mask is (35,50))
    p10 = np.array([[[0.0]], [[0.0]]])  # T=2, H=1, W=1
    p90 = np.array([[[0.0]], [[5.0]]])
    truth = np.array([[[0.0]], [[3.0]]])  # day 0 is dry (truth=0), day 1 is wet
    mask = np.array([[True]])
    scores = nonconformity_scores(p10, p90, truth, mask, wet_only=True)
    assert scores.size == 1  # only the wet day counted


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_all_inside_band():
    p10 = np.zeros((3, 2))
    p90 = np.full((3, 2), 10.0)
    truth = np.full((3, 2), 5.0)
    mask = np.array([True, True])
    result = coverage(p10, p90, truth, mask)
    assert result["coverage"] == 1.0
    assert result["below_p10_frac"] == 0.0
    assert result["above_p90_frac"] == 0.0


def test_coverage_all_outside_above():
    p10 = np.zeros((2, 1))
    p90 = np.full((2, 1), 1.0)
    truth = np.full((2, 1), 100.0)
    mask = np.array([True])
    result = coverage(p10, p90, truth, mask)
    assert result["coverage"] == 0.0
    assert result["above_p90_frac"] == 1.0


def test_coverage_wet_only_dry_day_doesnt_count_as_trivial_win():
    """A day where p10=p90=truth=0 (correctly-predicted dry day) inflates plain coverage but must be
    excluded entirely from wet_only - this is the exact real finding that changed the honest read of
    this project's calibration numbers, so it must stay correct."""
    # T=2 days, H=1, W=2 pixels: pixel 0 dry both days-ish; simpler: 2 "pixel-days" via 2 columns, T=1
    p10 = np.array([[[0.0, 5.0]]])  # T=1, H=1, W=2
    p90 = np.array([[[0.0, 6.0]]])
    truth = np.array([[[0.0, 20.0]]])  # col0 dry day (trivially "covered"), col1 wet day (genuinely outside band)
    mask = np.array([[True, True]])
    all_days = coverage(p10, p90, truth, mask, wet_only=False)
    wet_only = coverage(p10, p90, truth, mask, wet_only=True)
    assert all_days["coverage"] == 0.5  # dry day counts as covered, inflating the average
    assert wet_only["coverage"] == 0.0  # only the wet (miscovered) day counts once dry days are excluded
    assert wet_only["n_pixel_days"] == 1


def test_coverage_wet_only_zero_wet_days_returns_nan_not_crash():
    p10 = np.zeros((2, 1, 1))
    p90 = np.zeros((2, 1, 1))
    truth = np.zeros((2, 1, 1))  # every day is dry
    mask = np.array([[True]])
    result = coverage(p10, p90, truth, mask, wet_only=True)
    assert result["n_pixel_days"] == 0
    assert np.isnan(result["coverage"])
