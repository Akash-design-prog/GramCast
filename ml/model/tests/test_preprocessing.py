"""Tests for preprocessing.py's ERA5 integration (humidity/wind channels added 2026-09-26).

Run: python -m pytest ml/model/tests/test_preprocessing.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import (
    ERA5_CHANNELS,
    WORLDCOVER_CLASSES,
    build_input_tensor,
    channel_names,
    compute_normalization_stats,
)

T, H, W = 6, 5, 4


def make_terrain():
    rng = np.random.default_rng(0)
    return {
        "dem": rng.uniform(100, 900, size=(H, W)).astype("float32"),
        "slope": rng.uniform(0, 40, size=(H, W)).astype("float32"),
        "aspect": rng.uniform(0, 360, size=(H, W)).astype("float32"),
        "worldcover": rng.choice(WORLDCOVER_CLASSES, size=(H, W)).astype("float32"),
    }


def make_era5(t=T):
    rng = np.random.default_rng(1)
    return {
        "humidity_proxy": rng.uniform(280, 300, size=(t, H, W)).astype("float32"),
        "wind_u": rng.uniform(-10, 10, size=(t, H, W)).astype("float32"),
        "wind_v": rng.uniform(-10, 10, size=(t, H, W)).astype("float32"),
    }


def make_coarse(t=T):
    rng = np.random.default_rng(2)
    return rng.uniform(0, 50, size=(t, H, W)).astype("float32")


# ---------------------------------------------------------------------------
# compute_normalization_stats
# ---------------------------------------------------------------------------

def test_stats_include_era5_keys():
    terrain = make_terrain()
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    for name in ERA5_CHANNELS:
        assert f"{name}_mean" in stats
        assert f"{name}_std" in stats


def test_stats_values_match_manual_computation():
    terrain = make_terrain()
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    for name in ERA5_CHANNELS:
        assert stats[f"{name}_mean"] == pytest.approx(float(era5[name].mean()))
        assert stats[f"{name}_std"] == pytest.approx(float(era5[name].std()))


# ---------------------------------------------------------------------------
# build_input_tensor: shape and channel count
# ---------------------------------------------------------------------------

def test_output_channel_count_matches_channel_names():
    terrain = make_terrain()
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    tensor = build_input_tensor(coarse, terrain, stats, era5)
    names = channel_names()
    assert tensor.shape == (T, len(names), H, W)
    assert len(names) == 1 + len(ERA5_CHANNELS) + 4 + len(WORLDCOVER_CLASSES)  # rain + era5 + (dem,slope,sin,cos) + worldcover


def test_no_nan_in_output_given_no_nan_input():
    terrain = make_terrain()
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    tensor = build_input_tensor(coarse, terrain, stats, era5)
    assert not np.isnan(tensor).any()


def test_channel_order_matches_channel_names_semantically():
    """rain_bicubic must be channel 0, era5 channels next, in ERA5_CHANNELS order."""
    terrain = make_terrain()
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    tensor = build_input_tensor(coarse, terrain, stats, era5)

    rain_norm = (coarse - stats["rain_mean"]) / stats["rain_std"]
    assert np.allclose(tensor[:, 0], rain_norm)
    for i, name in enumerate(ERA5_CHANNELS, start=1):
        expected = (era5[name] - stats[f"{name}_mean"]) / stats[f"{name}_std"]
        assert np.allclose(tensor[:, i], expected), f"channel {i} ({name}) doesn't match expected normalized values"


# ---------------------------------------------------------------------------
# build_input_tensor: the shape-mismatch guard - test the test, not just that it exists
# ---------------------------------------------------------------------------

def test_mismatched_era5_shape_raises():
    terrain = make_terrain()
    coarse = make_coarse(t=T)
    era5 = make_era5(t=T - 1)  # deliberately wrong T
    stats = compute_normalization_stats(terrain, make_coarse(t=T - 1), era5)
    with pytest.raises(ValueError, match="doesn't match"):
        build_input_tensor(coarse, terrain, stats, era5)


def test_missing_era5_key_raises():
    terrain = make_terrain()
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    del era5["wind_v"]
    with pytest.raises(KeyError):
        build_input_tensor(coarse, terrain, stats, era5)


def test_missing_stat_key_raises():
    terrain = make_terrain()
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    del stats["humidity_proxy_std"]
    with pytest.raises(KeyError):
        build_input_tensor(coarse, terrain, stats, era5)


# ---------------------------------------------------------------------------
# Unknown WorldCover class guard (pre-existing behaviour) still works alongside the new era5 arg
# ---------------------------------------------------------------------------

def test_unknown_worldcover_class_still_raises():
    terrain = make_terrain()
    terrain["worldcover"] = terrain["worldcover"].copy()
    terrain["worldcover"][0, 0] = 999  # not in WORLDCOVER_CLASSES
    coarse = make_coarse()
    era5 = make_era5()
    stats = compute_normalization_stats(terrain, coarse, era5)
    with pytest.raises(ValueError, match="WorldCover class"):
        build_input_tensor(coarse, terrain, stats, era5)


# ---------------------------------------------------------------------------
# Integration: real data, full pipeline (skipped if the real training-pairs dataset isn't present)
# ---------------------------------------------------------------------------

REAL_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "training_pairs"


@pytest.mark.skipif(not (REAL_DATA_DIR / "era5_humidity_wind.npz").exists(), reason="real training-pairs dataset not present")
def test_real_dataset_full_pipeline():
    fine = np.load(REAL_DATA_DIR / "fine_rainfall.npy")
    coarse = np.load(REAL_DATA_DIR / "coarse_rainfall_bicubic.npy")
    splits = np.load(REAL_DATA_DIR / "split_indices.npz")
    terrain = dict(np.load(REAL_DATA_DIR / "terrain_static.npz"))
    era5_npz = np.load(REAL_DATA_DIR / "era5_humidity_wind.npz")
    era5 = {name: era5_npz[name] for name in ERA5_CHANNELS}

    train_idx = splits["train"]
    era5_train = {name: arr[train_idx] for name, arr in era5.items()}
    stats = compute_normalization_stats(terrain, coarse[train_idx], era5_train)

    era5_sliced = {name: arr[train_idx] for name, arr in era5.items()}
    tensor = build_input_tensor(coarse[train_idx], terrain, stats, era5_sliced)

    assert tensor.shape == (len(train_idx), 17, 35, 50)
    assert not np.isnan(tensor).any()
    # train-only normalization: each era5 channel's mean over the TRAIN slice should be ~0 after normalizing
    for i, name in enumerate(ERA5_CHANNELS, start=1):
        assert abs(float(tensor[:, i].mean())) < 0.05, f"{name} not properly zero-centered on its own train stats"
