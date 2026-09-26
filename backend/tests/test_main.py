"""Tests for the FastAPI app (backend/main.py) via TestClient - real data/model, since the endpoint's
whole job is wiring those real pieces together correctly.

Run: python -m pytest backend/tests/test_main.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DEFAULT_CHECKPOINT, TRAINING_PAIRS_DIR

pytestmark = pytest.mark.skipif(
    not (TRAINING_PAIRS_DIR / "era5_humidity_wind.npz").exists() or not DEFAULT_CHECKPOINT.exists(),
    reason="real training-pairs dataset or default checkpoint not present",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_dates_available"] == 5582


def test_forecast_happy_path(client):
    r = client.get("/forecast", params={"lat": 18.75, "lon": 73.41, "date": "2023-07-15"})
    assert r.status_code == 200
    body = r.json()
    assert "Lonavala" in body["village"]["name"]
    assert body["panchayat_level"]["p10_mm"] <= body["panchayat_level"]["p50_mm"] + 1e-4
    assert body["panchayat_level"]["p50_mm"] <= body["panchayat_level"]["p90_mm"] + 1e-4
    assert body["advisory"]["category"] in ["no_rain", "light", "moderate", "heavy"]
    assert body["advisory"]["advisory_text"]


def test_forecast_unknown_date_returns_404(client):
    r = client.get("/forecast", params={"lat": 18.75, "lon": 73.41, "date": "1900-01-01"})
    assert r.status_code == 404


def test_forecast_invalid_crop_stage_returns_400(client):
    r = client.get("/forecast", params={"lat": 18.75, "lon": 73.41, "date": "2023-07-15", "crop_stage": "not_a_real_stage"})
    assert r.status_code == 400


def test_forecast_with_crop_stage(client):
    r = client.get("/forecast", params={"lat": 18.75, "lon": 73.41, "date": "2023-07-15", "crop_stage": "sowing"})
    assert r.status_code == 200
    assert r.json()["advisory"]["crop_stage"] == "sowing"


def test_forecast_missing_required_param_returns_422(client):
    r = client.get("/forecast", params={"lat": 18.75, "date": "2023-07-15"})  # no lon
    assert r.status_code == 422
