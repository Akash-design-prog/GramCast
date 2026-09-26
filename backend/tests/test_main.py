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


def test_cors_allows_dashboard_dev_origin(client):
    """The dashboard runs on a different origin (Vite on :5173) than the API (:8000) - without CORS
    headers the browser silently blocks every fetch, which is exactly what happened during frontend
    integration (curl succeeded, the browser fetch failed with no useful error)."""
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    r = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


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


@pytest.mark.network
def test_forecast_voice_returns_audio(client):
    r = client.get("/forecast/voice", params={"lat": 18.75, "lon": 73.41, "date": "2023-07-15"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert len(r.content) > 1000


@pytest.mark.network
def test_forecast_voice_default_language_is_marathi(client):
    """No lang param given - default must be 'mr' per the Pune pilot, not silently 'en'."""
    r_default = client.get("/forecast/voice", params={"lat": 18.75, "lon": 73.41, "date": "2023-07-15"})
    r_english = client.get("/forecast/voice", params={"lat": 18.75, "lon": 73.41, "date": "2023-07-15", "lang": "en"})
    assert r_default.content != r_english.content  # different language must produce different audio


def test_forecast_voice_unknown_date_returns_404(client):
    r = client.get("/forecast/voice", params={"lat": 18.75, "lon": 73.41, "date": "1900-01-01"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_forecast_crop_stage_is_case_sensitive_rejects_capitalized(client):
    """Capitalized 'Sowing' must be rejected with a clear 400, not silently accepted or crashing -
    CropStage's enum values are lowercase, and there's no case-folding in main.py."""
    r = client.get("/forecast", params={"lat": 18.75, "lon": 73.41, "date": "2023-07-15", "crop_stage": "Sowing"})
    assert r.status_code == 400


def test_forecast_nonexistent_calendar_date_returns_404_not_crash():
    """Feb 30 doesn't exist on any calendar - must be treated as 'not in dates.npy' (404), not raise
    a date-parsing exception."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        r = c.get("/forecast", params={"lat": 18.75, "lon": 73.41, "date": "2023-02-30"})
    assert r.status_code == 404


def test_forecast_extreme_out_of_range_lat_lon_does_not_crash(client):
    """Physically nonsensical coordinates (lat=200, lon=500) must still resolve to SOME village via
    the nearest-centroid fallback, not crash the endpoint."""
    r = client.get("/forecast", params={"lat": 200.0, "lon": 500.0, "date": "2023-07-15"})
    assert r.status_code == 200
    assert r.json()["village"]["matched_by"] == "nearest_centroid_fallback"


def test_forecast_map_happy_path(client):
    r = client.get("/forecast/map", params={"date": "2023-07-15"})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "FeatureCollection"
    assert body["date"] == "2023-07-15"
    assert len(body["features"]) == 2003  # full Pune district village count
    feature = body["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
    assert feature["id"] == 0
    props = feature["properties"]
    for key in ("name", "sub_district", "block_mm", "p10_mm", "p50_mm", "p90_mm"):
        assert key in props


def test_forecast_map_feature_ids_are_unique_and_sequential(client):
    """Regression test for a real bug: the dashboard used to highlight a clicked village by NAME, but
    383/2003 villages share a name with at least one other village elsewhere in the district (e.g.
    'Shindewadi' appears in 6 different talukas) - clicking one lit up every same-named village on the
    map. Every feature needs a genuinely unique id (NAME and NAME+SUB_DIST both fail this - see
    bulk_village_stats' own docstring), so the map can highlight exactly the one polygon clicked."""
    r = client.get("/forecast/map", params={"date": "2023-07-15"})
    ids = [f["id"] for f in r.json()["features"]]
    assert ids == list(range(2003))
    assert len(set(ids)) == len(ids)


def test_forecast_map_unknown_date_returns_404(client):
    r = client.get("/forecast/map", params={"date": "1900-01-01"})
    assert r.status_code == 404


def test_forecast_map_missing_date_returns_422(client):
    r = client.get("/forecast/map")
    assert r.status_code == 422


def test_forecast_map_quantiles_ordered_for_every_covered_village(client):
    """Full-scale check across the whole district, not a handful of villages: wherever a village has
    real coverage, P10 <= P50 <= P90 must hold after the GeoJSON round trip."""
    r = client.get("/forecast/map", params={"date": "2023-07-15"})
    body = r.json()
    checked = 0
    for feature in body["features"]:
        p = feature["properties"]
        if p["p10_mm"] is None or p["p50_mm"] is None or p["p90_mm"] is None:
            continue
        assert p["p10_mm"] <= p["p50_mm"] + 1e-6
        assert p["p50_mm"] <= p["p90_mm"] + 1e-6
        checked += 1
    assert checked > 1900  # nearly every village should have real coverage (100% per zonal_stats.py)


def test_forecast_map_values_are_json_numbers_not_nan(client):
    """json.dumps would silently emit the invalid literal NaN for an uncleaned float - every value must
    already be a finite number or an explicit null."""
    r = client.get("/forecast/map", params={"date": "2023-07-15"})
    for feature in r.json()["features"]:
        for key in ("block_mm", "p10_mm", "p50_mm", "p90_mm"):
            v = feature["properties"][key]
            assert v is None or (isinstance(v, (int, float)) and v == v)  # v == v is False only for NaN


def test_feedback_long_unicode_reply_text_logs_successfully(client):
    long_text = "no rain here " * 200 + "नाही"  # Devanagari "no" appended
    r = client.post("/feedback", json={"lat": 18.75, "lon": 73.41, "date": "2023-07-15", "reply_text": long_text})
    assert r.status_code == 200
    assert r.json()["logged"]["reply_text"] == long_text.strip()
