"""FastAPI app: one endpoint that takes a place (lat/lon) and a date, returns the block-level value
(what today's forecast gives everyone in that ~0.25deg block) next to the panchayat-level value (this
project's actual output), plus the advisory text - per the guide's own framing of the API's job
(section 5b: "one endpoint that takes latitude, longitude and date and returns the values and the
advisory text").

Run: uvicorn backend.main:app --reload
"""
import sys
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap_data import ensure_data_available
from inference import GramCastInference
from villages import VillageIndex, panchayat_value
from advisory.rules import CropStage, generate_advisory
from tts import SUPPORTED_LANGUAGES, synthesize_speech
from feedback import log_feedback

_inference: GramCastInference | None = None
_village_index: VillageIndex | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loaded once at process startup, not per-request - the checkpoint and the full training-pair
    arrays are expensive to load (see GramCastInference's own docstring)."""
    global _inference, _village_index
    ensure_data_available()
    _inference = GramCastInference()
    _village_index = VillageIndex()
    yield


app = FastAPI(title="GramCast API", description="SIH26074 - block-to-panchayat rainfall downscaling", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": _inference.checkpoint_path.name if _inference else None,
        "n_dates_available": len(_inference.available_dates()) if _inference else 0,
    }


def _compute_forecast(lat: float, lon: float, date: str, crop_stage: str | None) -> dict:
    """Shared by /forecast and /forecast/voice - both need the exact same village lookup, model
    inference, zonal-stats overlay, and advisory generation; only the response format differs."""
    if _inference is None or _village_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    village = _village_index.find(lat, lon)

    try:
        pred = _inference.predict_day(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    parsed_crop_stage = None
    if crop_stage is not None:
        try:
            parsed_crop_stage = CropStage(crop_stage)
        except ValueError:
            valid = [s.value for s in CropStage]
            raise HTTPException(status_code=400, detail=f"crop_stage must be one of {valid}, got '{crop_stage}'")

    panchayat = {
        "p10": panchayat_value(pred["p10"], village["name"], village["sub_district"], date),
        "p50": panchayat_value(pred["p50"], village["name"], village["sub_district"], date),
        "p90": panchayat_value(pred["p90"], village["name"], village["sub_district"], date),
    }
    block = panchayat_value(pred["block_value"], village["name"], village["sub_district"], date)

    advisory = generate_advisory(
        p10=panchayat["p10"]["mean_mm"],
        p50=panchayat["p50"]["mean_mm"],
        p90=panchayat["p90"]["mean_mm"],
        crop_stage=parsed_crop_stage,
    )

    return {
        "village": village,
        "date": date,
        "block_level_mm": block["mean_mm"],
        "panchayat_level": {
            "p10_mm": panchayat["p10"]["mean_mm"],
            "p50_mm": panchayat["p50"]["mean_mm"],
            "p90_mm": panchayat["p90"]["mean_mm"],
        },
        "advisory": advisory,
        "model": {"checkpoint": pred["checkpoint"], "epoch": pred["checkpoint_epoch"]},
    }


@app.get("/forecast")
def forecast(
    lat: float = Query(..., description="Latitude of the place to look up"),
    lon: float = Query(..., description="Longitude of the place to look up"),
    date: str = Query(..., description="Date in the historical dataset, YYYY-MM-DD (past-event mode - see docs/ISSUES_PLAN.md for live-forecast mode)"),
    crop_stage: str | None = Query(None, description="Optional: sowing, vegetative, flowering, or harvest - refines the advisory text"),
):
    return _compute_forecast(lat, lon, date, crop_stage)


@app.get("/forecast/voice")
def forecast_voice(
    lat: float = Query(..., description="Latitude of the place to look up"),
    lon: float = Query(..., description="Longitude of the place to look up"),
    date: str = Query(..., description="Date in the historical dataset, YYYY-MM-DD"),
    crop_stage: str | None = Query(None, description="Optional: sowing, vegetative, flowering, or harvest"),
    lang: str = Query("mr", description=f"Voice language: one of {sorted(SUPPORTED_LANGUAGES)} - Marathi ('mr') by default, per the Pune pilot"),
):
    """Same advisory as /forecast, but as a spoken MP3 in the local language instead of JSON - the
    guide's own differentiator (section 8b): a farmer hears the advisory, doesn't have to read it."""
    result = _compute_forecast(lat, lon, date, crop_stage)
    audio_bytes, engine, media_type = synthesize_speech(result["advisory"]["advisory_text"], lang)
    return Response(content=audio_bytes, media_type=media_type, headers={"X-TTS-Engine": engine})


class FeedbackIn(BaseModel):
    lat: float
    lon: float
    date: str
    reply_text: str


@app.post("/feedback")
def feedback(body: FeedbackIn):
    """Logs a farmer's reply against the forecast actually issued for their village and date - the
    channel that collects the reply (WhatsApp webhook, docs/ISSUES_PLAN.md item 4b) isn't built yet,
    but the logging capability itself doesn't need to wait on that."""
    if _inference is None or _village_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    village = _village_index.find(body.lat, body.lon)
    try:
        pred = _inference.predict_day(body.date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    p50 = panchayat_value(pred["p50"], village["name"], village["sub_district"], body.date)["mean_mm"]

    try:
        row = log_feedback(village["name"], village["sub_district"], body.date, p50, body.reply_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"logged": row}
