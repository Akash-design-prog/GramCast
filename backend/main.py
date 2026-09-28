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

import base64
import math
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap_data import ensure_data_available
from inference import GramCastInference
from villages import VillageIndex, block_grid_geojson, bulk_village_stats, panchayat_value_by_index
from advisory.rules import CropStage, generate_advisory
from tts import SUPPORTED_LANGUAGES, synthesize_speech
from feedback import log_feedback
import whatsapp_client
import whatsapp_registry
from whatsapp_bot import handle_incoming_message

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

# The dashboard (frontend/dashboard) is a separate origin (Vite dev server on :5173, a different static
# host in production) - the browser blocks the cross-origin fetch without this. GRAMCAST_DASHBOARD_ORIGINS
# lets the deployed origin be set via env var; the two local dev ports are always allowed so `npm run dev`
# works out of the box.
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [o.strip() for o in os.environ.get("GRAMCAST_DASHBOARD_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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

    row_index = village["row_index"]
    panchayat = {
        "p10": panchayat_value_by_index(pred["p10"], row_index),
        "p50": panchayat_value_by_index(pred["p50"], row_index),
        "p90": panchayat_value_by_index(pred["p90"], row_index),
    }
    block = panchayat_value_by_index(pred["block_value"], row_index)
    # terrain["dem"] is static (built once at load time, doesn't vary by date) but lives on the exact
    # same (35, 50) grid as every rainfall raster, so the same O(1) per-village lookup already built for
    # rainfall works unchanged - no new data loading or overlay logic needed.
    elevation_m = panchayat_value_by_index(_inference.terrain["dem"], row_index)["mean_mm"]

    advisory = generate_advisory(
        p10=panchayat["p10"]["mean_mm"],
        p50=panchayat["p50"]["mean_mm"],
        p90=panchayat["p90"]["mean_mm"],
        crop_stage=parsed_crop_stage,
    )

    return {
        "village": {k: v for k, v in village.items() if k != "row_index"},  # internal detail, not API contract
        "date": date,
        "elevation_m": round(elevation_m, 1),
        "block_level_mm": block["mean_mm"],
        "panchayat_level": {
            "p10_mm": panchayat["p10"]["mean_mm"],
            "p50_mm": panchayat["p50"]["mean_mm"],
            "p90_mm": panchayat["p90"]["mean_mm"],
        },
        "advisory": advisory,
        "model": {"checkpoint": pred["checkpoint"], "epoch": pred["checkpoint_epoch"]},
    }


@app.get("/dates")
def dates():
    """Every date the dataset actually has a prediction for - the dashboard's date picker needs this to
    only offer real dates instead of a full 1981-2026 calendar range (the data is Jun-Sep monsoon season
    only, with real gaps, e.g. the missing 2026-09 month - see docs/ISSUES_PLAN.md train/val/test split)."""
    if _inference is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    available = _inference.sorted_available_dates()
    return {"dates": available, "min": available[0], "max": available[-1]}


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


@app.get("/forecast/map")
def forecast_map(
    date: str = Query(..., description="Date in the historical dataset, YYYY-MM-DD"),
):
    """Every village's block-level and panchayat-level (P10/P50/P90) rainfall for one day, as a single
    GeoJSON FeatureCollection - the dashboard's map layer needs the whole district in one request, not
    one round trip per village (see docs/ISSUES_PLAN.md, frontend bucket)."""
    if _inference is None or _village_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        pred = _inference.predict_day(date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    stats = bulk_village_stats(pred)

    def _clean(value: float) -> float | None:
        return None if value is None or math.isnan(value) else round(float(value), 2)

    features = []
    for i, (_, row) in enumerate(stats.iterrows()):
        features.append({
            "type": "Feature",
            # A stable per-row index, NOT a business key - NAME alone repeats 383/2003 times across the
            # district (e.g. "Shindewadi" appears in 6 different talukas) and NAME+SUB_DIST isn't unique
            # either (see bulk_village_stats' own docstring). The dashboard's map needs a real unique id
            # per polygon to highlight the one actually clicked instead of every village sharing its name.
            "id": i,
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "name": row["NAME"],
                "sub_district": row["SUB_DIST"],
                "block_mm": _clean(row["block_mm"]),
                "p10_mm": _clean(row["p10_mm"]),
                "p50_mm": _clean(row["p50_mm"]),
                "p90_mm": _clean(row["p90_mm"]),
            },
        })

    return {
        "type": "FeatureCollection",
        "date": date,
        "model": {"checkpoint": pred["checkpoint"], "epoch": pred["checkpoint_epoch"]},
        "features": features,
        # Real 0.25deg CHIRPS grid squares (not village-shaped patches) for the dashboard's block-layer
        # pane - same block_mm data as each feature's "block_mm" property above, just carried on its own
        # true grid geometry instead of village boundaries (see zonal_stats.block_grid_geojson).
        "grid_cells": block_grid_geojson(pred["block_value"]),
    }


class FeedbackIn(BaseModel):
    lat: float
    lon: float
    date: str
    reply_text: str


@app.post("/feedback")
def feedback(body: FeedbackIn):
    """Logs a farmer's reply against the forecast actually issued for their village and date - callable
    directly (e.g. from the dashboard) or via the WhatsApp webhook below, which logs a non-keyword
    reply from a registered farmer the same way."""
    if _inference is None or _village_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    village = _village_index.find(body.lat, body.lon)
    try:
        pred = _inference.predict_day(body.date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    p50 = panchayat_value_by_index(pred["p50"], village["row_index"])["mean_mm"]

    try:
        row = log_feedback(village["name"], village["sub_district"], body.date, p50, body.reply_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"logged": row}


class FarmerIn(BaseModel):
    phone: str
    village_name: str
    sub_district: str


@app.post("/whatsapp/farmers")
def register_farmer(body: FarmerIn):
    """Registers (or updates) one farmer's WhatsApp number against a real village - docs/ISSUES_PLAN.md
    item 4b's "manual entry table for the Pune pilot villages", exposed as an endpoint instead of asking
    Akash to hand-edit a CSV. Rejects a village name that doesn't actually exist in the real boundaries
    (a typo caught here is much cheaper than one only discovered when an alert silently never sends)."""
    if _village_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        _village_index.find_row_index_for(body.village_name, body.sub_district)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        row = whatsapp_registry.register_farmer(body.phone, body.village_name, body.sub_district)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"registered": row}


@app.get("/whatsapp/farmers")
def list_farmers():
    return {"farmers": whatsapp_registry.list_farmers()}


@app.get("/whatsapp/webhook")
def whatsapp_webhook_verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
):
    """Meta's one-time webhook verification handshake (see whatsapp_client.verify_webhook_challenge's
    own docstring) - Meta calls this GET when Akash saves the webhook config in the app dashboard, and
    refuses to activate the webhook unless this echoes back hub.challenge exactly."""
    try:
        challenge = whatsapp_client.verify_webhook_challenge(hub_mode, hub_verify_token, hub_challenge)
    except whatsapp_client.WhatsAppError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return Response(content=challenge, media_type="text/plain")


@app.post("/whatsapp/webhook")
def whatsapp_webhook_receive(body: dict):
    """Real incoming-message handling (see whatsapp_bot.handle_incoming_message's own docstring for the
    reply logic). Always returns 200 with an empty body - Meta retries a webhook that doesn't respond
    200 quickly, and a message this bot can't usefully handle should still be acknowledged, not treated
    as a delivery failure and retried."""
    if _inference is None or _village_index is None:
        return {}

    for msg in whatsapp_client.parse_incoming_messages(body):
        if not msg["from"] or msg["text"] is None:
            continue  # non-text message (image/audio/status update/etc) - not handled by this MVP bot
        result = handle_incoming_message(_inference, _village_index, msg["from"], msg["text"])
        if not whatsapp_client.is_configured():
            continue  # no real credentials yet - computed the reply, just can't send it out
        whatsapp_client.send_text_message(msg["from"], result["reply_text"])
        if result["advisory_text"]:
            # Real forecast reply - also attach the same Marathi voice note /forecast/voice already
            # produces, per docs/ISSUES_PLAN.md's own "attach TTS voice note to the WhatsApp reply".
            audio_bytes, _, media_type = synthesize_speech(result["advisory_text"], "mr")
            media_id = whatsapp_client.upload_media(audio_bytes, media_type)
            whatsapp_client.send_audio_message(msg["from"], media_id)
    return {}


class WhatsAppSimulateIn(BaseModel):
    from_: str = Field(alias="from")
    text: str


@app.post("/whatsapp/simulate")
def whatsapp_simulate(body: WhatsAppSimulateIn):
    """Demo-only endpoint for the WhatsApp-styled web simulator (frontend/dashboard/public/whatsapp-
    simulator.html) - a fallback for the pitch video/demo if real Meta credentials aren't ready in time.

    Calls the EXACT SAME handle_incoming_message() the real /whatsapp/webhook uses - this is not a
    separate reimplementation, so whatever the simulator shows is genuinely what the bot would say over
    real WhatsApp too, not a scripted/fake demo. The one real difference: this returns the reply (and a
    base64 voice note) directly in the HTTP response for the browser to render, instead of pushing it
    out through the Cloud API - there is no real WhatsApp message involved anywhere in this endpoint."""
    if _inference is None or _village_index is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    result = handle_incoming_message(_inference, _village_index, body.from_, body.text)
    response = {"reply_text": result["reply_text"], "audio_base64": None, "audio_media_type": None}
    if result["advisory_text"]:
        audio_bytes, _, media_type = synthesize_speech(result["advisory_text"], "mr")
        response["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
        response["audio_media_type"] = media_type
    return response
