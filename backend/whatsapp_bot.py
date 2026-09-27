"""WhatsApp bot message-handling logic - deliberately decoupled from FastAPI routing and Meta's webhook
JSON shape (that parsing lives in whatsapp_client.parse_incoming_messages()) so this is fully unit-
testable without a real HTTP round trip: main.py's POST /whatsapp/webhook route just parses the payload,
calls handle_incoming_message() per message, and sends whatever reply it returns via whatsapp_client.

Per docs/ISSUES_PLAN.md item 4b's own scope (two-way webhook: farmer sends a keyword -> bot replies
with forecast + advisory; feedback replies wired into backend logging):

  1. Message names a real village ("Baramati weather") -> forecast for THAT village, regardless of
     whether the sender is a registered farmer.
  2. Message is a bare keyword ("weather"/"forecast"/"rain"/...) and the sender IS a registered farmer
     -> forecast for their registered village.
  3. Anything else from a registered farmer -> logged as feedback against their last known forecast
     (backend/feedback.py), not silently dropped.
  4. Unregistered sender, no village named, no keyword -> a short help message.

"Today" is always the dataset's own latest available date (GramCastInference.sorted_available_dates()
[-1]) - this project's backend is explicitly past-event-mode only (see backend/inference.py's own
docstring), so there is no real "today" to ask a live model for; the most recent real historical day
is the closest honest stand-in, not a live forecast.
"""
from advisory.rules import generate_advisory
from feedback import log_feedback
from villages import panchayat_value_by_index
import whatsapp_registry

_FORECAST_KEYWORDS = {"weather", "forecast", "rain", "rainfall", "मौसम", "पाऊस"}


def _forecast_reply(inference, row_index: int, village_name: str, sub_district: str, date: str) -> tuple[str, str]:
    """Returns (reply_text, advisory_text). Raises ValueError (from panchayat_value_by_index) if this
    village genuinely has no pixel coverage for this date - the caller decides how to handle that."""
    pred = inference.predict_day(date)
    p50 = panchayat_value_by_index(pred["p50"], row_index)
    p10 = panchayat_value_by_index(pred["p10"], row_index)
    p90 = panchayat_value_by_index(pred["p90"], row_index)
    advisory = generate_advisory(p10=p10["mean_mm"], p50=p50["mean_mm"], p90=p90["mean_mm"])
    reply_text = (
        f"{village_name} ({sub_district}), {date}\n"
        f"Rainfall forecast: {p50['mean_mm']:.1f}mm (likely range {p10['mean_mm']:.0f}-{p90['mean_mm']:.0f}mm)\n"
        f"{advisory['advisory_text']}"
    )
    return reply_text, advisory["advisory_text"]


def handle_incoming_message(inference, village_index, from_phone: str, text: str) -> dict:
    """inference: a GramCastInference instance. village_index: a VillageIndex instance. Both passed in
    explicitly (not imported from main.py's module-level globals) so this function has no import-time
    dependency on FastAPI or app startup order, and can be unit-tested with lightweight fakes.

    Returns {"reply_text": str, "advisory_text": str | None} - advisory_text is set only for a genuine
    forecast reply; the webhook route uses it to decide whether to also synthesize and attach a TTS
    voice note (never doing so for a plain acknowledgement/help message)."""
    text = (text or "").strip()
    text_lower = text.lower()

    mentioned = village_index.find_by_name_fragment(text) if text else None
    farmer = whatsapp_registry.find_farmer_by_phone(from_phone) if from_phone else None

    target = None
    if mentioned is not None:
        target = mentioned
    # Exact match against the keyword set, not "keyword appears anywhere in the message" - a real bug
    # caught here via "test the test": farmer feedback like "no rain fell here today" legitimately
    # contains the word "rain" too, so substring-anywhere containment misread real feedback as a bare
    # forecast request. A farmer asking for the forecast types just "weather"/"rain"/etc, not a sentence.
    elif farmer is not None and text_lower in _FORECAST_KEYWORDS:
        target = {
            "name": farmer["village_name"],
            "sub_district": farmer["sub_district"],
            "row_index": village_index.find_row_index_for(farmer["village_name"], farmer["sub_district"]),
        }

    if target is not None:
        date = inference.sorted_available_dates()[-1]
        try:
            reply_text, advisory_text = _forecast_reply(inference, target["row_index"], target["name"], target["sub_district"], date)
        except ValueError:
            return {
                "reply_text": f"No rainfall coverage found for {target['name']} on {date} - try mentioning a nearby village name.",
                "advisory_text": None,
            }
        return {"reply_text": reply_text, "advisory_text": advisory_text}

    if farmer is not None:
        if not text:
            return {"reply_text": "Please send a message - a village name, 'weather', or your feedback.", "advisory_text": None}
        date = inference.sorted_available_dates()[-1]
        row_index = village_index.find_row_index_for(farmer["village_name"], farmer["sub_district"])
        pred = inference.predict_day(date)
        p50_mm = panchayat_value_by_index(pred["p50"], row_index)["mean_mm"]
        log_feedback(farmer["village_name"], farmer["sub_district"], date, p50_mm, text)
        return {"reply_text": "Thanks - your feedback has been logged.", "advisory_text": None}

    return {
        "reply_text": "Hi! Reply with your village name, or 'weather' for today's forecast (if you're a registered farmer).",
        "advisory_text": None,
    }
