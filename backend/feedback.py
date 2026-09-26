"""Farmer feedback logging, per the guide's own framing (section 8b / FAQ): "log each reply (for
example 'no rain here' or 'it hailed') against the model's prediction for that panchayat and date" -
even without retraining, this is a real talking point (the system improves with real-world use in
exactly the places IMD has no station).

Deliberately a plain append-only CSV, not a database - matches this project's existing convention
(ml/train.py's own CSV experiment log) and the guide's explicit "keep it simple" framing for this
feature. The WhatsApp webhook that will actually call this (docs/ISSUES_PLAN.md, item 4b) isn't built
yet - that's Akash's own Meta Business API setup - but the logging capability itself doesn't need to
wait on that; whatever channel eventually collects a reply just needs to call log_feedback().
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_LOG_PATH = Path(__file__).resolve().parent / "data" / "farmer_feedback.csv"
_FIELDS = ["logged_at_utc", "village_name", "sub_district", "forecast_date", "predicted_p50_mm", "reply_text"]


def log_feedback(village_name: str, sub_district: str, forecast_date: str, predicted_p50_mm: float, reply_text: str) -> dict:
    """Appends one row. Raises ValueError for an empty reply (a blank reply is not useful feedback -
    fail loudly here rather than silently log noise)."""
    if not reply_text or not reply_text.strip():
        raise ValueError("reply_text must not be empty")

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not FEEDBACK_LOG_PATH.exists()

    row = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "village_name": village_name,
        "sub_district": sub_district,
        "forecast_date": forecast_date,
        "predicted_p50_mm": predicted_p50_mm,
        "reply_text": reply_text.strip(),
    }
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_feedback() -> list[dict]:
    """Returns every logged row, oldest first. Empty list if nothing has been logged yet (not an error -
    a fresh deployment legitimately has no feedback yet)."""
    if not FEEDBACK_LOG_PATH.exists():
        return []
    with open(FEEDBACK_LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
