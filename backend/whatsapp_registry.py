"""Farmer registry: phone number -> panchayat, per docs/ISSUES_PLAN.md item 4b ("panchayat -> phone
numbers, manual entry table for the Pune pilot villages"). Deliberately a plain CSV, same convention
as backend/feedback.py (and ml/train.py's own experiment log before that) - not a database, matching
this project's established "keep it simple" pattern for pilot-stage state.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "whatsapp_farmers.csv"
_FIELDS = ["registered_at_utc", "phone", "village_name", "sub_district"]


def register_farmer(phone: str, village_name: str, sub_district: str) -> dict:
    """Adds or updates one farmer's registration. phone is Meta's own format: E.164 digits, no '+'
    (e.g. '919876543210'). Re-registering the same phone overwrites their village (a farmer who moves
    fields shouldn't need a separate "unregister" step first) rather than appending a duplicate row."""
    phone = phone.strip()
    if not phone or not phone.isdigit():
        raise ValueError(f"phone must be digits only (E.164 without '+'), got {phone!r}")
    if not village_name.strip():
        raise ValueError("village_name must not be empty")

    farmers = {f["phone"]: f for f in list_farmers()}
    row = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "phone": phone,
        "village_name": village_name.strip(),
        "sub_district": sub_district.strip(),
    }
    farmers[phone] = row

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in farmers.values():
            writer.writerow(r)
    return row


def list_farmers() -> list[dict]:
    """Every registered farmer, oldest-registered first. Empty list if none registered yet - not an
    error, a fresh deployment legitimately has no farmers yet."""
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_farmer_by_phone(phone: str) -> dict | None:
    for farmer in list_farmers():
        if farmer["phone"] == phone.strip():
            return farmer
    return None
