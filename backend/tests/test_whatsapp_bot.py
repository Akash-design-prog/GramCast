"""Tests for backend/whatsapp_bot.py - the reply logic a real webhook message gets routed through.
Real inference + village boundaries (same skip condition as test_main.py: no reasonable synthetic
substitute for a real checkpoint + real 2,003 village polygons), with the farmer registry and feedback
log isolated to a temp path per test so nothing touches real project data.

Run: python -m pytest backend/tests/test_whatsapp_bot.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DEFAULT_CHECKPOINT, GramCastInference, TRAINING_PAIRS_DIR
from villages import VillageIndex
import feedback
import whatsapp_registry
from whatsapp_bot import handle_incoming_message

pytestmark = pytest.mark.skipif(
    not (TRAINING_PAIRS_DIR / "era5_humidity_wind.npz").exists() or not DEFAULT_CHECKPOINT.exists(),
    reason="real training-pairs dataset or default checkpoint not present",
)


@pytest.fixture(scope="module")
def inference():
    return GramCastInference()


@pytest.fixture(scope="module")
def village_index():
    return VillageIndex()


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own empty farmer registry and feedback log - never touches real project data."""
    monkeypatch.setattr(whatsapp_registry, "REGISTRY_PATH", tmp_path / "whatsapp_farmers.csv")
    monkeypatch.setattr(feedback, "FEEDBACK_LOG_PATH", tmp_path / "farmer_feedback.csv")


def test_message_names_a_real_village_returns_forecast(inference, village_index):
    result = handle_incoming_message(inference, village_index, "919876543210", "Lonavala weather")
    assert "Lonavala" in result["reply_text"]
    assert result["advisory_text"] is not None
    assert result["advisory_text"] in result["reply_text"]


def test_bare_keyword_from_registered_farmer_returns_their_village_forecast(inference, village_index):
    whatsapp_registry.register_farmer("919876543210", "Lonavala (M Cl)", "Mawal")
    result = handle_incoming_message(inference, village_index, "919876543210", "weather")
    assert "Lonavala" in result["reply_text"]
    assert result["advisory_text"] is not None


def test_named_village_overrides_registered_village(inference, village_index):
    """An ad-hoc query for a different village should win over the sender's own registration - a
    registered farmer asking about a neighbour's village shouldn't get their own forecast instead."""
    whatsapp_registry.register_farmer("919876543210", "Lonavala (M Cl)", "Mawal")
    result = handle_incoming_message(inference, village_index, "919876543210", "how about Pashan")
    assert "Pashan" in result["reply_text"]
    assert "Lonavala" not in result["reply_text"]


def test_bare_keyword_from_unregistered_sender_gets_help_message(inference, village_index):
    result = handle_incoming_message(inference, village_index, "000000000000", "weather")
    assert result["advisory_text"] is None
    assert "village name" in result["reply_text"].lower()


def test_non_keyword_message_from_registered_farmer_logs_feedback(inference, village_index):
    whatsapp_registry.register_farmer("919876543211", "Lonavala (M Cl)", "Mawal")
    result = handle_incoming_message(inference, village_index, "919876543211", "no rain fell here today")
    assert result["advisory_text"] is None
    assert "logged" in result["reply_text"].lower()

    logged = feedback.read_feedback()
    assert len(logged) == 1
    assert logged[0]["village_name"] == "Lonavala (M Cl)"
    assert logged[0]["reply_text"] == "no rain fell here today"


def test_empty_message_from_registered_farmer_asks_for_content(inference, village_index):
    whatsapp_registry.register_farmer("919876543212", "Lonavala (M Cl)", "Mawal")
    result = handle_incoming_message(inference, village_index, "919876543212", "   ")
    assert result["advisory_text"] is None
    assert "send a message" in result["reply_text"].lower()
    assert feedback.read_feedback() == []


def test_unrecognized_message_from_unregistered_sender_gets_help_message(inference, village_index):
    result = handle_incoming_message(inference, village_index, "000000000000", "hello there")
    assert result["advisory_text"] is None
    assert "reply with your village name" in result["reply_text"].lower()
