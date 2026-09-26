"""Tests for backend/feedback.py - uses a temp file (monkeypatched FEEDBACK_LOG_PATH) rather than the
real log, so tests never write into the actual farmer-feedback file.

Run: python -m pytest backend/tests/test_feedback.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feedback as feedback_module
from feedback import log_feedback, read_feedback


@pytest.fixture(autouse=True)
def temp_log_path(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_module, "FEEDBACK_LOG_PATH", tmp_path / "farmer_feedback.csv")
    yield


def test_read_feedback_empty_before_any_logging():
    assert read_feedback() == []


def test_log_and_read_roundtrip():
    log_feedback("Lonavala (M Cl)", "Mawal", "2023-07-15", 12.5, "no rain here")
    rows = read_feedback()
    assert len(rows) == 1
    assert rows[0]["village_name"] == "Lonavala (M Cl)"
    assert rows[0]["sub_district"] == "Mawal"
    assert rows[0]["forecast_date"] == "2023-07-15"
    assert rows[0]["predicted_p50_mm"] == "12.5"
    assert rows[0]["reply_text"] == "no rain here"
    assert rows[0]["logged_at_utc"]  # a real timestamp was written


def test_log_appends_not_overwrites():
    log_feedback("Village A", "Sub A", "2023-01-01", 1.0, "first reply")
    log_feedback("Village B", "Sub B", "2023-01-02", 2.0, "second reply")
    rows = read_feedback()
    assert len(rows) == 2
    assert rows[0]["reply_text"] == "first reply"
    assert rows[1]["reply_text"] == "second reply"


def test_log_feedback_rejects_empty_reply():
    with pytest.raises(ValueError, match="empty"):
        log_feedback("Village A", "Sub A", "2023-01-01", 1.0, "")


def test_log_feedback_rejects_whitespace_only_reply():
    with pytest.raises(ValueError, match="empty"):
        log_feedback("Village A", "Sub A", "2023-01-01", 1.0, "   ")


def test_log_feedback_strips_whitespace():
    log_feedback("Village A", "Sub A", "2023-01-01", 1.0, "  it hailed  ")
    rows = read_feedback()
    assert rows[0]["reply_text"] == "it hailed"
