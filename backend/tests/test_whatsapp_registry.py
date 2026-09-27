"""Tests for backend/whatsapp_registry.py - a plain CSV store, same convention as test_feedback.py
covers backend/feedback.py.

Run: python -m pytest backend/tests/test_whatsapp_registry.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import whatsapp_registry


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Every test gets its own empty CSV path - never touches the real data/whatsapp_farmers.csv."""
    monkeypatch.setattr(whatsapp_registry, "REGISTRY_PATH", tmp_path / "whatsapp_farmers.csv")


def test_list_farmers_empty_when_no_file_yet():
    assert whatsapp_registry.list_farmers() == []


def test_register_and_list_farmer():
    whatsapp_registry.register_farmer("919876543210", "Lonavala (M Cl)", "Mawal")
    farmers = whatsapp_registry.list_farmers()
    assert len(farmers) == 1
    assert farmers[0]["phone"] == "919876543210"
    assert farmers[0]["village_name"] == "Lonavala (M Cl)"
    assert farmers[0]["sub_district"] == "Mawal"


def test_register_same_phone_twice_updates_not_duplicates():
    whatsapp_registry.register_farmer("919876543210", "Lonavala (M Cl)", "Mawal")
    whatsapp_registry.register_farmer("919876543210", "Baramati (M Cl)", "Baramati")
    farmers = whatsapp_registry.list_farmers()
    assert len(farmers) == 1
    assert farmers[0]["village_name"] == "Baramati (M Cl)"


def test_register_multiple_distinct_farmers():
    whatsapp_registry.register_farmer("919876543210", "Lonavala (M Cl)", "Mawal")
    whatsapp_registry.register_farmer("919876543211", "Baramati (M Cl)", "Baramati")
    assert len(whatsapp_registry.list_farmers()) == 2


def test_register_rejects_non_digit_phone():
    with pytest.raises(ValueError, match="digits only"):
        whatsapp_registry.register_farmer("+91 98765 43210", "Lonavala (M Cl)", "Mawal")


def test_register_rejects_empty_village_name():
    with pytest.raises(ValueError, match="village_name"):
        whatsapp_registry.register_farmer("919876543210", "", "Mawal")


def test_find_farmer_by_phone_found_and_not_found():
    whatsapp_registry.register_farmer("919876543210", "Lonavala (M Cl)", "Mawal")
    assert whatsapp_registry.find_farmer_by_phone("919876543210")["village_name"] == "Lonavala (M Cl)"
    assert whatsapp_registry.find_farmer_by_phone("000000000000") is None
