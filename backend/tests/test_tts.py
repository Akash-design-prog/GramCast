"""Tests for backend/tts.py - translation lookup and speech synthesis.

The completeness test is the important one: it enumerates every advisory_text
backend.advisory.rules.generate_advisory() can actually produce and confirms each one has a real
Hindi and Marathi translation on file - if rules.py's text changes without tts.py's table being
updated alongside it, THIS is what catches the drift, not a runtime surprise for a farmer.

Speech-synthesis tests hit the real gTTS API (network required) - acceptable here since there's no
useful way to test "does this produce real audio" without actually calling it.

Run: python -m pytest backend/tests/test_tts.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts import SUPPORTED_LANGUAGES, _TRANSLATIONS, synthesize_speech, translate_advisory
from advisory.rules import CropStage, generate_advisory


def _all_possible_advisory_texts() -> set[str]:
    """Every advisory_text generate_advisory() can produce: every (p50 in each category) x (crop_stage
    incl. None) combination, plus the dry-spell variant."""
    texts = set()
    category_p50s = [1.0, 5.0, 30.0, 100.0]  # one representative value per RainCategory
    for p50 in category_p50s:
        for stage in [None, *CropStage]:
            result = generate_advisory(p10=p50, p50=p50, p90=p50, crop_stage=stage)
            texts.add(result["advisory_text"])
    # dry-spell variant (only reachable for NO_RAIN/LIGHT categories)
    dry_spell_result = generate_advisory(p10=0.0, p50=1.0, p90=1.0, forecast_series_mm=[1.0, 1.0, 1.0, 1.0])
    texts.add(dry_spell_result["advisory_text"])
    return texts


def test_translation_table_covers_every_real_advisory_text():
    real_texts = _all_possible_advisory_texts()
    missing_hi = [t for t in real_texts if t not in _TRANSLATIONS or "hi" not in _TRANSLATIONS[t]]
    missing_mr = [t for t in real_texts if t not in _TRANSLATIONS or "mr" not in _TRANSLATIONS[t]]
    assert not missing_hi, f"missing Hindi translation for: {missing_hi}"
    assert not missing_mr, f"missing Marathi translation for: {missing_mr}"


def test_translate_english_returns_original():
    text = "Heavy rain likely. Delay spraying and fertiliser application; clear field drainage; secure harvested produce."
    assert translate_advisory(text, "en") == text


def test_translate_known_text_known_lang_differs_from_english():
    text = "Heavy rain likely. Delay spraying and fertiliser application; clear field drainage; secure harvested produce."
    mr = translate_advisory(text, "mr")
    hi = translate_advisory(text, "hi")
    assert mr != text
    assert hi != text
    assert mr != hi


def test_translate_unknown_text_falls_back_to_english_not_crash():
    unknown = "This advisory string does not exist in the table."
    assert translate_advisory(unknown, "mr") == unknown


def test_translate_unsupported_language_falls_back_to_english():
    text = "Heavy rain likely. Delay spraying and fertiliser application; clear field drainage; secure harvested produce."
    assert translate_advisory(text, "xx") == text


def test_supported_languages_contains_expected_set():
    assert SUPPORTED_LANGUAGES == {"en", "hi", "mr"}


@pytest.mark.network
def test_synthesize_speech_returns_nonempty_audio():
    audio, engine, media_type = synthesize_speech("Little or no rain expected today. Normal field operations; irrigate if soil is dry.", "mr")
    assert isinstance(audio, bytes)
    assert len(audio) > 1000  # a real audio file for a full sentence, not an empty/near-empty stub
    assert engine in ("bhashini", "gtts")
    assert media_type in ("audio/wav", "audio/mpeg")


@pytest.mark.network
def test_synthesize_speech_unsupported_language_does_not_crash():
    audio, engine, media_type = synthesize_speech("Little or no rain expected today. Normal field operations; irrigate if soil is dry.", "xx")
    assert len(audio) > 1000


@pytest.mark.network
def test_synthesize_speech_falls_back_to_gtts_when_bhashini_not_configured(monkeypatch):
    """Regression test for the hybrid's core promise: with no Bhashini credentials set, the engine
    used must be gtts, not a crash."""
    monkeypatch.delenv("BHASHINI_USER_ID", raising=False)
    monkeypatch.delenv("BHASHINI_API_KEY", raising=False)
    audio, engine, media_type = synthesize_speech("Little or no rain expected today. Normal field operations; irrigate if soil is dry.", "mr")
    assert engine == "gtts"
    assert media_type == "audio/mpeg"
    assert len(audio) > 1000


@pytest.mark.network
def test_synthesize_speech_falls_back_when_bhashini_raises(monkeypatch):
    """Bhashini configured but genuinely failing (bad key, network down, DIBD approval not through
    yet) must fall back to gTTS, not propagate the Bhashini error to the caller."""
    import bhashini_tts

    monkeypatch.setenv("BHASHINI_USER_ID", "fake")
    monkeypatch.setenv("BHASHINI_API_KEY", "fake")

    def _always_fails(text, lang):
        raise bhashini_tts.BhashiniError("simulated failure")

    monkeypatch.setattr(bhashini_tts, "synthesize_speech_bhashini", _always_fails)
    import tts as tts_module

    monkeypatch.setattr(tts_module, "synthesize_speech_bhashini", _always_fails)

    audio, engine, media_type = tts_module.synthesize_speech(
        "Little or no rain expected today. Normal field operations; irrigate if soil is dry.", "mr"
    )
    assert engine == "gtts"
    assert len(audio) > 1000
