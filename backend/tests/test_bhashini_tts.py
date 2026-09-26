"""Tests for backend/bhashini_tts.py - mocked HTTP calls, since we don't have real Bhashini credentials
(registration needs DIBD approval, not yet granted). These tests verify the request/response PARSING
logic is correct against the documented API shape, so that whenever real credentials do arrive, the
integration is already verified rather than being tested for the first time live.

Run: python -m pytest backend/tests/test_bhashini_tts.py -v
"""
import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bhashini_tts
from bhashini_tts import BhashiniError, _extract_config_fields, is_configured, synthesize_speech_bhashini


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------

def test_is_configured_false_when_no_env_vars(monkeypatch):
    monkeypatch.delenv("BHASHINI_USER_ID", raising=False)
    monkeypatch.delenv("BHASHINI_API_KEY", raising=False)
    assert is_configured() is False


def test_is_configured_false_when_only_one_env_var_set(monkeypatch):
    monkeypatch.setenv("BHASHINI_USER_ID", "some-id")
    monkeypatch.delenv("BHASHINI_API_KEY", raising=False)
    assert is_configured() is False


def test_is_configured_true_when_both_set(monkeypatch):
    monkeypatch.setenv("BHASHINI_USER_ID", "some-id")
    monkeypatch.setenv("BHASHINI_API_KEY", "some-key")
    assert is_configured() is True


# ---------------------------------------------------------------------------
# _extract_config_fields - the exact documented response shape, and "test the test"
# via a genuinely malformed response
# ---------------------------------------------------------------------------

def _real_shaped_config_response(service_id="ai4bharat/tts-mr-gpu", url="https://dhruva-api.bhashini.gov.in/services/inference/pipeline", auth_value="secret-token"):
    return {
        "pipelineResponseConfig": [{"taskType": "tts", "config": [{"serviceId": service_id}]}],
        "pipelineInferenceAPIEndPoint": {
            "callbackUrl": url,
            "inferenceApiKey": {"name": "Authorization", "value": auth_value},
        },
    }


def test_extract_config_fields_real_shape():
    response = _real_shaped_config_response()
    service_id, compute_url, auth_name, auth_value = _extract_config_fields(response)
    assert service_id == "ai4bharat/tts-mr-gpu"
    assert compute_url == "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    assert auth_name == "Authorization"
    assert auth_value == "secret-token"


def test_extract_config_fields_missing_key_raises_bhashini_error():
    malformed = {"pipelineResponseConfig": [{"config": [{}]}]}  # no serviceId at all
    with pytest.raises(BhashiniError, match="unexpected pipeline config response shape"):
        _extract_config_fields(malformed)


def test_extract_config_fields_empty_response_raises():
    with pytest.raises(BhashiniError):
        _extract_config_fields({})


# ---------------------------------------------------------------------------
# synthesize_speech_bhashini - full flow with mocked HTTP, both success and each
# documented failure mode
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_synthesize_speech_bhashini_full_success_flow(monkeypatch):
    monkeypatch.setenv("BHASHINI_USER_ID", "test-user")
    monkeypatch.setenv("BHASHINI_API_KEY", "test-key")

    config_resp = _mock_response(200, _real_shaped_config_response())
    fake_audio_bytes = b"FAKE_WAV_AUDIO_DATA"
    compute_resp = _mock_response(200, {
        "pipelineResponse": [{"audio": [{"audioContent": base64.b64encode(fake_audio_bytes).decode()}]}]
    })

    with patch("bhashini_tts.requests.post", side_effect=[config_resp, compute_resp]) as mock_post:
        result = synthesize_speech_bhashini("test text", "mr")

    assert result == fake_audio_bytes
    assert mock_post.call_count == 2
    # first call must hit the config endpoint with userID/ulcaApiKey headers
    first_call_kwargs = mock_post.call_args_list[0].kwargs
    assert first_call_kwargs["headers"] == {"userID": "test-user", "ulcaApiKey": "test-key"}
    # second call must hit the compute URL from the config response, with the auth header FROM that response
    second_call_args = mock_post.call_args_list[1]
    assert second_call_args.args[0] == "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    assert second_call_args.kwargs["headers"] == {"Authorization": "secret-token"}


def test_synthesize_speech_bhashini_raises_without_env_vars(monkeypatch):
    monkeypatch.delenv("BHASHINI_USER_ID", raising=False)
    monkeypatch.delenv("BHASHINI_API_KEY", raising=False)
    with pytest.raises(BhashiniError, match="not set"):
        synthesize_speech_bhashini("test text", "mr")


def test_synthesize_speech_bhashini_config_call_non_200_raises(monkeypatch):
    monkeypatch.setenv("BHASHINI_USER_ID", "test-user")
    monkeypatch.setenv("BHASHINI_API_KEY", "test-key")
    with patch("bhashini_tts.requests.post", return_value=_mock_response(401, text="unauthorized")):
        with pytest.raises(BhashiniError, match="401"):
            synthesize_speech_bhashini("test text", "mr")


def test_synthesize_speech_bhashini_compute_call_non_200_raises(monkeypatch):
    monkeypatch.setenv("BHASHINI_USER_ID", "test-user")
    monkeypatch.setenv("BHASHINI_API_KEY", "test-key")
    config_resp = _mock_response(200, _real_shaped_config_response())
    compute_resp = _mock_response(500, text="server error")
    with patch("bhashini_tts.requests.post", side_effect=[config_resp, compute_resp]):
        with pytest.raises(BhashiniError, match="500"):
            synthesize_speech_bhashini("test text", "mr")


def test_synthesize_speech_bhashini_network_error_raises_bhashini_error(monkeypatch):
    monkeypatch.setenv("BHASHINI_USER_ID", "test-user")
    monkeypatch.setenv("BHASHINI_API_KEY", "test-key")
    import requests

    with patch("bhashini_tts.requests.post", side_effect=requests.ConnectionError("no network")):
        with pytest.raises(BhashiniError, match="pipeline config call failed"):
            synthesize_speech_bhashini("test text", "mr")


def test_synthesize_speech_bhashini_malformed_compute_response_raises(monkeypatch):
    monkeypatch.setenv("BHASHINI_USER_ID", "test-user")
    monkeypatch.setenv("BHASHINI_API_KEY", "test-key")
    config_resp = _mock_response(200, _real_shaped_config_response())
    compute_resp = _mock_response(200, {"pipelineResponse": []})  # missing audio entirely
    with patch("bhashini_tts.requests.post", side_effect=[config_resp, compute_resp]):
        with pytest.raises(BhashiniError, match="unexpected pipeline compute response shape"):
            synthesize_speech_bhashini("test text", "mr")


def test_compute_tts_bad_base64_raises_bhashini_error():
    """audioContent that isn't valid base64 at all - a genuinely malformed response, not just a
    missing field - must still raise BhashiniError, not a bare binascii.Error."""
    compute_resp = _mock_response(200, {"pipelineResponse": [{"audio": [{"audioContent": "not-valid-base64!!"}]}]})
    with patch("bhashini_tts.requests.post", return_value=compute_resp):
        with pytest.raises(BhashiniError, match="base64"):
            bhashini_tts._compute_tts("text", "mr", "svc-id", "https://example.com", "Authorization", "token")
