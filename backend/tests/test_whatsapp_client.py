"""Tests for backend/whatsapp_client.py - mocked HTTP calls, since we don't have real Meta credentials
yet (Akash is mid-setup on the Cloud API test-mode flow). These verify the request/response shapes
match Meta's own documented API (verified directly against developers.facebook.com before writing this
client - see whatsapp_client.py's own module docstring), so the integration is already checked before
real credentials arrive, same pattern as test_bhashini_tts.py.

Run: python -m pytest backend/tests/test_whatsapp_client.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import whatsapp_client
from whatsapp_client import (
    WhatsAppError,
    is_configured,
    parse_incoming_messages,
    send_audio_message,
    send_template_message,
    send_text_message,
    upload_media,
    verify_webhook_challenge,
)


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------

def test_is_configured_false_when_no_env_vars(monkeypatch):
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    assert is_configured() is False


def test_is_configured_false_when_only_one_env_var_set(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "token")
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    assert is_configured() is False


def test_is_configured_true_when_both_set(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "12345")
    assert is_configured() is True


def test_send_without_config_raises_whatsapp_error(monkeypatch):
    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("WHATSAPP_PHONE_NUMBER_ID", raising=False)
    with pytest.raises(WhatsAppError, match="not set"):
        send_text_message("919876543210", "hello")


# ---------------------------------------------------------------------------
# send_text_message / send_template_message - real documented request shape
# ---------------------------------------------------------------------------

@patch("whatsapp_client.requests.post")
def test_send_text_message_real_request_shape(mock_post, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "106540352242922")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"messages": [{"id": "wamid.abc"}]})

    result = send_text_message("919876543210", "Heavy rain expected today")

    args, kwargs = mock_post.call_args
    assert args[0] == f"https://graph.facebook.com/{whatsapp_client.GRAPH_API_VERSION}/106540352242922/messages"
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["json"] == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "919876543210",
        "type": "text",
        "text": {"preview_url": False, "body": "Heavy rain expected today"},
    }
    assert result["messages"][0]["id"] == "wamid.abc"


@patch("whatsapp_client.requests.post")
def test_send_template_message_real_request_shape(mock_post, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "106540352242922")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"messages": [{"id": "wamid.def"}]})

    send_template_message("919876543210", "gramcast_rain_alert", "en", ["Baramati", "45.0", "heavy"])

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["type"] == "template"
    assert kwargs["json"]["template"]["name"] == "gramcast_rain_alert"
    assert kwargs["json"]["template"]["language"] == {"code": "en"}
    params = kwargs["json"]["template"]["components"][0]["parameters"]
    assert params == [{"type": "text", "text": "Baramati"}, {"type": "text", "text": "45.0"}, {"type": "text", "text": "heavy"}]


@patch("whatsapp_client.requests.post")
def test_send_message_non_2xx_raises_whatsapp_error(mock_post, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "106540352242922")
    mock_post.return_value = MagicMock(status_code=401, text='{"error": {"message": "Invalid OAuth token"}}')

    with pytest.raises(WhatsAppError, match="401"):
        send_text_message("919876543210", "hello")


@patch("whatsapp_client.requests.post")
def test_send_message_network_error_raises_whatsapp_error(mock_post, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "106540352242922")
    import requests
    mock_post.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(WhatsAppError, match="network error"):
        send_text_message("919876543210", "hello")


# ---------------------------------------------------------------------------
# upload_media / send_audio_message
# ---------------------------------------------------------------------------

@patch("whatsapp_client.requests.post")
def test_upload_media_real_request_shape(mock_post, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "106540352242922")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"id": "media-id-123"})

    media_id = upload_media(b"fake-audio-bytes", "audio/mpeg", filename="advisory.mp3")

    args, kwargs = mock_post.call_args
    assert args[0] == f"https://graph.facebook.com/{whatsapp_client.GRAPH_API_VERSION}/106540352242922/media"
    assert kwargs["data"] == {"messaging_product": "whatsapp"}
    assert kwargs["files"]["file"] == ("advisory.mp3", b"fake-audio-bytes", "audio/mpeg")
    assert media_id == "media-id-123"


@patch("whatsapp_client.requests.post")
def test_upload_media_missing_id_in_response_raises(mock_post, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "106540352242922")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"unexpected": "shape"})

    with pytest.raises(WhatsAppError, match="missing 'id'"):
        upload_media(b"bytes", "audio/mpeg")


@patch("whatsapp_client.requests.post")
def test_send_audio_message_real_request_shape(mock_post, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "106540352242922")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"messages": [{"id": "wamid.ghi"}]})

    send_audio_message("919876543210", "media-id-123")

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["type"] == "audio"
    assert kwargs["json"]["audio"] == {"id": "media-id-123"}


# ---------------------------------------------------------------------------
# verify_webhook_challenge - Meta's real GET verification handshake
# ---------------------------------------------------------------------------

def test_verify_webhook_challenge_success(monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my_verify_token")
    assert verify_webhook_challenge("subscribe", "my_verify_token", "1158201444") == "1158201444"


def test_verify_webhook_challenge_wrong_token_raises(monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my_verify_token")
    with pytest.raises(WhatsAppError, match="failed"):
        verify_webhook_challenge("subscribe", "wrong_token", "1158201444")


def test_verify_webhook_challenge_wrong_mode_raises(monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my_verify_token")
    with pytest.raises(WhatsAppError, match="failed"):
        verify_webhook_challenge("unsubscribe", "my_verify_token", "1158201444")


def test_verify_webhook_challenge_not_configured_raises(monkeypatch):
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)
    with pytest.raises(WhatsAppError, match="not set"):
        verify_webhook_challenge("subscribe", "anything", "123")


# ---------------------------------------------------------------------------
# parse_incoming_messages - Meta's real webhook POST payload shape, and "test the test" via a
# deliberately malformed/partial payload (status updates, retried deliveries don't always carry
# 'messages')
# ---------------------------------------------------------------------------

def _real_shaped_incoming_payload(from_phone="919876543210", text="weather", message_id="wamid.xyz"):
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "WABA_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "911234567890", "phone_number_id": "106540352242922"},
                    "contacts": [{"profile": {"name": "Farmer"}, "wa_id": from_phone}],
                    "messages": [{"from": from_phone, "id": message_id, "timestamp": "1234567890", "type": "text", "text": {"body": text}}],
                },
                "field": "messages",
            }],
        }],
    }


def test_parse_incoming_messages_real_shape():
    payload = _real_shaped_incoming_payload()
    messages = parse_incoming_messages(payload)
    assert messages == [{"from": "919876543210", "text": "weather", "message_id": "wamid.xyz"}]


def test_parse_incoming_messages_status_update_payload_has_no_messages():
    """A delivery-status webhook (sent/delivered/read receipts) has the same outer envelope but no
    'messages' key inside value - must return an empty list, not crash."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{"id": "WABA_ID", "changes": [{"value": {"statuses": [{"id": "wamid.abc", "status": "delivered"}]}, "field": "messages"}]}],
    }
    assert parse_incoming_messages(payload) == []


def test_parse_incoming_messages_completely_empty_payload():
    assert parse_incoming_messages({}) == []


def test_parse_incoming_messages_non_text_message_has_none_text():
    payload = _real_shaped_incoming_payload()
    payload["entry"][0]["changes"][0]["value"]["messages"][0] = {"from": "919876543210", "id": "wamid.img", "type": "image", "image": {"id": "media123"}}
    messages = parse_incoming_messages(payload)
    assert messages == [{"from": "919876543210", "text": None, "message_id": "wamid.img"}]
