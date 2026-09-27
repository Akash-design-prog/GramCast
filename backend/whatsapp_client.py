"""WhatsApp Cloud API client (Meta) - the farmer-facing channel per docs/ISSUES_PLAN.md item 4b.

Real request/response shapes below were verified against Meta's own current developer docs
(developers.facebook.com/docs/whatsapp/cloud-api/guides/send-messages/) and its webhook docs, not
guessed - same rigor as bhashini_tts.py's own reverse-engineering note.

Requires three environment variables (never hardcode real credentials in source):
  WHATSAPP_ACCESS_TOKEN    - the app's access token (temporary 24h token during test mode, or a
                             permanent System User token later)
  WHATSAPP_PHONE_NUMBER_ID - the test/production phone number's ID (not the phone number itself)
  WHATSAPP_VERIFY_TOKEN    - a string Akash makes up himself and enters in the Meta dashboard's
                             webhook config; main.py's GET /whatsapp/webhook checks incoming
                             hub.verify_token against this to confirm a verification request is real

is_configured() is the same switch pattern bhashini_tts.py already uses: callers check it before
attempting a real call, so a demo without real credentials yet fails predictably (WhatsAppError,
"not configured") instead of a confusing network error.
"""
import os

import requests

GRAPH_API_VERSION = "v21.0"
REQUEST_TIMEOUT_S = 15


class WhatsAppError(Exception):
    """Anything that goes wrong calling the WhatsApp Cloud API - missing config, network failure, or
    a non-2xx response. Caught as a single type by callers (backend/whatsapp_alerts.py, the webhook
    handler in main.py) so they don't need to know every possible failure mode."""


def is_configured() -> bool:
    return bool(os.environ.get("WHATSAPP_ACCESS_TOKEN")) and bool(os.environ.get("WHATSAPP_PHONE_NUMBER_ID"))


def _require_config() -> tuple[str, str]:
    token = os.environ.get("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
    if not token or not phone_number_id:
        raise WhatsAppError("WHATSAPP_ACCESS_TOKEN/WHATSAPP_PHONE_NUMBER_ID not set - call is_configured() first")
    return token, phone_number_id


def _messages_url(phone_number_id: str) -> str:
    return f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages"


def _post_message(body: dict) -> dict:
    token, phone_number_id = _require_config()
    try:
        resp = requests.post(
            _messages_url(phone_number_id),
            json=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise WhatsAppError(f"network error calling WhatsApp Cloud API: {e}") from e

    if resp.status_code >= 400:
        raise WhatsAppError(f"WhatsApp API returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def send_text_message(to: str, body: str) -> dict:
    """to: E.164 phone number WITHOUT a leading '+' (Meta's own convention - country code then number,
    e.g. '919876543210' for a +91 Indian number). Returns the raw API response (has a message id under
    messages[0].id on success)."""
    return _post_message({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": body},
    })


def send_template_message(to: str, template_name: str, language_code: str, body_params: list[str]) -> dict:
    """Template messages are the only way to message a farmer OUTSIDE the 24-hour customer-service
    window Meta enforces (i.e. any message we send first, not in reply to one of theirs) - and the
    template itself must already be created and APPROVED in the Meta dashboard before this call will
    work; this function can't create templates, only send an already-approved one. body_params fills
    the template's {{1}}, {{2}}, ... placeholders in order."""
    return _post_message({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
            "components": [{
                "type": "body",
                "parameters": [{"type": "text", "text": p} for p in body_params],
            }],
        },
    })


def upload_media(file_bytes: bytes, mime_type: str, filename: str = "voice_note") -> str:
    """Step 1 of sending a voice note: upload the audio bytes, get back a media id to reference in
    send_audio_message(). Meta's own limit is 100MB per file - GramCast's TTS clips are seconds long,
    nowhere near that, so no size check needed here."""
    token, phone_number_id = _require_config()
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/media"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            data={"messaging_product": "whatsapp"},
            files={"file": (filename, file_bytes, mime_type)},
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise WhatsAppError(f"network error uploading media to WhatsApp Cloud API: {e}") from e

    if resp.status_code >= 400:
        raise WhatsAppError(f"WhatsApp media upload returned {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    media_id = body.get("id")
    if not media_id:
        raise WhatsAppError(f"WhatsApp media upload response missing 'id': {body}")
    return media_id


def send_audio_message(to: str, media_id: str) -> dict:
    return _post_message({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "audio",
        "audio": {"id": media_id},
    })


def verify_webhook_challenge(mode: str | None, token: str | None, challenge: str | None) -> str:
    """Meta's webhook verification handshake (GET /whatsapp/webhook?hub.mode=subscribe&hub.verify_
    token=...&hub.challenge=...): must echo back `challenge` exactly if mode is 'subscribe' and token
    matches WHATSAPP_VERIFY_TOKEN, or Meta refuses to activate the webhook. Raises WhatsAppError
    (mapped to a 403 by main.py) for a mismatched token or wrong mode - never guess/accept blindly,
    since anyone could hit this endpoint."""
    expected_token = os.environ.get("WHATSAPP_VERIFY_TOKEN")
    if not expected_token:
        raise WhatsAppError("WHATSAPP_VERIFY_TOKEN not set - set it to whatever string you enter in the Meta dashboard")
    if mode != "subscribe" or token != expected_token:
        raise WhatsAppError("webhook verification failed: hub.mode or hub.verify_token did not match")
    if challenge is None:
        raise WhatsAppError("webhook verification failed: missing hub.challenge")
    return challenge


def parse_incoming_messages(payload: dict) -> list[dict]:
    """Extracts every real inbound message from a webhook POST body (the payload's real, documented
    shape: object.entry[].changes[].value.messages[]) into a flat list of {from, text, message_id}
    dicts - text is None for a non-text message (image/audio/etc, which this bot doesn't handle yet).
    Never raises on a malformed/partial payload (Meta's own retried-delivery and status-update payloads
    don't always carry a 'messages' key) - just returns fewer/no messages, since a webhook endpoint
    that 500s on an unexpected shape risks Meta disabling it after repeated failures."""
    results = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                text = msg.get("text", {}).get("body") if msg.get("type") == "text" else None
                results.append({"from": msg.get("from"), "text": text, "message_id": msg.get("id")})
    return results
