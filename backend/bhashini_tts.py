"""Bhashini (Digital India Bhashini Division / MeitY) text-to-speech client - the government-built
alternative to gTTS, per Akash's request to prefer government infrastructure for an SIH submission.

Two-call flow, per Bhashini's own published API docs (bhashini.gitbook.io/bhashini-apis) and a working
reference implementation (github.com/AdityaKukreti/bhashini-api), since Bhashini's own docs describe
the flow conceptually but don't quote exact endpoint URLs or header names on the same page:

1. Pipeline CONFIG call (https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline) -
   authenticated with the account-level userID/ulcaApiKey - asks "what TTS service exists for this
   language", gets back a serviceId, the actual compute endpoint URL, and a SEPARATE per-session
   inference API key (NOT the same key used for this call).
2. Pipeline COMPUTE call (URL from step 1's response, not hardcoded here - it's account/config
   dependent) - authenticated with the inference key from step 1 - actually does the TTS, returns
   base64-encoded audio (WAV) at pipelineResponse[0].audio[0].audioContent.

Requires two environment variables (never hardcode real credentials in source):
  BHASHINI_USER_ID, BHASHINI_API_KEY (the ulcaApiKey from the Bhashini dashboard's My Profile page)

Registration requires DIBD team approval (not instant) - see is_configured(), which is the switch the
hybrid wrapper in tts.py uses to decide whether to even attempt this before falling back to gTTS.
"""
import base64
import os

import requests

CONFIG_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
# Fixed pipeline ID for Bhashini's standard ASR/translation/TTS pipeline - the same constant used
# across every public reference implementation found, not account-specific.
PIPELINE_ID = "64392f96daac500b55c543cd"
REQUEST_TIMEOUT_S = 15


class BhashiniError(Exception):
    """Anything that goes wrong calling Bhashini - missing config, network failure, or an unexpected
    response shape. Caught as a single type by tts.py's hybrid wrapper so it can fall back to gTTS
    without needing to know every possible failure mode."""


def is_configured() -> bool:
    return bool(os.environ.get("BHASHINI_USER_ID")) and bool(os.environ.get("BHASHINI_API_KEY"))


def _get_pipeline_config(lang: str) -> dict:
    user_id = os.environ.get("BHASHINI_USER_ID")
    api_key = os.environ.get("BHASHINI_API_KEY")
    if not user_id or not api_key:
        raise BhashiniError("BHASHINI_USER_ID/BHASHINI_API_KEY not set - call is_configured() first")

    body = {
        "pipelineTasks": [{"taskType": "tts", "config": {"language": {"sourceLanguage": lang}}}],
        "pipelineRequestConfig": {"pipelineId": PIPELINE_ID},
    }
    try:
        resp = requests.post(
            CONFIG_URL,
            json=body,
            headers={"userID": user_id, "ulcaApiKey": api_key},
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise BhashiniError(f"pipeline config call failed: {e}") from e
    if resp.status_code != 200:
        raise BhashiniError(f"pipeline config call returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _extract_config_fields(config_response: dict) -> tuple[str, str, str, str]:
    """Returns (service_id, compute_url, auth_header_name, auth_header_value). Raises BhashiniError
    with the actual response attached in the message if the expected shape isn't there - a malformed
    or changed API response should fail loudly and specifically, not with a bare KeyError."""
    try:
        service_id = config_response["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        endpoint = config_response["pipelineInferenceAPIEndPoint"]
        compute_url = endpoint["callbackUrl"]
        auth_name = endpoint["inferenceApiKey"]["name"]
        auth_value = endpoint["inferenceApiKey"]["value"]
    except (KeyError, IndexError) as e:
        raise BhashiniError(f"unexpected pipeline config response shape (missing {e}): {config_response}") from e
    return service_id, compute_url, auth_name, auth_value


def _compute_tts(text: str, lang: str, service_id: str, compute_url: str, auth_name: str, auth_value: str) -> bytes:
    body = {
        "pipelineTasks": [
            {"taskType": "tts", "config": {"language": {"sourceLanguage": lang}, "serviceId": service_id, "gender": "female"}}
        ],
        "inputData": {"input": [{"source": text}]},
    }
    try:
        resp = requests.post(compute_url, json=body, headers={auth_name: auth_value}, timeout=REQUEST_TIMEOUT_S)
    except requests.RequestException as e:
        raise BhashiniError(f"pipeline compute call failed: {e}") from e
    if resp.status_code != 200:
        raise BhashiniError(f"pipeline compute call returned {resp.status_code}: {resp.text[:500]}")

    try:
        audio_b64 = resp.json()["pipelineResponse"][0]["audio"][0]["audioContent"]
    except (KeyError, IndexError) as e:
        raise BhashiniError(f"unexpected pipeline compute response shape (missing {e})") from e

    try:
        return base64.b64decode(audio_b64)
    except Exception as e:
        raise BhashiniError(f"could not base64-decode audioContent: {e}") from e


def synthesize_speech_bhashini(text: str, lang: str) -> bytes:
    """Returns WAV audio bytes for `text` spoken in `lang` via Bhashini. Raises BhashiniError on any
    failure - callers (tts.py's hybrid wrapper) are expected to catch this and fall back to gTTS."""
    config_response = _get_pipeline_config(lang)
    service_id, compute_url, auth_name, auth_value = _extract_config_fields(config_response)
    return _compute_tts(text, lang, service_id, compute_url, auth_name, auth_value)
