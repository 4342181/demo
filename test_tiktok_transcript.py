"""Tests for tiktok_transcript.get_tiktok_transcript.

These mock the network layer (requests.get) so the suite runs with no API
key, no credits, and no outbound calls.
"""

import requests
import pytest

import tiktok_transcript
from tiktok_transcript import get_tiktok_transcript, TikTokTranscriptError


SAMPLE_RESPONSE = {
    "id": "7499229683859426602",
    "url": "https://www.tiktok.com/@stoolpresidente/video/7499229683859426602",
    "transcript": "WEBVTT\n\n00:00:00.120 --> 00:00:01.840\nAlright, pizza review time.\n",
}


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, json_data=None, text="", raise_json=False):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("No JSON object could be decoded")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("SCRAPECREATORS_API_KEY", "test-key")
    return "test-key"


def test_success_returns_parsed_json(monkeypatch, api_key):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured.update(url=url, headers=headers, params=params, timeout=timeout)
        return FakeResponse(200, json_data=SAMPLE_RESPONSE)

    monkeypatch.setattr(tiktok_transcript.requests, "get", fake_get)

    result = get_tiktok_transcript(SAMPLE_RESPONSE["url"], language="en")

    assert result == SAMPLE_RESPONSE
    # Endpoint, auth header, and required param are sent correctly.
    assert captured["url"] == tiktok_transcript.TRANSCRIPT_API_URL
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["params"]["url"] == SAMPLE_RESPONSE["url"]
    assert captured["params"]["language"] == "en"
    # Optional fallback not requested -> param omitted.
    assert "use_ai_as_fallback" not in captured["params"]


def test_fallback_param_only_sent_when_enabled(monkeypatch, api_key):
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured.update(params=params)
        return FakeResponse(200, json_data=SAMPLE_RESPONSE)

    monkeypatch.setattr(tiktok_transcript.requests, "get", fake_get)

    get_tiktok_transcript(SAMPLE_RESPONSE["url"], use_ai_as_fallback=True)
    assert captured["params"]["use_ai_as_fallback"] == "true"
    # Language omitted when not provided.
    assert "language" not in captured["params"]


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.delenv("SCRAPECREATORS_API_KEY", raising=False)
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured.update(headers=headers)
        return FakeResponse(200, json_data=SAMPLE_RESPONSE)

    monkeypatch.setattr(tiktok_transcript.requests, "get", fake_get)

    get_tiktok_transcript(SAMPLE_RESPONSE["url"], api_key="explicit-key")
    assert captured["headers"]["x-api-key"] == "explicit-key"


def test_missing_url_raises_value_error(api_key):
    with pytest.raises(ValueError, match="'url' is required"):
        get_tiktok_transcript("")


def test_missing_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("SCRAPECREATORS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        get_tiktok_transcript("https://example.com/video")


def test_invalid_language_raises_value_error(api_key):
    with pytest.raises(ValueError, match="Unsupported language"):
        get_tiktok_transcript("https://example.com/video", language="xx")


def test_http_error_includes_status_and_detail(monkeypatch, api_key):
    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResponse(401, json_data={"error": "Invalid API key"})

    monkeypatch.setattr(tiktok_transcript.requests, "get", fake_get)

    with pytest.raises(TikTokTranscriptError, match="HTTP 401: Invalid API key"):
        get_tiktok_transcript(SAMPLE_RESPONSE["url"])


def test_timeout_raises_transcript_error(monkeypatch, api_key):
    def fake_get(url, headers=None, params=None, timeout=None):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(tiktok_transcript.requests, "get", fake_get)

    with pytest.raises(TikTokTranscriptError, match="timed out"):
        get_tiktok_transcript(SAMPLE_RESPONSE["url"])


def test_connection_error_raises_transcript_error(monkeypatch, api_key):
    def fake_get(url, headers=None, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(tiktok_transcript.requests, "get", fake_get)

    with pytest.raises(TikTokTranscriptError, match="Request failed"):
        get_tiktok_transcript(SAMPLE_RESPONSE["url"])


def test_non_json_response_raises_transcript_error(monkeypatch, api_key):
    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResponse(200, raise_json=True, text="<html>not json</html>")

    monkeypatch.setattr(tiktok_transcript.requests, "get", fake_get)

    with pytest.raises(TikTokTranscriptError, match="not valid JSON"):
        get_tiktok_transcript(SAMPLE_RESPONSE["url"])
