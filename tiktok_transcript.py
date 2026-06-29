"""Client for the ScrapeCreators TikTok video transcript API.

Wraps GET https://api.scrapecreators.com/v1/tiktok/video/transcript and
returns the parsed transcript. The API key is read from the
SCRAPECREATORS_API_KEY environment variable (loaded from .env), matching the
credential-handling pattern used elsewhere in this project.
"""

import os
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Constants
TRANSCRIPT_API_URL = "https://api.scrapecreators.com/v1/tiktok/video/transcript"
DEFAULT_TIMEOUT_SECONDS = 30

# Valid 2-letter language codes documented by the API.
SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "it", "ja", "ko", "zh"}


class TikTokTranscriptError(Exception):
    """Raised when the transcript cannot be fetched or parsed."""


def get_tiktok_transcript(
    url,
    api_key=None,
    language=None,
    use_ai_as_fallback=False,
    timeout=DEFAULT_TIMEOUT_SECONDS,
):
    """Fetch the transcript for a TikTok video.

    Args:
        url: The TikTok video URL (required).
        api_key: ScrapeCreators API key. Falls back to the
            SCRAPECREATORS_API_KEY environment variable.
        language: Optional 2-letter language code (e.g. 'en', 'es').
        use_ai_as_fallback: If True, use AI as a fallback when no transcript
            is found. Costs 10 credits and only works for videos under 2
            minutes.
        timeout: Request timeout in seconds.

    Returns:
        A dict with the parsed JSON response, e.g.:
            {"id": ..., "url": ..., "transcript": "WEBVTT\\n\\n..."}

    Raises:
        ValueError: If required arguments are missing or invalid.
        TikTokTranscriptError: If the request fails or the response is invalid.
    """
    if not url:
        raise ValueError("'url' is required")

    api_key = api_key or os.getenv("SCRAPECREATORS_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is required. Pass api_key= or set SCRAPECREATORS_API_KEY."
        )

    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
        )

    headers = {"x-api-key": api_key}

    # Only send optional params when set so we don't override API defaults.
    params = {"url": url}
    if language is not None:
        params["language"] = language
    if use_ai_as_fallback:
        params["use_ai_as_fallback"] = "true"

    try:
        response = requests.get(
            TRANSCRIPT_API_URL,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise TikTokTranscriptError(
            f"Request timed out after {timeout}s"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        # Surface the API's error body when available for easier debugging.
        status = response.status_code
        detail = _safe_error_detail(response)
        raise TikTokTranscriptError(
            f"API returned HTTP {status}: {detail}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise TikTokTranscriptError(f"Request failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise TikTokTranscriptError(
            "Response was not valid JSON: " + response.text[:200]
        ) from exc


def _safe_error_detail(response):
    """Best-effort extraction of an error message from a failed response."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or "<empty response body>"
    if isinstance(body, dict):
        return body.get("error") or body.get("message") or str(body)
    return str(body)


if __name__ == "__main__":
    # Simple CLI for manual testing:
    #   python tiktok_transcript.py <tiktok_url> [language]
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tiktok_transcript.py <tiktok_url> [language]")
        raise SystemExit(1)

    video_url = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = get_tiktok_transcript(video_url, language=lang)
    except (ValueError, TikTokTranscriptError) as err:
        print(f"Error: {err}")
        raise SystemExit(1)

    print(f"Video ID:   {result.get('id')}")
    print(f"Video URL:  {result.get('url')}")
    if "credits_remaining" in result:
        print(f"Credits:    {result.get('credits_remaining')}")
    print("Transcript:")

    transcript = result.get("transcript")
    if transcript:
        print(transcript)
    else:
        # A successful request can still return transcript: null when the
        # video has no captions. The AI fallback can generate one.
        print("<no transcript available for this video>")
        print(
            "Tip: retry with use_ai_as_fallback=True to AI-generate one "
            "(costs 10 credits, videos under 2 minutes only)."
        )
