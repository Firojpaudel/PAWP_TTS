"""Configurable HTTP multipart/form-data ASR evaluation adapter."""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from pawp_tts.domain.entities.models import AsrTranscript, LanguageCode


class HttpMultipartAsrAdapter:
    """
    Calls an external HTTP multipart ASR API configured via environment variables.
    All non-language metadata fields default to anonymized `"string"` placeholders.
    """

    def __init__(
        self,
        api_url: str | None = None,
        bucket_name: str | None = None,
        filename: str | None = None,
        user_id: str | None = None,
        organization: str | None = None,
        scope: str | None = None,
        service: str | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self._api_url = api_url or os.environ.get("PAWP_ASR_API_URL", "")
        self._bucket = bucket_name or os.environ.get("PAWP_ASR_BUCKET", "string")
        self._filename = filename or os.environ.get("PAWP_ASR_FILENAME", "string")
        self._user_id = user_id or os.environ.get("PAWP_ASR_USER_ID", "string")
        self._org = organization or os.environ.get("PAWP_ASR_ORG", "string")
        self._scope = scope or os.environ.get("PAWP_ASR_SCOPE", "string")
        self._service = service or os.environ.get("PAWP_ASR_SERVICE", "string")
        self._timeout = timeout_sec or float(os.environ.get("PAWP_ASR_TIMEOUT_SEC", "30.0"))

    def transcribe(
        self,
        audio_path: str,
        language: LanguageCode,
        context_words: str = "string",
    ) -> AsrTranscript:
        """POST an audio/wav file to the configured ASR endpoint and extract transcript text."""
        if not self._api_url:
            raise RuntimeError("PAWP_ASR_API_URL is not configured in environment.")

        lang_code = self._map_language(language)
        wav_path = Path(audio_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        form_data = {
            "bucket_name": self._bucket,
            "filename": self._filename,
            "user_id": self._user_id,
            "organization": self._org,
            "scope": self._scope,
            "service": self._service,
            "language": lang_code,
            "output_type": "text",
            "context_words": context_words,
        }

        t0 = time.perf_counter()
        with wav_path.open("rb") as fh, httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                self._api_url,
                data=form_data,
                files={"audio": ("audio.wav", fh, "audio/wav")},
                headers={"accept": "*/*"},
            )
            resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = self._extract_text(resp)
        return AsrTranscript(
            text=text,
            language=language,
            latency_ms=latency_ms,
            provider="http_asr_api",
        )

    @staticmethod
    def _map_language(language: LanguageCode) -> str:
        if language == LanguageCode.EN:
            return "eng"
        if language == LanguageCode.NE:
            return "nep"
        raise ValueError(f"ASR API supports EN ('eng') or NE ('nep'); got {language}")

    @staticmethod
    def _extract_text(resp: httpx.Response) -> str:
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = resp.json()
            if isinstance(payload, str):
                return payload.strip()
            if isinstance(payload, dict):
                for key in ("text", "transcript", "transcription", "result", "data"):
                    val = payload.get(key)
                    if isinstance(val, str):
                        return val.strip()
                    if isinstance(val, dict) and isinstance(val.get("text"), str):
                        return str(val["text"]).strip()
        return resp.text.strip().strip('"')
