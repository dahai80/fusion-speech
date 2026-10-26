from __future__ import annotations

import asyncio
import importlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from xml.sax.saxutils import escape as xml_escape

import httpx
import numpy as np

from fusion_speech.cloud import (
    CloudEngineError,
    CloudSTTEngine,
    CloudTTSEngine,
    cloud_call,
    cloud_retry,
    config_azure_credentials,
)
from fusion_speech.engines import register_asr_engine, register_tts_engine
from fusion_speech.mlx_bridge import _parse_wav_sample_rate

logger = logging.getLogger(__name__)

# Azure AI Speech (issue #4). STT uses the azure-cognitiveservices-speech SDK
# (push-stream + one-shot recognize) because word-level alignment and language
# ID are SDK/batch-only features — the short-audio REST API returns neither.
# TTS is plain REST (httpx, already a core dep): one POST returns a RIFF WAV.
# Both fetch credentials from the provider PER CALL so fusion-identity-issued
# keys rotate without a daemon restart.

_TTS_TIMEOUT = 120.0
# Matches the daemon pipeline contract (16 kHz mono) so StreamingPlayer and
# the WS sink consume azure TTS audio with no resampling.
_TTS_OUTPUT_FORMAT = "riff-16khz-16bit-mono-pcm"
_WAV_HEADER_LEN = 44

# CancellationErrorCode name -> (reason slug, retryable). Everything not
# listed falls back to (azure_failed, non-retryable) — unknown codes must not
# put the job pipeline into a re-queue loop.
_STT_ERROR_MAP = {
    "AuthenticationFailure": ("azure_auth", False),
    "ConnectionFailure": ("azure_unreachable", True),
    "TimeoutFailure": ("azure_timeout", True),
    "ServiceError": ("azure_service", True),
    "TooManyRequests": ("azure_rate_limited", True),
    "BadRequest": ("azure_bad_request", False),
}

# HTTP status -> (reason slug, retryable) for the TTS REST route.
_TTS_STATUS_MAP = {
    401: ("azure_auth", False),
    403: ("azure_auth", False),
    429: ("azure_rate_limited", True),
    400: ("azure_bad_request", False),
}


def parse_language_candidates(raw: str) -> list[str]:
    # "ar-EG,ar-SA,en-US" -> ["ar-EG", "ar-SA", "en-US"]; empty -> [] (fixed
    # language mode, no LID). Dialect detection (exam G1) REQUIRES the
    # candidate list — LID is what tells ar-EG from ar-SA.
    return [t.strip() for t in raw.split(",") if t.strip()]


class AzureSTTEngine(CloudSTTEngine):
    # Buffers PCM until finish() (same shape as LocalASREngine), then pushes
    # the whole utterance into the SDK push stream and recognizes once. Word
    # alignment + LID come from the service's detailed JSON result; retry
    # rebuilds the recognizer from the retained buffer, so transient
    # connection failures re-play the full audio safely.
    name = "azure"

    def __init__(
        self,
        config,
        *,
        credentials: Callable[[], Awaitable[dict]] | None = None,
        languages: list[str] | None = None,
        sdk=None,
        on_event=None,
    ):
        super().__init__(on_event=on_event)
        self.config = config
        self._credentials = credentials or config_azure_credentials(config)
        self._languages = languages or []
        self._sdk = sdk
        self._chunks: list[np.ndarray] = []
        self._started = False
        self._language = ""

    def _load_sdk(self):
        if self._sdk is not None:
            return self._sdk
        try:
            # Optional extra (pyproject [project.optional-dependencies].azure).
            # Import failure is a structured non-retryable error, not a crash —
            # the daemon stays up and the local engine keeps working.
            return importlib.import_module("azure.cognitiveservices.speech")
        except ImportError as e:
            raise CloudEngineError(
                f"azure-cognitiveservices-speech not installed (pip install fusion-speech[azure]): {e}",
                reason="azure_sdk_missing",
                retryable=False,
                engine=self.name,
            ) from e

    async def start_stream(self, *, language: str = "") -> None:
        self._chunks = []
        self._language = language or getattr(self.config, "azure_language", "en-US")
        self._started = True
        logger.info("azure stt stream start lang=%s lid=%s", self._language, bool(self._languages))
        await self._emit({"event": "speech.asr.engine", "engine": self.name, "state": "listening"})

    async def feed(self, pcm: np.ndarray) -> None:
        if not self._started:
            raise CloudEngineError("asr engine stream not started", reason="azure_not_started", retryable=False)
        if pcm.size:
            self._chunks.append(np.asarray(pcm, dtype=np.float32).reshape(-1))

    async def cancel(self) -> None:
        self._started = False
        self._chunks = []
        logger.info("azure stt stream cancelled")

    async def finish(self) -> dict:
        if not self._started:
            raise CloudEngineError("asr engine stream not started", reason="azure_not_started", retryable=False)
        self._started = False
        pcm = np.concatenate(self._chunks) if self._chunks else np.empty(0, dtype=np.float32)
        self._chunks = []
        duration = round(pcm.size / self.config.sample_rate, 3)
        if pcm.size == 0:
            logger.info("azure stt finish: empty stream")
            return {"text": "", "language": self._language, "duration": 0.0, "engine": self.name}
        pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        await self._emit(
            {"event": "speech.asr.engine", "engine": self.name, "state": "transcribing", "duration": duration}
        )
        async with cloud_call("stt", self.name):
            creds = await self._credentials()
            result = await cloud_retry(
                "azure stt",
                lambda: asyncio.to_thread(self._recognize, creds, pcm16),
            )
        result.setdefault("duration", duration)
        logger.info("azure stt finish: %d samples -> %r", pcm.size, (result.get("text") or "")[:80])
        return result

    def _recognize(self, creds: dict, pcm16: bytes) -> dict:
        # Runs in a worker thread (SDK calls block). Builds a fresh recognizer
        # per attempt — retry-safe because pcm16 is retained by the caller.
        sdk = self._load_sdk()
        speech_config = sdk.SpeechConfig(subscription=creds["key"], region=creds["region"])
        # Word-level timestamps come back in the detailed JSON result.
        request_word_timestamps = getattr(speech_config, "request_word_level_timestamps", None)
        if callable(request_word_timestamps):
            speech_config.request_word_level_timestamps()
        push_stream = sdk.audio.PushAudioInputStream()
        audio_config = sdk.audio.AudioConfig(stream=push_stream)
        if self._languages:
            auto_detect = sdk.languageconfig.AutoDetectSourceLanguageConfig(languages=self._languages)
            recognizer = sdk.SpeechRecognizer(speech_config, auto_detect, audio_config)
        else:
            recognizer = sdk.SpeechRecognizer(
                speech_config=speech_config, language=self._language, audio_config=audio_config
            )
        try:
            push_stream.write(pcm16)
            push_stream.close()
            result = recognizer.recognize_once_async().get()
        except Exception as e:
            raise CloudEngineError(
                f"azure stt transport failure: {e}", reason="azure_unreachable", retryable=True, engine=self.name
            ) from e
        return self._result_to_dict(sdk, result)

    def _result_to_dict(self, sdk, result) -> dict:
        reason = getattr(result, "reason", None)
        if reason == getattr(sdk.ResultReason, "NoMatch", -1):
            logger.info("azure stt: no speech matched")
            return {"text": "", "language": self._language, "no_match": True, "engine": self.name}
        if reason != getattr(sdk.ResultReason, "RecognizedSpeech", 1):
            self._raise_canceled(sdk, result)
        raw_json = ""
        prop_id = getattr(sdk.PropertyId, "SpeechServiceResponse_JsonResult", None)
        if prop_id is not None:
            raw_json = result.properties.get(prop_id) or ""
        detail: dict = {}
        if raw_json:
            try:
                detail = json.loads(raw_json)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("azure stt: malformed JsonResult (%s), words/dialect unavailable", e)
        detected = ""
        lid_prop = getattr(sdk.PropertyId, "SpeechServiceResponse_AutoDetectDetectedLanguage", None)
        if lid_prop is not None:
            detected = result.properties.get(lid_prop) or ""
        if not detected and isinstance(detail.get("PrimaryLanguage"), dict):
            detected = detail["PrimaryLanguage"].get("Language") or ""
        words = self._extract_words(detail)
        dialect = detected if self._languages and detected else ""
        return {
            "text": getattr(result, "text", "") or "",
            "language": detected or self._language,
            "duration": 0.0,
            "engine": self.name,
            "words": words,
            "dialect": dialect,
            "dialect_detected": bool(dialect),
        }

    @staticmethod
    def _extract_words(detail: dict) -> list[dict]:
        nbest = detail.get("NBest")
        if not isinstance(nbest, list) or not nbest:
            return []
        raw_words = nbest[0].get("Words")
        if not isinstance(raw_words, list):
            return []
        words = []
        for w in raw_words:
            if not isinstance(w, dict):
                continue
            # Azure offsets are 100 ns ticks; ms = ticks / 10 000.
            words.append(
                {
                    "word": w.get("Word", ""),
                    "offset_ms": round(w.get("Offset", 0) / 10000.0, 1),
                    "duration_ms": round(w.get("Duration", 0) / 10000.0, 1),
                    "confidence": w.get("Confidence"),
                }
            )
        return words

    def _raise_canceled(self, sdk, result) -> None:
        details_cls = getattr(sdk, "CancellationDetails", None)
        error_code = ""
        message = "recognition canceled"
        if details_cls is not None:
            details = details_cls(result)
            # Real SDK returns a CancellationErrorCode enum: str() is
            # "CancellationErrorCode.AuthenticationFailure" — normalize to the
            # bare member name so the map stays enum-repr-independent.
            error_code = str(getattr(details, "error_code", "") or "").rsplit(".", 1)[-1]
            message = str(getattr(details, "error_details", "") or message)
        reason_slug, retryable = _STT_ERROR_MAP.get(error_code, ("azure_failed", False))
        raise CloudEngineError(
            f"azure stt failed ({error_code or 'unknown'}): {message}",
            reason=reason_slug,
            retryable=retryable,
            engine=self.name,
        )


class AzureTTSEngine(CloudTTSEngine):
    # REST synthesis: POST SSML, stream the returned RIFF WAV chunk by chunk
    # (same byte-stream contract as the local engine). Retry only the initial
    # connection — mid-stream failures propagate (re-playing audio would
    # duplicate playback).
    name = "azure"

    def __init__(self, config, *, credentials: Callable[[], Awaitable[dict]] | None = None, client_factory=None):
        self.config = config
        self._credentials = credentials or config_azure_credentials(config)
        self._client_factory = client_factory
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def _make_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(timeout=_TTS_TIMEOUT)

    def _build_ssml(self, text: str, voice: str) -> str:
        # Voice name carries the locale (en-US-AriaNeural -> xml:lang en-US).
        lang = "-".join(voice.split("-")[:2]) if "-" in voice else "en-US"
        return (
            f"<speak version='1.0' xml:lang='{lang}' xmlns='http://www.w3.org/2001/10/synthesis'>"
            f"<voice name='{xml_escape(voice)}'>{xml_escape(text)}</voice></speak>"
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        model: str = "",
        ref_audio: str = "",
        ref_text: str = "",
    ) -> AsyncIterator[bytes]:
        return self._run(text, voice)

    async def _run(self, text: str, voice: str) -> AsyncIterator[bytes]:
        async with cloud_call("tts", self.name):
            creds = await self._credentials()
            tts_voice = voice or getattr(self.config, "azure_tts_voice", "en-US-AriaNeural")
            ssml = self._build_ssml(text, tts_voice)
            url = f"https://{creds['region']}.tts.speech.microsoft.com/cognitiveservices/v1"
            headers = {
                "Ocp-Apim-Subscription-Key": creds["key"],
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": _TTS_OUTPUT_FORMAT,
                "User-Agent": "fusion-speech",
            }
            logger.info("azure tts: voice=%s chars=%d", tts_voice, len(text))
            client = self._make_client()
            self._active = True
            try:

                async def _open():
                    cm = client.stream("POST", url, headers=headers, content=ssml.encode("utf-8"))
                    r = await cm.__aenter__()
                    if r.status_code != 200:
                        raw = await r.aread()
                        await cm.__aexit__(None, None, None)
                        reason_slug, retryable = _TTS_STATUS_MAP.get(
                            r.status_code, ("azure_service", 500 <= r.status_code < 600)
                        )
                        raise CloudEngineError(
                            f"azure tts failed: HTTP {r.status_code} {raw[:200]!r}",
                            reason=reason_slug,
                            retryable=retryable,
                            status=r.status_code,
                            engine=self.name,
                        )
                    return cm, r

                resp_cm, opened = await cloud_retry("azure tts.connect", _open)
                try:
                    header_sent = False
                    async for chunk in opened.aiter_bytes():
                        if not chunk:
                            continue
                        if not header_sent:
                            if _parse_wav_sample_rate(chunk[:_WAV_HEADER_LEN]) != 16000:
                                raise CloudEngineError(
                                    "azure tts: response is not a 16 kHz RIFF WAV",
                                    reason="azure_invalid_response",
                                    retryable=False,
                                    engine=self.name,
                                )
                            header_sent = True
                        yield chunk
                    if not header_sent:
                        raise CloudEngineError(
                            "azure tts: empty response body",
                            reason="azure_invalid_response",
                            retryable=True,
                            engine=self.name,
                        )
                    logger.info("azure tts stream complete")
                finally:
                    await resp_cm.__aexit__(None, None, None)
            finally:
                self._active = False
                await client.aclose()

    async def cancel(self) -> None:
        # Real abort is the consumer closing the iterator; this marks intent
        # (parity with LocalTTSEngine).
        self._active = False
        logger.info("azure tts engine cancel requested")


def register_azure_engines(credentials: Callable[[], Awaitable[dict]] | None = None) -> None:
    # Registers "azure" ASR/TTS factories. `credentials` is the fusion-identity
    # hook: business-side embedding passes a provider that fetches fresh keys
    # per call; the daemon's default (None) reads FUSION_SPEECH_AZURE_KEY /
    # _REGION from config. Re-registration overwrites the previous factory.
    def _asr_factory(config, **kwargs):
        langs = parse_language_candidates(getattr(config, "azure_stt_languages", ""))
        return AzureSTTEngine(config, credentials=credentials, languages=langs, **kwargs)

    def _tts_factory(config, **kwargs):
        return AzureTTSEngine(config, credentials=credentials, **kwargs)

    register_asr_engine("azure", _asr_factory)
    register_tts_engine("azure", _tts_factory)
    logger.info("azure engines registered (credentials=%s)", "custom" if credentials is not None else "config-env")
