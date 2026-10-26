from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from fusion_speech.engines import ASREngine, TTSEngine
from fusion_speech.metrics import metrics

logger = logging.getLogger(__name__)

# Issue #4: pluggable cloud engine layer. Cloud engines implement the same
# ASREngine/TTSEngine contracts as the local (fusion-mlx) ones, plus a
# credential-provider hook (keys are issued by fusion-identity, not carried by
# business apps) and structured failure reporting for the audio job pipeline.

# Same backoff shape as mlx_bridge: transient transport/5xx only, never 4xx.
CLOUD_RETRY_BACKOFFS = (0.5, 1.0, 2.0)

_CredentialProvider = Callable[[], Awaitable[dict]]


class CloudEngineError(RuntimeError):
    # Structured failure for the audio job pipeline: `reason` is a stable slug
    # the caller can branch on (azure_auth / azure_unreachable / ...),
    # `retryable` tells the pipeline whether re-queuing makes sense. The
    # str(error) embeds the slug (reason=xxx retryable=...) because the UDS
    # JSON-RPC layer carries message-only errors — the exam pipeline regexes
    # the slug out of the message.
    def __init__(
        self,
        message: str,
        *,
        reason: str = "cloud_error",
        retryable: bool = False,
        status: int = 0,
        engine: str = "",
    ):
        self.reason = reason
        self.retryable = retryable
        self.status = status
        self.engine = engine
        super().__init__(f"reason={reason} retryable={retryable}: {message}")


class StaticCredentials:
    # Simplest provider: fixed key/region (env-config path). Refresh-capable
    # providers (fusion-identity token issuance) implement the same call shape
    # and are fetched per request, so rotated keys take effect without restart.
    def __init__(self, key: str, region: str):
        self.key = key
        self.region = region

    async def __call__(self) -> dict:
        if not self.key or not self.region:
            raise CloudEngineError(
                "azure credentials not configured (set FUSION_SPEECH_AZURE_KEY / FUSION_SPEECH_AZURE_REGION)",
                reason="azure_not_configured",
                retryable=False,
                engine="azure",
            )
        return {"key": self.key, "region": self.region}


def config_azure_credentials(config) -> _CredentialProvider:
    return StaticCredentials(getattr(config, "azure_key", ""), getattr(config, "azure_region", ""))


async def cloud_retry(label: str, fn):
    # fn() -> awaitable. Retries transient failures with the same backoff
    # cadence as the mlx bridge (0.5/1/2s), up to 3 retries.
    for attempt in range(len(CLOUD_RETRY_BACKOFFS) + 1):
        try:
            return await fn()
        except Exception as e:
            retryable = getattr(e, "retryable", False)
            if not retryable or attempt == len(CLOUD_RETRY_BACKOFFS):
                raise
            delay = CLOUD_RETRY_BACKOFFS[attempt]
            logger.warning("%s attempt %d failed (%r), retry in %.1fs", label, attempt + 1, e, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


@contextlib.asynccontextmanager
async def cloud_call(kind: str, engine_name: str) -> AsyncIterator[None]:
    # Metrics envelope for one cloud inference call: request counter on entry,
    # failure counter with the stable reason slug on the way out.
    metrics.cloud_requests_total.inc(1, {"engine": engine_name, "kind": kind})
    try:
        yield
    except CloudEngineError as e:
        metrics.cloud_errors_total.inc(1, {"engine": engine_name, "kind": kind, "reason": e.reason})
        raise
    except Exception:
        metrics.cloud_errors_total.inc(1, {"engine": engine_name, "kind": kind, "reason": "internal"})
        raise


class CloudSTTEngine(ASREngine):
    # Cloud ASR base: batch convenience over the streaming contract. finish()
    # results carry alignment + dialect metadata on top of the plain text:
    #   words: [{word, offset_ms, duration_ms, confidence}]
    #   dialect / dialect_detected: LID locale (e.g. ar-EG) when the backend
    #   ran language identification — the exam pipeline raises human-review
    #   priority on Arabic dialects.
    def __init__(self, *, on_event: Callable[[dict], Awaitable[None]] | None = None):
        super().__init__(on_event=on_event)

    async def transcribe_wav(self, wav_bytes: bytes, *, language: str = "") -> dict:
        pcm = wav_to_pcm(wav_bytes)
        await self.start_stream(language=language)
        await self.feed(pcm)
        return await self.finish()


class CloudTTSEngine(TTSEngine):
    # Cloud TTS base: marker + shared WAV-header sanity helper. Concrete
    # engines yield a RIFF WAV byte stream exactly like the local engine, so
    # StreamingPlayer and the RPC synthesize path need no branching.
    def __init__(self):
        pass


def wav_to_pcm(wav_bytes: bytes) -> np.ndarray:
    # Minimal RIFF parser (pcm_to_wav's inverse) for the batch RPC path: the
    # client uploads a WAV; cloud engines need raw float32 PCM. Enforces the
    # pipeline contract (16 kHz mono 16-bit) — anything else is a caller bug,
    # surfaced as a structured non-retryable error.
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise CloudEngineError("not a RIFF/WAVE file", reason="invalid_audio", retryable=False)
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(wav_bytes):
        cid = wav_bytes[pos : pos + 4]
        size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
        body = wav_bytes[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
        if fmt is not None and data is not None:
            break
    if fmt is None or data is None or len(fmt) < 16:
        raise CloudEngineError("WAV missing fmt/data chunk", reason="invalid_audio", retryable=False)
    channels, sample_rate, bits = (
        struct.unpack_from("<HII", fmt, 2)[0],
        struct.unpack_from("<I", fmt, 4)[0],
        struct.unpack_from("<H", fmt, 14)[0],
    )
    if sample_rate != 16000 or channels != 1 or bits != 16:
        raise CloudEngineError(
            f"WAV must be 16 kHz mono 16-bit (pipeline contract), got {sample_rate} Hz {channels}ch {bits}-bit",
            reason="invalid_audio",
            retryable=False,
        )
    if len(data) % 2 != 0:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
