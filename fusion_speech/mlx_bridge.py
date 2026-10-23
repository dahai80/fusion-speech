from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fusion_core.mlx_client import FusionMLXClient

logger = logging.getLogger(__name__)

# FusionMLXClient.base_url already includes the /v1 prefix, so audio paths are
# relative to it (NOT /v1/audio/...). See fusion_core.config.default_mlx_base_url.
_STT_PATH = "/audio/transcriptions"
_TTS_PATH = "/audio/speech"
_TTS_TIMEOUT = 120.0
_WAV_HEADER_LEN = 44


class AudioBridgeError(RuntimeError):
    def __init__(self, message: str, *, status: int = 0):
        self.status = status
        super().__init__(message)


async def probe_mlx_audio_lane(config) -> dict:
    probe_client = FusionMLXClient(
        base_url=config.mlx_base_url,
        api_key=config.mlx_api_key,
        timeout=5.0,
    )
    lane: dict[str, Any] = {"tts": False, "stt": False, "probed": True, "error": ""}
    try:
        healthy = await probe_client.health()
        if not healthy:
            lane["error"] = "fusion-mlx unreachable"
            logger.warning("audio lane probe: fusion-mlx unreachable at %s", config.mlx_base_url)
            return lane
        try:
            models = await probe_client.list_models()
            stt_ids, tts_ids = _filter_audio_models(models)
            lane["tts"] = len(tts_ids) > 0
            lane["stt"] = len(stt_ids) > 0
            lane["tts_models"] = tts_ids
            lane["stt_models"] = stt_ids
            logger.info(
                "audio lane probe: %d models, stt=%d tts=%d",
                len(models),
                len(stt_ids),
                len(tts_ids),
            )
            if not stt_ids:
                lane["error"] = (lane["error"] + "; no STT model discovered").strip("; ")
            if not tts_ids:
                lane["error"] = (lane["error"] + "; no TTS model discovered").strip("; ")
        except Exception as e:
            lane["error"] = (lane["error"] + f"; list_models failed: {e}").strip("; ")
            logger.warning("audio lane probe: list_models failed: %s", e)
    finally:
        await probe_client.close()
    return lane


def _filter_audio_models(models: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    stt_ids: list[str] = []
    tts_ids: list[str] = []
    for m in models:
        mid = m.get("id", "")
        if not mid:
            continue
        low = mid.lower()
        if m.get("modality") == "audio" or any(k in low for k in ("whisper", "parakeet", "asr")):
            if any(k in low for k in ("tts", "kokoro", "chatterbox", "vibevoice", "voxcpm", "dia", "kitten")):
                tts_ids.append(mid)
            elif any(k in low for k in ("deepfilter", "denoise", "enhance", "separate")):
                # audio-utility models (denoise/enhancement/source-separation)
                # are neither STT nor TTS — classifying them as STT poisons the
                # probe-driven model fallback (400 from fusion-mlx)
                logger.debug("audio model %r classified as utility (neither stt nor tts)", mid)
            else:
                stt_ids.append(mid)
    return stt_ids, tts_ids


def _build_client(config, *, timeout: float) -> FusionMLXClient:
    return FusionMLXClient(
        base_url=config.mlx_base_url,
        api_key=config.mlx_api_key,
        timeout=timeout,
    )


def _extract_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        msg = body.get("error", body).get("message", "") if isinstance(body.get("error"), dict) else str(body)
        return msg or f"HTTP {resp.status_code}"
    except Exception:
        return f"HTTP {resp.status_code}"


_RETRY_BACKOFFS = (0.5, 1.0, 2.0)  # audit M1: retry transient mlx failures


def _retryable(exc: BaseException) -> bool:
    # Retry 5xx (encoded in AudioBridgeError.status), transport/timeout errors,
    # and JSONDecodeError (a 200 with a malformed/empty body is often a
    # transient mid-load hiccup from fusion-mlx). Never retry 4xx (client
    # error — model id, bad request).
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException, json.JSONDecodeError)):
        return True
    status = getattr(exc, "status", 0)
    return 500 <= status < 600


async def _with_retry(label: str, fn):
    # fn() -> awaitable; returns its result. Retries up to len(_RETRY_BACKOFFS)
    # times on _retryable errors, backing off between attempts.
    last_exc: BaseException | None = None
    for attempt in range(len(_RETRY_BACKOFFS) + 1):
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            if not _retryable(e) or attempt == len(_RETRY_BACKOFFS):
                raise
            delay = _RETRY_BACKOFFS[attempt]
            logger.warning("%s attempt %d failed (%r), retry in %.1fs", label, attempt + 1, e, delay)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def transcribe_stream(
    config,
    audio_bytes: bytes,
    *,
    filename: str = "audio.wav",
    language: str = "",
    model: str = "",
) -> dict:
    stt_model = model or config.stt_model
    lang = language or config.language
    client = _build_client(config, timeout=120.0)
    try:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": stt_model}
        if lang:
            data["language"] = lang
        logger.info("transcribe: model=%s lang=%s bytes=%d", stt_model, lang, len(audio_bytes))

        async def _post():
            r = await client.client.post(_STT_PATH, files=files, data=data)
            if r.status_code != 200:
                msg = _extract_error(r)
                logger.warning("transcribe failed: %d %s", r.status_code, msg)
                raise AudioBridgeError(f"transcribe failed: {msg}", status=r.status_code)
            return r

        resp = await _with_retry("transcribe", _post)
        # E3: a 200 with a non-JSON / malformed body is a silent failure —
        # resp.json() raises a bare JSONDecodeError with no status context.
        # Wrap so the client sees an explicit AudioBridgeError carrying status.
        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("transcribe: malformed response (status %d): %s", resp.status_code, e)
            raise AudioBridgeError(
                f"transcribe: malformed response (status {resp.status_code}): {e}",
                status=resp.status_code,
            ) from e
        logger.info("transcribe ok: text=%r", payload.get("text", "")[:80])
        return payload
    finally:
        await client.close()


async def synthesize_stream(
    config,
    text: str,
    *,
    voice: str = "",
    model: str = "",
    stream: bool = True,
    streaming_interval: float | None = None,
    ref_audio: str = "",
    ref_text: str = "",
) -> AsyncIterator[bytes]:
    tts_model = model or config.tts_model
    tts_voice = voice or config.tts_voice
    client = _build_client(config, timeout=_TTS_TIMEOUT)
    try:
        body: dict[str, Any] = {
            "model": tts_model,
            "input": text,
            "response_format": "wav",
            "stream": stream,
        }
        if tts_voice:
            body["voice"] = tts_voice
        if stream and streaming_interval is not None:
            body["streaming_interval"] = streaming_interval
        # zero-shot cloning params (issue #2 M-C): forwarded only when set so
        # backends without cloning support see a clean body. fusion-mlx
        # /audio/speech does not accept these yet (upstream gap, issue filed).
        if ref_audio:
            body["ref_audio"] = ref_audio
        if ref_text:
            body["ref_text"] = ref_text
        logger.info("synthesize: model=%s voice=%s stream=%s chars=%d", tts_model, tts_voice, stream, len(text))
        # Retry only the initial connection; once bytes start flowing, errors
        # propagate (re-playing mid-stream audio would duplicate playback).
        resp_cm = None
        opened = None

        async def _open():
            cm = client.client.stream("POST", _TTS_PATH, json=body)
            r = await cm.__aenter__()
            if r.status_code != 200:
                raw = await r.aread()
                msg = _extract_error(httpx.Response(r.status_code, content=raw))
                logger.warning("synthesize failed: %d %s", r.status_code, msg)
                await cm.__aexit__(None, None, None)
                raise AudioBridgeError(f"synthesize failed: {msg}", status=r.status_code)
            return cm, r

        resp_cm, opened = await _with_retry("synthesize.connect", _open)
        try:
            header_sent = False
            sample_rate = 0
            async for chunk in opened.aiter_bytes():
                if not chunk:
                    continue
                if not header_sent:
                    if len(chunk) < _WAV_HEADER_LEN:
                        logger.warning("synthesize: first chunk shorter than WAV header (%d)", len(chunk))
                    else:
                        sample_rate = _parse_wav_sample_rate(chunk[:_WAV_HEADER_LEN])
                        logger.info("synthesize stream started: sample_rate=%d", sample_rate)
                    header_sent = True
                    # E3: sample_rate=0 means the WAV header was malformed (not
                    # a real RIFF/WAVE, or a truncated/empty body returned with
                    # 200). Fail loud rather than silently streaming zero-rate
                    # PCM that plays as garbage or hangs the player.
                    if sample_rate == 0:
                        logger.warning("synthesize: malformed WAV (sample_rate=0), aborting stream")
                        raise AudioBridgeError("synthesize: malformed WAV (sample_rate=0)", status=200)
                yield chunk
            logger.info("synthesize stream complete")
        finally:
            await resp_cm.__aexit__(None, None, None)
    finally:
        await client.close()


def _parse_wav_sample_rate(header: bytes) -> int:
    try:
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return 0
        if header[12:16] != b"fmt ":
            return 0
        sample_rate = struct.unpack_from("<I", header, 24)[0]
        return int(sample_rate)
    except struct.error:
        return 0


async def synthesize_full(
    config,
    text: str,
    *,
    voice: str = "",
    model: str = "",
) -> bytes:
    collected = bytearray()
    async for chunk in synthesize_stream(config, text, voice=voice, model=model, stream=False):
        collected.extend(chunk)
    logger.info("synthesize_full: %d bytes", len(collected))
    return bytes(collected)


if __name__ == "__main__":
    from fusion_speech.config import load_config

    async def _run():
        cfg = load_config()
        print(await probe_mlx_audio_lane(cfg))

    asyncio.run(_run())
