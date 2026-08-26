from __future__ import annotations

import asyncio
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
        resp = await client.client.post(_STT_PATH, files=files, data=data)
        if resp.status_code != 200:
            msg = _extract_error(resp)
            logger.warning("transcribe failed: %d %s", resp.status_code, msg)
            raise AudioBridgeError(f"transcribe failed: {msg}", status=resp.status_code)
        payload = resp.json()
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
        logger.info("synthesize: model=%s voice=%s stream=%s chars=%d", tts_model, tts_voice, stream, len(text))
        async with client.client.stream("POST", _TTS_PATH, json=body) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                msg = _extract_error(httpx.Response(resp.status_code, content=raw))
                logger.warning("synthesize failed: %d %s", resp.status_code, msg)
                raise AudioBridgeError(f"synthesize failed: {msg}", status=resp.status_code)
            header_sent = False
            sample_rate = 0
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                if not header_sent:
                    if len(chunk) < _WAV_HEADER_LEN:
                        logger.warning("synthesize: first chunk shorter than WAV header (%d)", len(chunk))
                    else:
                        sample_rate = _parse_wav_sample_rate(chunk[:_WAV_HEADER_LEN])
                        logger.info("synthesize stream started: sample_rate=%d", sample_rate)
                    header_sent = True
                yield chunk
        logger.info("synthesize stream complete")
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
