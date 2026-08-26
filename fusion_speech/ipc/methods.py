from __future__ import annotations

import base64
import logging
import time

logger = logging.getLogger(__name__)


async def handle_ping(params: dict, daemon) -> dict:
    logger.debug("speech.ping id=%s", params.get("id"))
    return {"pong": True, "timestamp": time.time()}


async def handle_status(params: dict, daemon) -> dict:
    logger.debug("speech.status")
    return {
        "running": daemon.running,
        "socket_path": daemon.socket_path,
        "port": daemon.port,
        "uptime": time.time() - daemon.start_time if daemon.start_time else 0,
        "mlx_audio_lane": daemon.audio_lane,
        "stt_model": daemon.config.stt_model,
        "tts_model": daemon.config.tts_model,
        "listening": daemon.listening,
        "hotwords": list(daemon.hotwords),
    }


async def handle_transcribe(params: dict, daemon) -> dict:
    from fusion_speech.mlx_bridge import transcribe_stream

    audio_b64 = params.get("audio", "")
    if not audio_b64:
        raise ValueError("audio (base64) required")
    audio_bytes = base64.b64decode(audio_b64)
    language = params.get("language", "") or daemon.config.language
    model = params.get("model", "") or _resolve_stt_model(daemon)
    filename = params.get("filename", "audio.wav")
    logger.info("speech.transcribe bytes=%d model=%s", len(audio_bytes), model)
    result = await transcribe_stream(
        daemon.config,
        audio_bytes,
        filename=filename,
        language=language,
        model=model,
    )
    return result


async def handle_synthesize(params: dict, daemon) -> dict:
    from fusion_speech.mlx_bridge import synthesize_full

    text = params.get("text", "")
    if not text:
        raise ValueError("text required")
    voice = params.get("voice", "") or daemon.config.tts_voice
    model = params.get("model", "") or _resolve_tts_model(daemon)
    logger.info("speech.synthesize chars=%d model=%s", len(text), model)
    wav_bytes = await synthesize_full(daemon.config, text, voice=voice, model=model)
    return {"audio": base64.b64encode(wav_bytes).decode("ascii"), "format": "wav", "size": len(wav_bytes)}


def _resolve_tts_model(daemon) -> str:
    configured = daemon.config.tts_model
    available = daemon.audio_lane.get("tts_models") or []
    if configured in available or not available:
        return configured
    logger.warning("configured tts_model %r not in probed set, falling back to %r", configured, available[0])
    return available[0]


def _resolve_stt_model(daemon) -> str:
    configured = daemon.config.stt_model
    available = daemon.audio_lane.get("stt_models") or []
    if configured in available or not available:
        return configured
    logger.warning("configured stt_model %r not in probed set, falling back to %r", configured, available[0])
    return available[0]


async def handle_list_models(params: dict, daemon) -> dict:
    from fusion_speech.mlx_bridge import _build_client

    client = _build_client(daemon.config, timeout=10.0)
    try:
        models = await client.list_models()
    finally:
        await client.close()
    audio = [m.get("id", "") for m in models if m.get("modality") == "audio"]
    return {"total": len(models), "audio_models": audio}


def build_methods(daemon) -> dict:
    return {
        "speech.ping": lambda params: handle_ping(params, daemon),
        "speech.status": lambda params: handle_status(params, daemon),
        "speech.transcribe": lambda params: handle_transcribe(params, daemon),
        "speech.synthesize": lambda params: handle_synthesize(params, daemon),
        "speech.list_models": lambda params: handle_list_models(params, daemon),
    }
