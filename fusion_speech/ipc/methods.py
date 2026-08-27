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
        "wake_armed": daemon.wake_armed,
        "speaking": daemon.speaking,
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


async def handle_start_listening(params: dict, daemon) -> dict:
    # Start realtime mic capture + streaming ASR. speech.asr notifications are
    # pushed to /stream/stt WebSocket subscribers. Returns once capture starts.
    logger.info("speech.start_listening")
    return await daemon.start_listening()


async def handle_stop_listening(params: dict, daemon) -> dict:
    logger.info("speech.stop_listening")
    return await daemon.stop_listening()


async def handle_set_hotwords(params: dict, daemon) -> dict:
    words = params.get("hotwords") or []
    if not isinstance(words, list):
        raise ValueError("hotwords must be a list")
    accepted = daemon.set_hotwords(words)
    logger.info("speech.set_hotwords %s -> %s", words, accepted)
    return {"hotwords": accepted}


async def handle_wake(params: dict, daemon) -> dict:
    arm = bool(params.get("arm", True))
    logger.info("speech.wake arm=%s", arm)
    if arm:
        return await daemon.arm_wake()
    return await daemon.disarm_wake()


async def handle_speak(params: dict, daemon) -> dict:
    text = params.get("text", "")
    if not text:
        raise ValueError("text required")
    voice = params.get("voice", "") or daemon.config.tts_voice
    model = params.get("model", "") or _resolve_tts_model(daemon)
    logger.info("speech.speak chars=%d model=%s", len(text), model)
    return await daemon.speak(text, voice=voice, model=model)


async def handle_stop_tts(params: dict, daemon) -> dict:
    logger.info("speech.stop_tts")
    return await daemon.stop_tts()


def build_methods(daemon) -> dict:
    return {
        "speech.ping": lambda params: handle_ping(params, daemon),
        "speech.status": lambda params: handle_status(params, daemon),
        "speech.transcribe": lambda params: handle_transcribe(params, daemon),
        "speech.synthesize": lambda params: handle_synthesize(params, daemon),
        "speech.list_models": lambda params: handle_list_models(params, daemon),
        "speech.start_listening": lambda params: handle_start_listening(params, daemon),
        "speech.stop_listening": lambda params: handle_stop_listening(params, daemon),
        "speech.set_hotwords": lambda params: handle_set_hotwords(params, daemon),
        "speech.wake": lambda params: handle_wake(params, daemon),
        "speech.speak": lambda params: handle_speak(params, daemon),
        "speech.stop_tts": lambda params: handle_stop_tts(params, daemon),
    }
