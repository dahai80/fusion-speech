from __future__ import annotations

import base64
import logging
import time

from fusion_speech.metrics import metrics
from fusion_speech.mlx_bridge import _build_client, synthesize_full, transcribe_stream

logger = logging.getLogger(__name__)

# Bounds on client-supplied payloads (audit M2): a single RPC audio blob or a
# huge text could otherwise pin memory. dispatch() maps the raised ValueError
# to a JSON-RPC -32000 error the client sees.
MAX_AUDIO_B64 = 25 * 1024 * 1024  # ~25 MiB encoded, ~18 MiB decoded WAV
MAX_TEXT = 5000  # chars per speak/synthesize utterance


async def handle_ping(params: dict, daemon) -> dict:
    logger.debug("speech.ping id=%s", params.get("id"))
    return {"pong": True, "timestamp": time.time()}


async def handle_status(params: dict, daemon) -> dict:
    logger.debug("speech.status")
    rpc_snap = metrics.rpc_requests_total.snapshot()
    err_snap = metrics.rpc_errors_total.snapshot()
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
        "restart_count": getattr(daemon, "_restart_count", 0),
        "last_wake_word": daemon.state.last_wake_word if hasattr(daemon, "state") else "",
        "metrics": {
            "uptime_seconds": round(metrics.uptime_seconds(), 1),
            "rpc_requests_total": {str(k): v for k, v in rpc_snap.items()},
            "rpc_errors_total": {str(k): v for k, v in err_snap.items()},
            "asr_segments_total": metrics.asr_segments_total.snapshot().get((), 0),
            "wake_fires_total": metrics.wake_fires_total.snapshot().get((), 0),
            "tts_speak_total": metrics.tts_speak_total.snapshot().get((), 0),
        },
    }


async def handle_transcribe(params: dict, daemon) -> dict:
    audio_b64 = params.get("audio", "")
    if not audio_b64:
        raise ValueError("audio (base64) required")
    if len(audio_b64) > MAX_AUDIO_B64:
        raise ValueError(f"audio too large: {len(audio_b64)} > {MAX_AUDIO_B64} bytes")
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
    text = params.get("text", "")
    if not text:
        raise ValueError("text required")
    if len(text) > MAX_TEXT:
        raise ValueError(f"text too long: {len(text)} > {MAX_TEXT} chars")
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
    accepted = await daemon.set_hotwords(words)
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
    if len(text) > MAX_TEXT:
        raise ValueError(f"text too long: {len(text)} > {MAX_TEXT} chars")
    voice = params.get("voice", "") or daemon.config.tts_voice
    model = params.get("model", "") or _resolve_tts_model(daemon)
    logger.info("speech.speak chars=%d model=%s", len(text), model)
    return await daemon.speak(text, voice=voice, model=model)


async def handle_stop_tts(params: dict, daemon) -> dict:
    logger.info("speech.stop_tts")
    return await daemon.stop_tts()


async def handle_reprobe(params: dict, daemon) -> dict:
    # E1: explicitly re-probe fusion-mlx's audio lane. Refreshes audio_lane
    # (tts/stt availability + model lists) after models are loaded/unloaded
    # in fusion-mlx. Additive non-breaking RPC; returns the fresh lane.
    logger.info("speech.reprobe")
    lane = await daemon.probe_audio_lane()
    return {"audio_lane": lane}


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
        "speech.reprobe": lambda params: handle_reprobe(params, daemon),
    }
