from __future__ import annotations

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


def build_methods(daemon) -> dict:
    return {
        "speech.ping": lambda params: handle_ping(params, daemon),
        "speech.status": lambda params: handle_status(params, daemon),
    }
