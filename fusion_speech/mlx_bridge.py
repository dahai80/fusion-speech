from __future__ import annotations

import asyncio
import logging
from typing import Any

from fusion_core.mlx_client import FusionMLXClient

logger = logging.getLogger(__name__)


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
            import mlx_audio  # noqa: F401

            lane["tts"] = True
            lane["stt"] = True
            logger.info("audio lane probe: mlx_audio importable, lanes assumed ready")
        except ImportError as e:
            lane["error"] = f"mlx_audio missing: {e}"
            logger.warning("audio lane probe: mlx_audio not importable: %s", e)
        try:
            models = await probe_client.list_models()
            logger.info("audio lane probe: %d models available on fusion-mlx", len(models))
        except Exception as e:
            lane["error"] = (lane["error"] + f"; list_models failed: {e}").strip("; ")
            logger.warning("audio lane probe: list_models failed: %s", e)
    finally:
        await probe_client.close()
    return lane


async def transcribe_stream(config, pcm_bytes: bytes, *, language: str = "", model: str = "") -> str:
    raise NotImplementedError("transcribe_stream lands at M1")


async def synthesize_stream(config, text: str, *, voice: str = "", model: str = ""):
    raise NotImplementedError("synthesize_stream lands at M1")
    yield  # async generator marker; unreachable until M1


if __name__ == "__main__":
    from fusion_speech.config import load_config

    async def _run():
        cfg = load_config()
        print(await probe_mlx_audio_lane(cfg))

    asyncio.run(_run())
