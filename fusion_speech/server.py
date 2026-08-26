from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time

from fastapi import FastAPI

from fusion_speech import __version__
from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher, handle_client
from fusion_speech.ipc.methods import build_methods
from fusion_speech.logging import setup_logging

logger = logging.getLogger(__name__)


class SpeechDaemon:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.socket_path = self.config.socket_path
        self.port = self.config.port
        self.running = False
        self.start_time: float = 0.0
        self.audio_lane: dict = {"tts": False, "stt": False, "probed": False, "error": ""}
        self.listening = False
        self.hotwords: list[str] = []
        self._server: asyncio.Server | None = None
        self._app: FastAPI | None = None
        self._http_task: asyncio.Task | None = None
        self._dispatcher = JsonRpcDispatcher(build_methods(self))

    async def probe_audio_lane(self) -> dict:
        try:
            from fusion_speech.mlx_bridge import probe_mlx_audio_lane

            lane = await probe_mlx_audio_lane(self.config)
            self.audio_lane = lane
            logger.info(
                "audio lane probe: tts=%s stt=%s error=%s",
                lane.get("tts"),
                lane.get("stt"),
                lane.get("error", "") or "none",
            )
        except Exception as e:
            self.audio_lane = {"tts": False, "stt": False, "probed": False, "error": str(e)}
            logger.warning("audio lane probe failed: %s", e)
        return self.audio_lane

    def build_app(self) -> FastAPI:
        from fusion_core.http.app import create_app

        app = create_app("fusion-speech", version=__version__, enable_health=False)

        @app.get("/health")
        async def _health_detail():
            return {
                "status": "ok",
                "service": "fusion-speech",
                "version": __version__,
                "mlx_audio_lane": self.audio_lane,
                "stt_model": self.config.stt_model,
                "tts_model": self.config.tts_model,
                "listening": self.listening,
            }

        self._app = app
        return app

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._on_client, path=self.socket_path)
        os.chmod(self.socket_path, 0o600)
        logger.info("UDS listening on %s", self.socket_path)

        app = self.build_app()
        try:
            import uvicorn

            http_config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
            http_server = uvicorn.Server(http_config)
            self._http_task = asyncio.create_task(http_server.serve())
            logger.info("HTTP server starting on 127.0.0.1:%d", self.port)
        except Exception as e:
            logger.warning("HTTP server failed to start: %s", e)

        self.running = True
        self.start_time = time.time()
        await self.probe_audio_lane()
        logger.info("fusion-speech daemon started v%s", __version__)

    async def _on_client(self, reader, writer) -> None:
        await handle_client(reader, writer, self._dispatcher, lambda: self.running)

    async def stop(self) -> None:
        self.running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._http_task and not self._http_task.done():
            self._http_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._http_task
            self._http_task = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError as e:
                logger.warning("unlink socket %s failed: %s", self.socket_path, e)
        logger.info("fusion-speech daemon stopped")

    async def run_forever(self) -> None:
        await self.start()
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def _signal_handler():
            logger.info("shutdown signal received")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, _signal_handler)
        await stop_event.wait()
        await self.stop()


def main() -> None:
    cfg = load_config()
    setup_logging("fusion_speech", level=cfg.log_level, log_file=os.path.join(cfg.log_dir, "daemon.log"))
    daemon = SpeechDaemon(cfg)
    try:
        asyncio.run(daemon.run_forever())
    except KeyboardInterrupt:
        logger.info("interrupted by user")


if __name__ == "__main__":
    main()
