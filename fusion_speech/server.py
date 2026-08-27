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
        self.wake_armed = False
        self._wake = None  # WakeWordEngine, lazily built
        self.speaking = False
        self._player = None  # StreamingPlayer, lazily built
        self._server: asyncio.Server | None = None
        self._app: FastAPI | None = None
        self._http_task: asyncio.Task | None = None
        self._dispatcher = JsonRpcDispatcher(build_methods(self))
        self._listener = None  # StreamingListener, lazily built
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        logger.info("asr subscriber added (total=%d)", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)
        logger.info("asr subscriber removed (total=%d)", len(self._subscribers))

    async def broadcast(self, payload: dict) -> None:
        # fan-out speech.asr notifications to all WS subscribers. A full queue
        # (slow client) drops the oldest frame rather than blocking the capture
        # pipeline.
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except asyncio.QueueEmpty:
                    pass

    async def start_listening(self, audio_source=None) -> dict:
        from fusion_speech.mlx_bridge import transcribe_stream
        from fusion_speech.stt_stream import StreamingListener

        if self._listener is not None and self._listener.active:
            return {"listening": True, "already": True}
        self._listener = StreamingListener(
            self.config,
            transcribe=transcribe_stream,
            notifier=self.broadcast,
            audio_source=audio_source,
        )
        try:
            result = await self._listener.start()
        except Exception as e:
            self._listener = None
            self.listening = False
            logger.error("start_listening failed: %s", e)
            raise
        self.listening = True
        return result

    async def stop_listening(self) -> dict:
        listener = self._listener
        self.listening = False
        if listener is None:
            return {"listening": False}
        result = await listener.stop()
        self._listener = None
        return result

    async def arm_wake(self) -> dict:
        from fusion_speech.wake_word import WakeWordEngine

        if self._wake is not None and self._wake.active:
            return {"armed": True, "already": True}
        # share the mic stream with an active listener if one exists, else wake
        # builds + owns its own AudioSource.
        source = self._listener.source if (self._listener and self._listener.active) else None
        self._wake = WakeWordEngine(self.config, on_wake=self._on_wake, audio_source=source)
        if not self._wake.enabled:
            logger.warning("wake engine disabled, cannot arm")
            self.wake_armed = False
            return {"armed": False, "enabled": False}
        result = await self._wake.start()
        self.wake_armed = bool(result.get("armed"))
        if self.wake_armed:
            self.hotwords = list(self._wake.words)
        return result

    async def disarm_wake(self) -> dict:
        wake = self._wake
        self.wake_armed = False
        if wake is None:
            return {"armed": False}
        result = await wake.stop()
        self._wake = None
        return result

    async def speak(self, text: str, voice: str = "", model: str = "", *, synthesize=None, stream_factory=None) -> dict:
        from fusion_speech.tts_playback import StreamingPlayer

        if self._player is not None and self._player.active:
            logger.warning("speak rejected: already speaking")
            return {"played": False, "error": "already speaking"}
        player = StreamingPlayer(
            self.config, notifier=self.broadcast, synthesize=synthesize, stream_factory=stream_factory
        )
        self._player = player
        wake = self._wake
        was_listening = self._listener is not None and self._listener.active
        logger.info("speak chars=%d voice=%s was_listening=%s", len(text), voice, was_listening)
        if wake is not None and wake.active and not wake.paused:
            wake.pause()
        if was_listening:
            with contextlib.suppress(Exception):
                await self.stop_listening()
        try:
            self.speaking = True
            result = await player.speak(text, voice=voice, model=model)
        finally:
            self.speaking = False
            self._player = None
            if wake is not None and wake.active and wake.paused:
                try:
                    wake.resume()
                except Exception as e:
                    logger.warning("wake resume failed: %s", e)
            if was_listening:
                try:
                    await self.start_listening()
                except Exception as e:
                    logger.warning("start_listening resume failed: %s", e)
        return result

    async def stop_tts(self) -> dict:
        player = self._player
        logger.info("speech.stop_tts active=%s", player is not None)
        if player is None:
            return {"stopped": False}
        return await player.cancel()

    def set_hotwords(self, words: list[str]) -> list[str]:
        if self._wake is None:
            self.hotwords = list(words)
            return list(words)
        accepted = self._wake.set_hotwords(words)
        self.hotwords = list(accepted)
        return accepted

    async def _on_wake(self, word: str) -> None:
        # auto-listen handshake: announce the wake, pause detection (so the
        # user's following speech does not self-trigger), arm ASR for a fixed
        # window reusing wake's mic source, then stop ASR and resume wake.
        wake = self._wake
        if wake is None:
            return
        await self.broadcast({"event": "speech.wake", "word": word, "ts": time.time()})
        if self._listener is not None and self._listener.active:
            logger.info("wake fired but ASR already active, skip handshake")
            return
        wake.pause()
        try:
            source = wake.source if wake.own_source else None
            await self.start_listening(audio_source=source)
            await asyncio.sleep(self.config.wake_auto_listen_s)
        except asyncio.CancelledError:
            raise
        finally:
            await self.stop_listening()
            wake.resume()

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
        from fastapi import WebSocket, WebSocketDisconnect
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

        @app.websocket("/stream/stt")
        async def _stream_stt(ws: WebSocket):
            # Realtime ASR push channel. Client subscribes; server fans out
            # speech.asr frames (listening/speaking/transcribing/final/error)
            # from the active StreamingListener. Start/stop of capture is done
            # via RPC (speech.start_listening / stop_listening); the WS only
            # receives notifications.
            await ws.accept()
            q = self.subscribe()
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=30.0)
                    except TimeoutError:
                        await ws.send_json({"event": "speech.asr", "state": "keepalive"})
                        continue
                    await ws.send_json(payload)
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.exception("stream/stt websocket error")
            finally:
                self.unsubscribe(q)
                with contextlib.suppress(Exception):
                    await ws.close()

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
        if self._player is not None:
            logger.info("stopping TTS player on daemon stop")
            try:
                await self._player.cancel()
            except Exception as e:
                logger.warning("TTS player cancel on stop failed: %s", e)
            self._player = None
            self.speaking = False
        if self._wake is not None:
            with contextlib.suppress(Exception):
                await self._wake.stop()
            self._wake = None
            self.wake_armed = False
        if self._listener is not None:
            with contextlib.suppress(Exception):
                await self._listener.stop()
            self._listener = None
            self.listening = False
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
