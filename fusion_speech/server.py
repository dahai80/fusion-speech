from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fusion_core.http.app import create_app

from fusion_speech import __version__
from fusion_speech.components.asr_service import AsrService
from fusion_speech.components.audio_bus import AudioBus
from fusion_speech.components.event_bus import EventBus
from fusion_speech.components.tts_service import TtsService
from fusion_speech.components.wake_service import WakeService
from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher, handle_client
from fusion_speech.ipc.methods import build_methods
from fusion_speech.logging import setup_logging
from fusion_speech.metrics import metrics, render_prometheus
from fusion_speech.mlx_bridge import probe_mlx_audio_lane
from fusion_speech.state import DaemonState
from fusion_speech.ws_ingress import WsIngressSession

logger = logging.getLogger(__name__)

# R6: bounded drain on stop. Each leaf stop (player/wake/listener/UDS/http) is
# wrapped in wait_for(DRAIN_TIMEOUT) so a wedged component cannot hang shutdown.
# start.sh grace (E8, 15s) accommodates this + mlx latency. Module-level so
# tests monkeypatch it small (a real 10s wait would defeat the test).
DRAIN_TIMEOUT = 10.0


class SpeechDaemon:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.socket_path = self.config.socket_path
        self.port = self.config.port
        self.running = False
        self.start_time: float = 0.0
        self.audio_lane: dict = {"tts": False, "stt": False, "probed": False, "error": ""}
        self._server: asyncio.Server | None = None
        self._app: FastAPI | None = None
        self._http_task: asyncio.Task | None = None
        self._dispatcher = JsonRpcDispatcher(build_methods(self))
        # Issue #4: auto-register azure cloud engines when credentials are
        # provisioned (env-config path; keys issued via fusion-identity, never
        # committed). Fail-soft — absent creds keep the local engine default.
        if self.config.azure_key and self.config.azure_region:
            with contextlib.suppress(Exception):
                from fusion_speech.cloud_azure import register_azure_engines

                register_azure_engines()
        else:
            logger.info("azure credentials not configured, cloud engines stay unregistered")
        # Wave 2: five leaf services + facade. Services hold single-resource
        # state; facade holds orchestration + lifecycle only. Property
        # delegation keeps test patches landing on facade delegators.
        self.event_bus = EventBus(queue_maxsize=self.config.ws_subscriber_queue)
        self.audio_bus = AudioBus(self.config)
        self.asr = AsrService(self.config, self.broadcast, self.audio_bus)
        self.wake = WakeService(self.config, self.audio_bus, on_wake=self._on_wake)
        self.tts = TtsService(self.config, self.broadcast)
        self._wake_listen_done: asyncio.Event | None = None
        self._last_final_ts: float = 0.0
        # E6: persistent state across restarts (restart_count, uptime). Lazily
        # enter_start() in start(); not bumped on construction so tests that
        # build a daemon without starting don't dirty the real state file.
        self.state = DaemonState(self.socket_path)
        self._restart_count: int = 0
        # H5: serialize the auto-listen handshake so two rapid wakes cannot
        # interleave (check-then-act tear on _listener/_wake_listen_done).
        # Holds across start_listening/wait_for/stop_listening (asr lock is
        # acquired inside, never the reverse order -> no deadlock).
        self._handshake_lock = asyncio.Lock()
        # E1: background re-probe loop keeps audio_lane fresh (models may be
        # loaded/unloaded in fusion-mlx after daemon start). Cancelled in stop.
        self._probe_task: asyncio.Task | None = None

    # _subscribers stays a real mutable set on the facade so tests that
    # .add()/.clear()/len() keep working; it aliases the EventBus's subscriber
    # set. subscribe/unsubscribe/broadcast delegate to EventBus.
    @property
    def _subscribers(self) -> set[asyncio.Queue]:
        return self.event_bus._subscribers

    def subscribe(self) -> asyncio.Queue:
        return self.event_bus.subscribe()

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.event_bus.unsubscribe(q)

    async def broadcast(self, payload: dict) -> None:
        await self.event_bus.publish(payload)

    def _on_handshake_final(self, payload: dict) -> None:
        # ASR-final side effect moved out of broadcast (R3 structural root):
        # this fires out-of-band via EventBus's _asr_final_handler, NOT through
        # subscriber Queue backpressure, so a slow TTS subscriber's drop-oldest
        # cannot eat the handshake frame that ends the auto-listen window.
        if self._wake_listen_done is None:
            return
        now = time.monotonic()
        if now - self._last_final_ts >= self.config.wake_debounce_s:
            self._wake_listen_done.set()
        self._last_final_ts = now

    # --- facade property delegation (Wave 2 H1/H3/H4) ---
    # Single-resource state lives on the leaf services; these rw properties
    # keep tests that read/assign daemon._listener, daemon.listening, etc. on
    # the facade hitting the real service state.
    @property
    def _listener(self):
        return self.asr.listener

    @_listener.setter
    def _listener(self, value):
        self.asr.listener = value

    @property
    def listening(self) -> bool:
        return self.asr.listening

    @listening.setter
    def listening(self, value: bool) -> None:
        self.asr.listening = value

    @property
    def _wake(self):
        return self.wake.engine

    @_wake.setter
    def _wake(self, value) -> None:
        self.wake.engine = value

    @property
    def wake_armed(self) -> bool:
        return self.wake.armed

    @wake_armed.setter
    def wake_armed(self, value: bool) -> None:
        self.wake.armed = value

    @property
    def hotwords(self) -> list[str]:
        return self.wake.hotwords

    @hotwords.setter
    def hotwords(self, value: list[str]) -> None:
        self.wake.hotwords = value

    @property
    def speaking(self) -> bool:
        return self.tts.speaking

    @speaking.setter
    def speaking(self, value: bool) -> None:
        self.tts.speaking = value

    @property
    def _player(self):
        return self.tts.player

    @_player.setter
    def _player(self, value) -> None:
        self.tts.player = value

    # --- facade delegators (patch surface for tests) ---
    async def start_listening(self, audio_source=None) -> dict:
        return await self.asr.start_listening(audio_source=audio_source)

    async def stop_listening(self) -> dict:
        return await self.asr.stop_listening()

    async def arm_wake(self) -> dict:
        return await self.wake.arm_wake()

    async def disarm_wake(self) -> dict:
        return await self.wake.disarm_wake()

    async def set_hotwords(self, words: list[str]) -> list[str]:
        return await self.wake.set_hotwords(words)

    async def stop_tts(self) -> dict:
        return await self.tts.stop_tts()

    async def speak(self, text: str, voice: str = "", model: str = "", *, synthesize=None, stream_factory=None) -> dict:
        # Orchestration stays on the facade (Wave 2 keystone): wake pause/resume
        # + ASR stop/resume wrap the leaf TtsService._play so test patches on
        # daemon.speak / daemon.stop_listening / daemon.start_listening keep
        # hitting the real facade path.
        if self._player is not None and self._player.active:
            logger.warning("speak rejected: already speaking")
            return {"played": False, "error": "already speaking"}
        wake = self._wake
        was_listening = self._listener is not None and self._listener.active
        logger.info("speak chars=%d voice=%s was_listening=%s", len(text), voice, was_listening)
        if wake is not None and wake.active and not wake.paused:
            wake.pause()
        if was_listening:
            with contextlib.suppress(Exception):
                await self.stop_listening()
        try:
            result = await self.tts._play(
                text, voice=voice, model=model, synthesize=synthesize, stream_factory=stream_factory
            )
        finally:
            # R5: a resume failure leaves wake/ASR half-restored — previously
            # swallowed as a warning, silently degrading the pipeline. Now
            # broadcast a speech.tts error event with degraded=True so clients
            # learn the pipeline needs re-arm, and latch tts.degraded.
            resume_errors: list[str] = []
            if wake is not None and wake.active and wake.paused:
                try:
                    wake.resume()
                except Exception as e:
                    logger.warning("wake resume failed: %s", e)
                    resume_errors.append(f"wake resume: {e}")
            if was_listening:
                try:
                    await self.start_listening()
                except Exception as e:
                    logger.warning("start_listening resume failed: %s", e)
                    resume_errors.append(f"start_listening: {e}")
            if resume_errors:
                self.tts.degraded = True
                with contextlib.suppress(Exception):
                    await self.broadcast(
                        {
                            "event": "speech.tts",
                            "state": "error",
                            "error": "; ".join(resume_errors),
                            "degraded": True,
                            "ts": time.time(),
                        }
                    )
            else:
                self.tts.degraded = False
        return result

    async def _on_wake(self, word: str) -> None:
        # auto-listen handshake: announce the wake, pause detection (so the
        # user's following speech does not self-trigger), arm ASR reusing wake's
        # mic source, then stop ASR and resume wake. The window is bounded by
        # wake_auto_listen_s but ends early once ASR finalizes an utterance and
        # a short silence elapses (audit L1) — no fixed 8s dead-wait.
        wake = self._wake
        if wake is None:
            return
        metrics.wake_fires_total.inc()
        with contextlib.suppress(Exception):
            self.state.record_wake_word(word)
        await self.broadcast({"event": "speech.wake", "word": word, "ts": time.time()})
        # H5: serialize the whole handshake so two rapid wakes cannot interleave
        # the check-then-act on _listener/_wake_listen_done (god-object tear).
        # Full-body lock: a second wake blocks until the first's stop_listening
        # + resume finish, so at most one listener is live on the mic. The
        # asr-active guard below also catches a wake fired while a client
        # started listening directly via RPC. _handshake_lock -> asr lock is
        # one-directional (speak takes asr lock but not this) -> no deadlock.
        async with self._handshake_lock:
            if self._listener is not None and self._listener.active:
                logger.info("wake fired but ASR already active, skip handshake")
                return
            wake.pause()
            done = asyncio.Event()
            self._wake_listen_done = done
            self._last_final_ts = 0.0
            # R3 structural root: the ASR-final handshake frame fires out-of-band
            # via EventBus's handler registry, not through subscriber Queue
            # backpressure, so a slow TTS subscriber's drop-oldest cannot eat it.
            # Token returned here is cleared by THIS handshake's finally only —
            # a second wake (blocked on the lock) gets its own token, so the
            # first's finally cannot wipe the second's handler (R3 residual race).
            token = self.event_bus.set_asr_final_handler(self._on_handshake_final)
            try:
                source = wake.source if wake.own_source else None
                await self.start_listening(audio_source=source)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(done.wait(), timeout=self.config.wake_auto_listen_s)
            except asyncio.CancelledError:
                raise
            finally:
                self.event_bus.clear_asr_final_handler(token)
                self._wake_listen_done = None
                await self.stop_listening()
                if wake is not None and wake.active and wake.paused:
                    with contextlib.suppress(Exception):
                        wake.resume()

    async def probe_audio_lane(self) -> dict:
        try:
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

        # E2: Prometheus text exposition. Zero-dependency (hand-rolled Counter/
        # Gauge/Histogram in metrics.py). text/plain; version=0.0.4 so scrapers
        # accept it. Read-only snapshot of the single `metrics` registry.
        @app.get("/metrics")
        async def _metrics():
            from fastapi import Response

            return Response(
                content=render_prometheus(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

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
                        payload = await asyncio.wait_for(q.get(), timeout=self.config.ws_keepalive_s)
                    except TimeoutError:
                        # H2: keepalive is its own event type, not a disguised
                        # speech.asr frame. Clients filter speech.asr for real
                        # ASR state without sifting keepalive noise.
                        await ws.send_json({"event": "speech.keepalive"})
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

        # M-A (issue #2): WS PCM audio ingress — alternative transport to the
        # local mic. Bidirectional: client sends binary 16k mono PCM frames
        # (int16 LE default, float32 via start.encoding) plus JSON control
        # frames; server sends JSON events (same speech.* contract as
        # /stream/stt) and, with tts_sink=ws, binary int16 TTS PCM. Full-duplex
        # sessions keep VAD live during playback and fire speech.barge_in.
        @app.websocket("/stream/audio")
        async def _stream_audio(ws: WebSocket):
            if not self.config.ws_ingress_enabled:
                await ws.close(code=4404, reason="ws ingress disabled")
                return
            if self.config.ws_ingress_token:
                token = ws.query_params.get("token", "") or ws.headers.get("x-fusion-speech-token", "")
                if token != self.config.ws_ingress_token:
                    await ws.close(code=4401, reason="unauthorized")
                    return
            await ws.accept()
            session = None
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if msg.get("text") is not None:
                        try:
                            frame = json.loads(msg["text"])
                        except (json.JSONDecodeError, ValueError) as e:
                            await ws.send_json({"event": "speech.session.error", "error": f"bad json: {e}"})
                            continue
                        if not isinstance(frame, dict):
                            await ws.send_json(
                                {"event": "speech.session.error", "error": "control frame must be an object"}
                            )
                            continue
                        ftype = frame.get("type", "")
                        if ftype == "start":
                            if session is not None:
                                await ws.send_json(
                                    {"event": "speech.session.error", "error": "session already started"}
                                )
                                continue
                            session = WsIngressSession(
                                self,
                                ws,
                                mode=frame.get("mode", "full_duplex"),
                                encoding=frame.get("encoding", "int16"),
                                tts_sink=frame.get("tts_sink", "ws"),
                                asr_engine=frame.get("asr_engine", "local"),
                                tts_engine=frame.get("tts_engine", "local"),
                                barge_in=frame.get("barge_in"),
                                reprompt_min_confidence=frame.get("reprompt_min_confidence"),
                                session_id=frame.get("session", ""),
                            )
                            try:
                                await session.start()
                            except Exception as e:
                                logger.warning("ws session start rejected: %s", e)
                                await ws.send_json({"event": "speech.session.error", "error": str(e)})
                                await session.stop()
                                session = None
                        elif ftype == "stop":
                            if session is not None:
                                await session.stop()
                                session = None
                        elif ftype == "speak":
                            if session is None:
                                await ws.send_json({"event": "speech.session.error", "error": "session not started"})
                                continue
                            result = await session.speak(frame)
                            if result.get("error"):
                                await ws.send_json({"event": "speech.session.error", "error": result["error"]})
                        elif ftype == "tts_cancel":
                            if session is not None:
                                await session.cancel_tts()
                        else:
                            await ws.send_json(
                                {"event": "speech.session.error", "error": f"unknown control type {ftype!r}"}
                            )
                    elif msg.get("bytes") is not None:
                        if session is None:
                            await ws.send_json({"event": "speech.session.error", "error": "session not started"})
                            continue
                        data = msg["bytes"]
                        if len(data) > self.config.ws_max_frame_bytes:
                            await ws.send_json(
                                {
                                    "event": "speech.session.error",
                                    "error": f"audio frame too large: {len(data)} > {self.config.ws_max_frame_bytes}",
                                }
                            )
                            continue
                        try:
                            session.feed(data)
                        except ValueError as e:
                            await ws.send_json({"event": "speech.session.error", "error": str(e)})
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.exception("stream/audio websocket error")
            finally:
                if session is not None:
                    await session.stop()

        self._app = app
        return app

    async def start(self) -> None:
        with contextlib.suppress(FileNotFoundError, OSError):
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._on_client, path=self.socket_path)
        try:
            os.chmod(self.socket_path, 0o600)
        except OSError as e:
            logger.warning("chmod socket %s failed: %s", self.socket_path, e)
        logger.info("UDS listening on %s", self.socket_path)

        app = self.build_app()
        try:
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
        # E6: bump restart_count + persist. If this is a restart (count > 1),
        # notify subscribers so clients know the daemon was rebooted (crash
        # recovery or deliberate restart). Broadcast after probe so audio_lane
        # is populated for clients that re-act on the frame.
        self._restart_count = self.state.enter_start()
        await self.probe_audio_lane()
        await self._announce_restart()
        # E1: re-probe fusion-mlx's audio models periodically so a lane that
        # went stale (model unloaded mid-session) self-heals without a manual
        # speech.reprobe call. Failure leaves the prior lane in place.
        self._probe_task = asyncio.create_task(self._probe_loop())
        logger.info("fusion-speech daemon started v%s", __version__)

    async def _probe_loop(self) -> None:
        # E1: periodic background re-probe. Catches fusion-mlx model
        # load/unload after daemon start. Tolerates transient failures (keeps
        # the last good lane); stops cleanly on cancel.
        try:
            while self.running:
                await asyncio.sleep(self.config.probe_interval_s)
                if not self.running:
                    break
                with contextlib.suppress(Exception):
                    await self.probe_audio_lane()
        except asyncio.CancelledError:
            pass

    async def _announce_restart(self) -> None:
        # E6: broadcast speech.restarted when this boot is not the first.
        # Extracted so tests can exercise the broadcast without running the
        # full start() (UDS + uvicorn bind).
        if self._restart_count <= 1:
            return
        with contextlib.suppress(Exception):
            await self.broadcast(
                {
                    "event": "speech.restarted",
                    "restart_count": self._restart_count,
                    "ts": time.time(),
                }
            )
        logger.info("broadcast speech.restarted restart_count=%d", self._restart_count)

    async def _on_client(self, reader, writer) -> None:
        await handle_client(reader, writer, self._dispatcher, lambda: self.running)

    async def stop(self) -> None:
        self.running = False
        # E1: stop the background re-probe loop first so it cannot race with
        # teardown (a probe firing mid-stop would repopulate audio_lane).
        if self._probe_task and not self._probe_task.done():
            self._probe_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._probe_task
            self._probe_task = None
        # R6: notify WS subscribers before tearing down so clients can drain
        # their in-flight frames instead of seeing a hard socket close.
        with contextlib.suppress(Exception):
            await self.broadcast({"event": "speech.shutdown", "ts": time.time()})
        if self._player is not None:
            logger.info("stopping TTS player on daemon stop")
            try:
                await asyncio.wait_for(self._player.cancel(), timeout=DRAIN_TIMEOUT)
            except TimeoutError:
                logger.warning("TTS player cancel timed out after %ss", DRAIN_TIMEOUT)
            except Exception as e:
                logger.warning("TTS player cancel on stop failed: %s", e)
            self._player = None
            self.speaking = False
        if self._wake is not None:
            try:
                await asyncio.wait_for(self._wake.stop(), timeout=DRAIN_TIMEOUT)
            except TimeoutError:
                logger.warning("wake.stop timed out after %ss", DRAIN_TIMEOUT)
            except Exception as e:
                logger.warning("wake.stop failed: %s", e)
            self._wake = None
            self.wake_armed = False
        if self._listener is not None:
            try:
                await asyncio.wait_for(self._listener.stop(), timeout=DRAIN_TIMEOUT)
            except TimeoutError:
                logger.warning("listener.stop timed out after %ss", DRAIN_TIMEOUT)
            except Exception as e:
                logger.warning("listener.stop failed: %s", e)
            self._listener = None
            self.listening = False
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=DRAIN_TIMEOUT)
            except TimeoutError:
                logger.warning("UDS server wait_closed timed out after %ss", DRAIN_TIMEOUT)
            except Exception as e:
                logger.warning("UDS server wait_closed failed: %s", e)
            self._server = None
        if self._http_task and not self._http_task.done():
            self._http_task.cancel()
            try:
                await asyncio.wait_for(self._http_task, timeout=DRAIN_TIMEOUT)
            except (asyncio.CancelledError, TimeoutError):
                pass
            except Exception as e:
                logger.warning("http_task drain failed: %s", e)
            self._http_task = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError as e:
                logger.warning("unlink socket %s failed: %s", self.socket_path, e)
        # E6: persist uptime_at_stop on graceful shutdown so the next boot's
        # enter_start can report prior uptime. Suppress so a write failure
        # never masks a clean stop.
        with contextlib.suppress(Exception):
            self.state.enter_stop()
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
