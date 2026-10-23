from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid

import numpy as np

from fusion_speech.engines import make_tts_engine, tts_engine_adapter
from fusion_speech.metrics import metrics
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.session import VoiceSession
from fusion_speech.tts_playback import StreamingPlayer

logger = logging.getLogger(__name__)

_WS_SINK_QUEUE = 64  # frames; drop-oldest, same contract as the event bus

_CLOSE_POLICY_DISABLED = 4404
_CLOSE_POLICY_UNAUTHORIZED = 4401
_CLOSE_POLICY_BUSY = 4409
_CLOSE_POLICY_BAD_FRAME = 4402


class StreamAudioSource:
    # AudioSource-compatible PCM source fed from a WebSocket instead of a mic
    # device (M-A). Same RingBuffer contract as the mic path, so
    # StreamingListener consumes WS-fed audio identically to capture (parity).
    # feed() runs on the WS receive task; the listener's drain task is the
    # single reader (SPSC preserved). No hardware in start/stop — lifecycle
    # is the socket.
    def __init__(self, *, sample_rate: int = 16000, capacity_s: float = 5.0):
        self.ring = RingBuffer(int(sample_rate * capacity_s))
        self.sample_rate = sample_rate
        self._active = False
        self._fed = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def fed(self) -> int:
        return self._fed

    def start(self) -> None:
        self._active = True
        logger.info("ws audio source started rate=%d", self.sample_rate)

    def stop(self) -> None:
        self._active = False
        self.ring.close()
        logger.info("ws audio source stopped fed=%d dropped=%d", self._fed, self.ring.total_dropped)

    def feed(self, pcm: np.ndarray) -> int:
        if not self._active:
            return 0
        self._fed += pcm.size
        return self.ring.write(np.asarray(pcm, dtype=np.float32).reshape(-1))


def decode_pcm(data: bytes, encoding: str) -> np.ndarray:
    # Wire formats: int16 LE (default, compact) or float32 LE (numpy-native).
    # Both 16 kHz mono — the pipeline contract; anything else is upstream of
    # this layer (resample client-side).
    if encoding == "float32":
        if len(data) % 4 != 0:
            raise ValueError(f"float32 frame size {len(data)} not a multiple of 4")
        return np.frombuffer(data, dtype="<f4").astype(np.float32)
    if len(data) % 2 != 0:
        raise ValueError(f"int16 frame size {len(data)} not a multiple of 2")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class WsTtsSink:
    # OutputStream-like write target for StreamingPlayer when the session's
    # TTS sink is the WebSocket (browser plays the audio and owns AEC, M-B).
    # write() only enqueues — the player's write loop must never block on
    # socket backpressure — and a drain task converts float32 frames to int16
    # LE and sends them as binary WS messages. flush() drops pending frames
    # (barge-in must stop audio immediately, not drain the tail).
    def __init__(self, ws):
        self._ws = ws
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=_WS_SINK_QUEUE)
        self._drain_task: asyncio.Task | None = None
        self.active = False
        self.dropped = 0

    def start(self) -> None:
        self.active = True
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def _drain_loop(self) -> None:
        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break
                i16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
                await self._ws.send_bytes(i16.tobytes())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ws tts sink drain failed")

    def write(self, frame: np.ndarray) -> None:
        if not self.active:
            return
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except asyncio.QueueEmpty:
                pass
            self.dropped += 1

    def stop(self) -> None:
        # Sentinel lets the drain task send whatever is still queued, then
        # exit — natural completion must not truncate the audio tail. A full
        # queue drops one frame for the sentinel (never lose the exit signal:
        # a lost sentinel would leak the drain task).
        self.active = False
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)

    def flush(self) -> None:
        # Barge-in: drop every queued frame now. Called BEFORE player.cancel()
        # so the tail does not keep playing after the user interrupted.
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def close(self) -> None:
        pass


class WsIngressSession:
    # One /stream/audio connection = one voice session (M-A + M-B). Owns the
    # WS-fed StreamAudioSource, the StreamingListener (via AsrService — same
    # VAD -> segment -> transcribe pipeline as the mic path), the VoiceSession
    # state machine, and the TTS player. Full-duplex: the listener stays live
    # while TTS plays; VAD onset during playback (seen on the event pump)
    # fires barge-in + cancels TTS. Half-duplex keeps the mic-era behavior
    # (ASR stopped during playback). One ASR stream at a time (mic or WS) —
    # AsrService has a single listener slot by design.
    def __init__(
        self,
        daemon,
        ws,
        *,
        mode: str = "full_duplex",
        encoding: str = "int16",
        tts_sink: str = "ws",
        asr_engine: str = "local",
        tts_engine: str = "local",
        barge_in: bool | None = None,
        reprompt_min_confidence: float | None = None,
        session_id: str = "",
    ):
        self.daemon = daemon
        self.ws = ws
        self.config = daemon.config
        self.mode = mode
        self.encoding = encoding
        self.tts_sink = tts_sink
        self.asr_engine_name = asr_engine
        self.tts_engine_name = tts_engine
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.full_duplex = mode == "full_duplex"
        self.session = VoiceSession(
            self.session_id,
            barge_in=barge_in if barge_in is not None else (self.full_duplex and self.config.barge_in_enabled),
            reprompt_min_confidence=(
                reprompt_min_confidence if reprompt_min_confidence is not None else self.config.reprompt_min_confidence
            ),
            on_event=daemon.broadcast,
        )
        self.source = StreamAudioSource(sample_rate=self.config.sample_rate)
        self._pump_task: asyncio.Task | None = None
        self._pump_queue: asyncio.Queue | None = None
        self.player: StreamingPlayer | None = None
        self.sink: WsTtsSink | None = None
        self._speak_task: asyncio.Task | None = None
        self._stopping = False
        # Ownership latch: a session rejected at start (ASR slot busy) must not
        # tear down shared daemon state in stop() — it never took any.
        self._started = False

    async def start(self) -> dict:
        if self.daemon.asr.listener is not None and self.daemon.asr.listener.active:
            raise RuntimeError("ASR already active (mic listening or another WS session); one stream at a time")
        metrics.sessions.inc(1)
        await self.daemon.asr.start_listening(audio_source=self.source)
        await self.session.start()
        self._started = True
        self._pump_queue = self.daemon.subscribe()
        self._pump_task = asyncio.create_task(self._pump_loop())
        info = {
            "event": "speech.session.info",
            "session": self.session_id,
            "mode": self.mode,
            "encoding": self.encoding,
            "tts_sink": self.tts_sink,
            "asr_engine": self.asr_engine_name,
            "tts_engine": self.tts_engine_name,
            "barge_in": self.session.barge_in,
            "sample_rate": self.config.sample_rate,
        }
        await self.ws.send_json(info)
        logger.info("ws session %s started mode=%s encoding=%s", self.session_id, self.mode, self.encoding)
        return info

    async def _pump_loop(self) -> None:
        # Single forwarder: EventBus subscriber queue -> WS JSON frames.
        # Also watches the stream for the two session reactions: VAD onset
        # during playback (barge-in) and ASR final (low-confidence reprompt).
        assert self._pump_queue is not None
        try:
            while True:
                payload = await self._pump_queue.get()
                if payload.get("event") == "speech.asr":
                    if payload.get("state") == "speaking":
                        await self._maybe_barge_in()
                    elif payload.get("state") == "final":
                        await self.session.on_asr_final(payload)
                await self.ws.send_json(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ws session %s pump failed", self.session_id)

    async def _maybe_barge_in(self) -> None:
        # VAD onset while TTS plays: flush the sink (drop pending audio),
        # cancel the player, count the barge-in. All in-loop ops — bounded
        # well under the 100 ms server-side cancel budget.
        if not self.session.speaking or not self.session.barge_in:
            return
        t0 = time.monotonic()
        if self.sink is not None:
            self.sink.flush()
        if self.player is not None and self.player.active:
            await self.player.cancel()
        event = await self.session.on_vad_onset()
        metrics.barge_in_total.inc()
        logger.info(
            "ws session %s barge-in #%s cancel_ms=%.1f",
            self.session_id,
            event.get("count") if event else "?",
            (time.monotonic() - t0) * 1000.0,
        )

    def feed(self, data: bytes) -> None:
        pcm = decode_pcm(data, self.encoding)
        if pcm.size:
            self.source.feed(pcm)

    async def speak(self, params: dict) -> dict:
        # Runs as a task so the WS receive loop keeps consuming audio (the
        # whole point of full-duplex). One playback at a time.
        if self._speak_task is not None and not self._speak_task.done():
            return {"error": "already speaking"}
        text = params.get("text", "")
        if not text:
            return {"error": "text required"}
        if len(text) > 5000:
            return {"error": "text too long"}
        self._speak_task = asyncio.create_task(
            self._speak(
                text,
                voice=params.get("voice", ""),
                model=params.get("model", ""),
                ref_audio=params.get("ref_audio", ""),
                ref_text=params.get("ref_text", ""),
            )
        )
        return {"speaking": True, "session": self.session_id}

    async def _speak(
        self, text: str, *, voice: str = "", model: str = "", ref_audio: str = "", ref_text: str = ""
    ) -> None:
        if not await self.session.begin_speak():
            await self.daemon.broadcast(
                {"event": "speech.tts", "state": "error", "error": "session not listening", "session": self.session_id}
            )
            return
        if not self.full_duplex:
            # half-duplex: mic-era behavior — ASR paused during playback so
            # the outgoing audio is never transcribed (no browser AEC assumed).
            with contextlib.suppress(Exception):
                await self.daemon.asr.stop_listening()
        engine = make_tts_engine(self.tts_engine_name, self.config)
        sink = None
        stream_factory = None
        if self.tts_sink == "ws":
            sink = WsTtsSink(self.ws)
            self.sink = sink

            def stream_factory(**kwargs):
                return sink

        self.player = StreamingPlayer(
            self.config,
            notifier=self.daemon.broadcast,
            synthesize=tts_engine_adapter(engine, ref_audio=ref_audio, ref_text=ref_text),
            stream_factory=stream_factory,
        )
        try:
            result = await self.player.speak(text, voice=voice, model=model)
        finally:
            self.player = None
            self.sink = None
        await self.session.end_speak(cancelled=bool(result.get("cancelled")))
        if not self.full_duplex and not self._stopping:
            with contextlib.suppress(Exception):
                await self.daemon.asr.start_listening(audio_source=self.source)

    async def cancel_tts(self) -> dict:
        player = self.player
        if player is None:
            return {"stopped": False}
        if self.sink is not None:
            self.sink.flush()
        result = await player.cancel()
        return result

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if not self._started:
            # rejected at start: nothing owned, nothing to release (releasing
            # the shared ASR slot here would kill the ACTIVE session's listener)
            logger.info("ws session %s stop: never started, nothing to release", self.session_id)
            return
        if self._speak_task is not None and not self._speak_task.done():
            self._speak_task.cancel()
            # CancelledError is BaseException on py3.14 — suppress it with the
            # task awaits or it escapes and kills the caller (same pattern as
            # SpeechDaemon.stop, audit R6).
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._speak_task
            self._speak_task = None
        if self.player is not None and self.player.active:
            with contextlib.suppress(Exception):
                await self.player.cancel()
            self.player = None
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump_task
            self._pump_task = None
        if self._pump_queue is not None:
            self.daemon.unsubscribe(self._pump_queue)
            self._pump_queue = None
        with contextlib.suppress(Exception):
            await self.daemon.asr.stop_listening()
        with contextlib.suppress(Exception):
            await self.session.stop()
        metrics.sessions.dec(1)
        logger.info("ws session %s stopped", self.session_id)
