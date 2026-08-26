from __future__ import annotations

import asyncio
import contextlib
import logging
import wave
from collections.abc import Awaitable, Callable
from io import BytesIO

import numpy as np

from fusion_speech.audio_source import AudioSource
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.vad import SileroVadEngine, VadSegment

logger = logging.getLogger(__name__)

_RING_CAPACITY = 16000 * 5  # 5s @ 16k, drop-oldest on overrun


class SttStreamError(RuntimeError):
    pass


class StreamingListener:
    # One active mic-listening session. Producer = AudioSource (PortAudio
    # callback) writing PCM into the ring; consumer = the drain task, which
    # feeds VAD windows. VAD emits segment-open / segment-close; the listener
    # captures the PCM between start_sample and end_sample (clamped to what the
    # ring still holds), encodes a 16k mono WAV, posts it to mlx_bridge for
    # transcription, and emits a final speech.asr notification through
    # `notifier`. A short listening-start partial is emitted on onset. All
    # inference is async + off-thread; capture never blocks.
    def __init__(
        self,
        config,
        *,
        transcribe: Callable[..., Awaitable[dict]],
        notifier: Callable[[dict], Awaitable[None]] | None = None,
        audio_source: AudioSource | None = None,
        vad: SileroVadEngine | None = None,
    ):
        self.config = config
        self._transcribe = transcribe
        self._notifier = notifier
        self._own_source = audio_source is None
        if audio_source is not None:
            # reuse the source's ring so the producer and this consumer share
            # one buffer; the drain task reads from the same ring the callback
            # writes to.
            self.ring = audio_source.ring
        else:
            self.ring = RingBuffer(_RING_CAPACITY)
        self.source = audio_source or AudioSource(
            self.ring,
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self._own_vad = vad is None
        self.vad = vad or SileroVadEngine(
            sample_rate=config.sample_rate,
            threshold=config.vad_speech_threshold,
            on_open=self._on_open,
            on_close=self._on_close,
        )
        self._drain_task: asyncio.Task | None = None
        self._active = False
        self._capture: list[np.ndarray] = []
        self._capture_start: int | None = None
        self._segment_seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def active(self) -> bool:
        return self._active

    async def start(self) -> dict:
        if self._active:
            logger.warning("streaming listener already active")
            return {"listening": True, "already": True}
        self._loop = asyncio.get_running_loop()
        self.vad.reset()
        self._capture = []
        self._capture_start = None
        self._segment_seq = 0
        self.ring.reset()
        try:
            self.source.start()
        except Exception as e:
            logger.error("cannot start audio source: %s", e)
            raise SttStreamError(f"cannot start audio source: {e}") from e
        self._active = True
        self._drain_task = asyncio.create_task(self._drain_loop())
        logger.info(
            "streaming listener started rate=%d vad_energy_only=%s",
            self.config.sample_rate,
            self.vad.energy_only,
        )
        await self._notify({
            "event": "speech.asr",
            "state": "listening",
            "seq": 0,
        })
        return {"listening": True}

    async def stop(self) -> dict:
        if not self._active:
            return {"listening": False}
        self._active = False
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None
        self.vad.flush()
        if self._own_source:
            self.source.stop()
        logger.info("streaming listener stopped segments=%d", self._segment_seq)
        return {"listening": False, "segments": self._segment_seq}

    async def _drain_loop(self) -> None:
        # Consumer: pull everything the producer has written, feed VAD in
        # windows, and mirror every sample into _capture so a closed segment's
        # PCM is available without re-reading the (possibly already-drained)
        # ring. Polls the ring; short sleep when empty to avoid busy spin.
        try:
            while self._active:
                chunk = self.ring.read(self.ring.available())
                if chunk.size > 0:
                    self._capture.append(chunk)
                    self.vad.feed(chunk)
                else:
                    await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise

    def _on_open(self, start_sample: int) -> None:
        # remember how much capture precedes onset so on_close can slice from
        # here; keep a small lead-in for context.
        lead = min(int(self.config.sample_rate * 0.2), sum(c.size for c in self._capture))
        self._capture_start = sum(c.size for c in self._capture) - lead
        if self._capture_start < 0:
            self._capture_start = 0
        if self._loop is not None and self._notifier is not None:
            asyncio.run_coroutine_threadsafe(
                self._notify({"event": "speech.asr", "state": "speaking", "seq": 0}),
                self._loop,
            )

    def _on_close(self, segment: VadSegment) -> None:
        # Called on the drain task (asyncio thread) — safe to await directly.
        if self._loop is None:
            return
        self._segment_seq += 1
        seq = self._segment_seq
        asyncio.ensure_future(self._transcribe_segment(segment, seq), loop=self._loop)

    async def _transcribe_segment(self, segment: VadSegment, seq: int) -> None:
        # gate on the VAD's pure-speech length (segment.samples), not the
        # capture length (which includes a lead-in for STT context).
        if segment.samples < self.config.sample_rate // 10:  # < 0.1s -> skip
            logger.info("segment %d too short (%d speech samples), skip", seq, segment.samples)
            await self._notify({
                "event": "speech.asr",
                "state": "final",
                "seq": seq,
                "text": "",
                "skipped": True,
            })
            return
        pcm = self._collect_pcm(segment)
        if pcm.size == 0:
            logger.info("segment %d empty capture, skip", seq)
            await self._notify({
                "event": "speech.asr",
                "state": "final",
                "seq": seq,
                "text": "",
                "skipped": True,
            })
            return
        wav_bytes = pcm_to_wav(pcm, self.config.sample_rate)
        await self._notify({
            "event": "speech.asr",
            "state": "transcribing",
            "seq": seq,
            "duration": round(pcm.size / self.config.sample_rate, 3),
        })
        try:
            result = await self._transcribe(
                self.config,
                wav_bytes,
                filename=f"seg-{seq}.wav",
                language=self.config.language,
            )
        except Exception as e:
            logger.exception("segment %d transcribe failed", seq)
            await self._notify({
                "event": "speech.asr",
                "state": "error",
                "seq": seq,
                "error": str(e),
            })
            return
        text = (result.get("text") or "").strip()
        logger.info("segment %d final: %r", seq, text[:120])
        await self._notify({
            "event": "speech.asr",
            "state": "final",
            "seq": seq,
            "text": text,
            "language": result.get("language", ""),
            "duration": result.get("duration"),
        })

    def _collect_pcm(self, segment: VadSegment) -> np.ndarray:
        # Slice from the drain-mirrored _capture (onset marked by _on_open),
        # not the ring — the ring is already drained by the consumer task and
        # may have drop-oldest truncated long segments. _capture grows
        # unbounded only for the session; trim it after each segment below.
        if not self._capture:
            return np.empty(0, dtype=np.float32)
        full = np.concatenate(self._capture)
        start = self._capture_start or 0
        if start >= full.size:
            return np.empty(0, dtype=np.float32)
        pcm = full[start:].astype(np.float32, copy=False)
        # trim the consumed prefix so _capture does not grow without bound;
        # keep a small tail (last 0.2s) as lead-in for the next segment.
        keep = min(int(self.config.sample_rate * 0.2), pcm.size)
        cut = full.size - keep
        if cut > 0:
            self._capture = [full[cut:].copy()]
            self._capture_start = 0
        return pcm

    async def _notify(self, payload: dict) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier(payload)
        except Exception:
            logger.exception("asr notifier failed")


def pcm_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    # 16-bit PCM mono WAV for /audio/transcriptions.
    mono = np.asarray(pcm, dtype=np.float32).reshape(-1)
    clipped = np.clip(mono, -1.0, 1.0)
    i16 = (clipped * 32767.0).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(i16.tobytes())
    return buf.getvalue()
