from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from fusion_speech.mlx_bridge import _parse_wav_sample_rate, synthesize_stream

logger = logging.getLogger(__name__)

_WAV_HEADER_LEN = 44


class TtsPlaybackError(RuntimeError):
    pass


class StreamingPlayer:
    def __init__(
        self,
        config,
        *,
        synthesize: Callable[..., AsyncIterator[bytes]] | None = None,
        notifier: Callable[[dict], Awaitable[None]] | None = None,
        stream_factory: Callable[..., object] | None = None,
    ):
        self.config = config
        self._synthesize = synthesize or synthesize_stream
        self._notifier = notifier
        self._stream_factory = stream_factory
        self._out_stream = None
        self._queue: asyncio.Queue | None = None
        self._fetch_task: asyncio.Task | None = None
        self._write_task: asyncio.Task | None = None
        self._active = False
        self._speaking = False
        self._cancelled = False
        self._sample_rate = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def speaking(self) -> bool:
        return self._speaking

    async def speak(self, text: str, voice: str = "", model: str = "") -> dict:
        if self._active:
            logger.warning("speak already in flight")
            return {"played": False, "error": "already speaking"}
        self._active = True
        self._cancelled = False
        self._sample_rate = 0
        cap = max(1, int(self.config.tts_playback_buffer_s * self.config.sample_rate))
        self._queue = asyncio.Queue(maxsize=cap)
        await self._notify({"event": "speech.tts", "state": "starting", "chars": len(text)})
        self._fetch_task = asyncio.create_task(self._fetch_loop(text, voice, model))
        self._write_task = asyncio.create_task(self._write_loop())
        try:
            await asyncio.gather(self._fetch_task, self._write_task)
        except Exception as e:
            logger.warning("speak failed: %s", e)
            self._cancelled = True
            for task in (self._fetch_task, self._write_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(self._fetch_task, self._write_task, return_exceptions=True)
            await self._notify({"event": "speech.tts", "state": "error", "error": str(e)})
            return {"played": False, "error": str(e)}
        finally:
            self._close_stream()
            self._active = False
        if self._cancelled:
            await self._notify({"event": "speech.tts", "state": "cancelled"})
            return {"played": False, "cancelled": True}
        await self._notify({"event": "speech.tts", "state": "done", "chars": len(text)})
        return {"played": True, "chars": len(text), "sample_rate": self._sample_rate}

    async def _fetch_loop(self, text: str, voice: str, model: str) -> None:
        header_seen = False
        leftover = b""
        try:
            async for chunk in self._synthesize(
                self.config, text, voice=voice, model=model, stream=True
            ):
                if self._cancelled:
                    break
                if not chunk:
                    continue
                if not header_seen:
                    if len(chunk) < _WAV_HEADER_LEN:
                        logger.warning("tts first chunk < header (%d)", len(chunk))
                        header_seen = True
                        data = chunk
                    else:
                        self._sample_rate = _parse_wav_sample_rate(chunk[:_WAV_HEADER_LEN])
                        logger.info("tts stream rate=%d", self._sample_rate)
                        header_seen = True
                        data = chunk[_WAV_HEADER_LEN:]
                else:
                    data = chunk
                if leftover:
                    data = leftover + data
                    leftover = b""
                if len(data) % 2 != 0:
                    leftover = data[-1:]
                    data = data[:-1]
                if not data:
                    continue
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if self._sample_rate and self._sample_rate != self.config.sample_rate:
                    pcm = _resample(pcm, self._sample_rate, self.config.sample_rate)
                await self._queue.put(pcm)
            self._queue.put_nowait(None)
        except Exception:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)
            raise

    async def _write_loop(self) -> None:
        assert self._queue is not None
        started = False
        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break
                if self._cancelled:
                    continue
                if not started:
                    self._start_output()
                    started = True
                    self._speaking = True
                    await self._notify({"event": "speech.tts", "state": "speaking"})
                if self._out_stream is not None:
                    self._write_frame(frame)
        finally:
            self._speaking = False

    def _start_output(self) -> None:
        device = self.config.tts_output_device or None
        factory = self._stream_factory or self._default_stream_factory
        try:
            stream = factory(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                device=device,
            )
            stream.start()
            self._out_stream = stream
            logger.info("tts output started rate=%d device=%s", self.config.sample_rate, device)
        except Exception as e:
            logger.error("tts output open failed: %s", e)
            self._out_stream = None
            raise TtsPlaybackError(f"cannot open output stream: {e}") from e

    def _default_stream_factory(self, *, samplerate, channels, dtype, device):
        import sounddevice as sd

        return sd.OutputStream(samplerate=samplerate, channels=channels, dtype=dtype, device=device)

    def _write_frame(self, frame: np.ndarray) -> None:
        stream = self._out_stream
        if stream is None:
            return
        try:
            stream.write(np.ascontiguousarray(frame, dtype=np.float32))
        except Exception:
            logger.exception("tts write failed")

    def _close_stream(self) -> None:
        stream = self._out_stream
        if stream is None:
            return
        self._out_stream = None
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()
        logger.info("tts output closed")

    async def cancel(self) -> dict:
        if not self._active:
            return {"stopped": False}
        self._cancelled = True
        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)
        logger.info("tts cancel requested")
        return {"stopped": True}

    async def _notify(self, payload: dict) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier(payload)
        except Exception:
            logger.exception("tts notifier failed")


def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    n_out = int(round(pcm.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.empty(0, dtype=np.float32)
    idx = np.linspace(0, pcm.size - 1, n_out)
    out = np.interp(idx, np.arange(pcm.size), pcm).astype(np.float32)
    logger.debug("resample %d -> %d (%d samples)", src_rate, dst_rate, out.size)
    return out
