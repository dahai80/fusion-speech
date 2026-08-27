from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from fusion_speech.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AudioSourceError(RuntimeError):
    pass


class InputStreamLike:
    # minimal protocol satisfied by sounddevice.InputStream and test fakes:
    # start(), stop(), close(), active
    active: bool

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class AudioSource:
    # Realtime mic capture into a RingBuffer. Producer side of the SPSC ring:
    # the PortAudio callback runs on a realtime thread and must never block,
    # allocate, or await. It only downmixes to mono and pushes into the ring
    # (drop-oldest on overflow). The consumer (VAD/STT) drains from the asyncio
    # side. sounddevice is imported lazily so the module loads headless; tests
    # inject a fake stream_factory to avoid touching audio hardware.
    def __init__(
        self,
        ring: RingBuffer,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 0,
        device: int | str | None = None,
        stream_factory: Callable[..., InputStreamLike] | None = None,
    ):
        self.ring = ring
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self._stream_factory = stream_factory
        self._stream: InputStreamLike | None = None
        self._xruns = 0

    @property
    def is_active(self) -> bool:
        return self._stream is not None and getattr(self._stream, "active", False)

    @property
    def xruns(self) -> int:
        return self._xruns

    def start(self) -> None:
        if self._stream is not None:
            logger.warning("audio source already started")
            return
        factory = self._stream_factory or self._default_stream_factory
        try:
            stream = factory(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.blocksize,
                device=self.device,
                callback=self._on_audio,
            )
        except Exception as e:
            logger.error("audio source open failed: %s", e)
            raise AudioSourceError(f"cannot open input stream: {e}") from e
        stream.start()
        self._stream = stream
        logger.info(
            "audio source started: rate=%d ch=%d device=%s blocksize=%d",
            self.sample_rate,
            self.channels,
            self.device,
            self.blocksize,
        )

    def _default_stream_factory(self, *, samplerate, channels, blocksize, device, callback):
        import sounddevice as sd

        return sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
            device=device,
            dtype="float32",
            callback=callback,
        )

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # PortAudio callback: realtime thread. Never block/await here.
        if status:
            self._xruns += 1
            logger.warning("audio callback status=%s xruns=%d", status, self._xruns)
        mono = indata[:, 0] if self.channels > 1 and indata.ndim > 1 else indata.reshape(-1)
        self.ring.write(np.ascontiguousarray(mono, dtype=np.float32))

    def stop(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            stream.stop()
        except Exception:
            logger.exception("audio stream stop failed")
        try:
            stream.close()
        except Exception:
            logger.exception("audio stream close failed")
        logger.info("audio source stopped: xruns=%d", self._xruns)
        self.ring.close()
