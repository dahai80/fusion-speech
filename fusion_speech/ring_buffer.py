from __future__ import annotations

from collections.abc import Callable

import numpy as np
from fusion_core import config as _core_config

logger = _core_config.logger


class RingBuffer:
    """GIL-protected SPSC ring for realtime PCM audio (CPython only; a
    free-threaded build would need a real lock — re-audit before enabling).

    Producer = sounddevice callback (realtime thread, must never block).
    Consumer = asyncio task draining into VAD. Single producer, single
    consumer, so writes and reads touch disjoint fields. Available/free are
    derived from absolute in/out counters (never wrapped), so full and empty
    are unambiguous (full = in-out == capacity, empty = in == out); ring
    indices wrap modulo capacity only for buffer access. On overflow the
    oldest samples are dropped (realtime capture must never block); the
    overrun counter is surfaced in speech.status. The in/out counters are
    plain ints mutated under the GIL — not atomic on a free-threaded runtime.
    """

    def __init__(self, capacity_samples: int, dtype: np.dtype = np.float32):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self._buf = np.zeros(capacity_samples, dtype=dtype)
        self._capacity = capacity_samples
        self._total_in = 0
        self._total_out = 0
        self._overruns = 0
        self._total_dropped = 0
        self._closed = False
        self._notifier: Callable[[], None] | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def overruns(self) -> int:
        return self._overruns

    @property
    def total_written(self) -> int:
        return self._total_in

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    def set_notifier(self, notifier: Callable[[], None] | None) -> None:
        self._notifier = notifier

    def available(self) -> int:
        return self._total_in - self._total_out

    def write(self, samples: np.ndarray) -> int:
        if self._closed:
            return 0
        n = samples.size
        if n == 0:
            return 0
        # Overflow path runs on the PortAudio realtime thread (write() is
        # called from _on_audio). P2-1: no I/O here — only bump counters.
        # Overruns are surfaced in AudioSource.stop() (asyncio-side) and
        # via stats() for any consumer that wants to poll.
        free = self._capacity - self.available()
        dropped = 0
        if n > free:
            dropped = n - free
            self._overruns += 1
            self._total_dropped += dropped
            self._total_out += dropped
        w = self._total_in % self._capacity
        first = min(n, self._capacity - w)
        self._buf[w : w + first] = samples[:first]
        second = n - first
        if second > 0:
            self._buf[:second] = samples[first:]
        self._total_in += n
        if self._notifier is not None:
            try:
                self._notifier()
            except Exception:
                logger.exception("ring notifier callback failed")
        return n

    def read(self, max_samples: int) -> np.ndarray:
        if max_samples <= 0:
            return np.empty(0, dtype=self._buf.dtype)
        avail = self.available()
        if avail == 0:
            return np.empty(0, dtype=self._buf.dtype)
        n = min(max_samples, avail)
        r = self._total_out % self._capacity
        first = min(n, self._capacity - r)
        out = np.empty(n, dtype=self._buf.dtype)
        out[:first] = self._buf[r : r + first]
        second = n - first
        if second > 0:
            out[first:] = self._buf[:second]
        self._total_out += n
        return out

    def drain(self) -> np.ndarray:
        return self.read(self.available())

    def reset(self) -> None:
        self._total_in = 0
        self._total_out = 0
        self._overruns = 0
        self._total_dropped = 0
        # a closed ring stays closed — reset must not revive it (audit A3)

    def close(self) -> None:
        self._closed = True
        if self._notifier is not None:
            try:
                self._notifier()
            except Exception:
                logger.exception("ring notifier callback failed on close")

    def stats(self) -> dict:
        return {
            "capacity": self._capacity,
            "available": self.available(),
            "overruns": self._overruns,
            "total_written": self._total_in,
            "total_dropped": self._total_dropped,
            "closed": self._closed,
        }
