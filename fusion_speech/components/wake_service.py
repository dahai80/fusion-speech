from __future__ import annotations

import asyncio
import logging

from fusion_speech.metrics import metrics
from fusion_speech.wake_word import WakeWordEngine

logger = logging.getLogger(__name__)


class WakeService:
    # Owns the wake-word engine + armed/hotwords state. arm/disarm/set_hotwords
    # are leaf operations; the facade delegates so the facade's _on_wake
    # orchestration (which calls self.wake.pause/resume via this service's
    # engine) keeps test patches on the facade working. on_wake is NOT held
    # here — the facade passes its own _on_wake into the engine so the
    # handshake orchestration stays on the facade (test-compat keystone).
    def __init__(self, config, audio_bus, on_wake=None):
        self.config = config
        self._audio_bus = audio_bus
        self._on_wake = on_wake
        self.engine = None
        self.armed = False
        self.hotwords: list[str] = []
        # H5: serialize arm/disarm/set_hotwords so concurrent calls cannot
        # interleave the engine check-then-build/stop. pause/resume stay sync
        # (engine-level, no asyncio lock); they run inside the facade
        # _handshake_lock so they do not race arm/disarm there.
        self._lock = asyncio.Lock()

    async def arm_wake(self) -> dict:
        async with self._lock:
            if self.engine is not None and self.engine.active:
                return {"armed": True, "already": True}
            # H3: borrow the bus's canonical mic source if one is already live (e.g.
            # ASR listening), else the engine builds its own and we register it so a
            # later ASR handshake borrows the same source rather than peeking the
            # engine's internals.
            source = self._audio_bus.borrow("wake") if self._audio_bus else None
            self.engine = WakeWordEngine(self.config, on_wake=self._on_wake, audio_source=source)
            result = await self.engine.start()
            self.armed = bool(result.get("armed"))
            metrics.wake_armed.set(1 if self.armed else 0)
            if self.armed:
                self.hotwords = list(self.engine.words)
                if self._audio_bus is not None and source is None and self.engine.source is not None:
                    self._audio_bus.register(self.engine.source, "wake")
            return result

    async def disarm_wake(self) -> dict:
        async with self._lock:
            wake = self.engine
            self.armed = False
            metrics.wake_armed.set(0)
            if wake is None:
                return {"armed": False}
            result = await wake.stop()
            self.engine = None
            if self._audio_bus is not None:
                self._audio_bus.release("wake")
            return result

    async def set_hotwords(self, words: list[str]) -> list[str]:
        async with self._lock:
            if self.engine is None:
                self.hotwords = list(words)
                return list(words)
            accepted = await self.engine.set_hotwords(words)
            self.hotwords = list(accepted)
            return accepted

    def pause(self) -> None:
        if self.engine is not None:
            self.engine.pause()

    def resume(self) -> None:
        if self.engine is not None:
            self.engine.resume()
