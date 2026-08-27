from __future__ import annotations

import asyncio
import logging

from fusion_speech.metrics import metrics
from fusion_speech.mlx_bridge import transcribe_stream
from fusion_speech.stt_stream import StreamingListener

logger = logging.getLogger(__name__)


class AsrService:
    # Owns the streaming ASR listener + listening flag. start/stop_listening
    # are leaf operations; the facade delegates so _on_wake/speak (which call
    # self.start_listening) hit the facade delegator and test patches land.
    # audio_source is injected when sharing the wake mic (AudioBus borrow).
    def __init__(self, config, notifier, audio_bus):
        self.config = config
        self._notifier = notifier
        self._audio_bus = audio_bus
        self.listener = None
        self.listening = False
        # H5: serialize start/stop so concurrent calls cannot interleave a
        # check-then-set tear (e.g. two start_listening both passing the
        # already-active check and building two listeners on one mic).
        self._lock = asyncio.Lock()

    async def start_listening(self, audio_source=None) -> dict:
        async with self._lock:
            if self.listener is not None and self.listener.active:
                return {"listening": True, "already": True}
            # H3: if no source injected (bare production start), borrow the bus's
            # canonical mic; else the listener builds its own and we register it so
            # a later wake arm borrows the same source. The _on_wake handshake
            # passes wake.source explicitly (borrow), so that path skips the bus.
            if audio_source is None and self._audio_bus is not None:
                audio_source = self._audio_bus.borrow("asr")
            self.listener = StreamingListener(
                self.config,
                transcribe=transcribe_stream,
                notifier=self._notifier,
                audio_source=audio_source,
            )
            try:
                result = await self.listener.start()
            except Exception as e:
                self.listener = None
                self.listening = False
                metrics.listening.set(0)
                logger.error("start_listening failed: %s", e)
                raise
            self.listening = True
            metrics.listening.set(1)
            if audio_source is None and self._audio_bus is not None and self.listener.source is not None:
                self._audio_bus.register(self.listener.source, "asr")
            return result

    async def stop_listening(self) -> dict:
        async with self._lock:
            listener = self.listener
            self.listening = False
            metrics.listening.set(0)
            if listener is None:
                return {"listening": False}
            result = await listener.stop()
            self.listener = None
            if self._audio_bus is not None:
                self._audio_bus.release("asr")
            return result
