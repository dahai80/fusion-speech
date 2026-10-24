from __future__ import annotations

import asyncio
import logging

from fusion_speech.metrics import metrics

logger = logging.getLogger(__name__)


class EventBus:
    # Fan-out speech.* events to WS subscribers (one bounded Queue each) and
    # dispatch the ASR-final handshake signal out-of-band so a slow TTS
    # subscriber's drop-oldest overflow cannot eat the handshake frame (R3
    # structural root). subscribe/unsubscribe/publish form the public surface;
    # _on_wake registers an asr-final handler (gets a token) before its
    # wait_for and clears THAT token in finally, so the handshake does not
    # flow through subscriber backpressure and a second wake cannot overwrite
    # /wipe the first's handler (R3 residual race fixed by per-handshake token
    # registry).
    def __init__(self, *, queue_maxsize: int = 64):
        self._subscribers: set[asyncio.Queue] = set()
        self._queue_maxsize = queue_maxsize
        self._asr_final_handlers: dict[int, object] = {}
        self._next_token = 0
        # H5/R3: protect handler registry mutation + fan-out so publish and
        # set/clear do not tear. publish is async but the critical section
        # (registry snapshot + queue puts) is short; the lock serializes it.
        self._lock = asyncio.Lock()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers.add(q)
        metrics.subscribers.set(len(self._subscribers))
        logger.info("asr subscriber added (total=%d)", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)
        metrics.subscribers.set(len(self._subscribers))
        logger.info("asr subscriber removed (total=%d)", len(self._subscribers))

    async def publish(self, payload: dict) -> None:
        # Out-of-band ASR-final dispatch: registered handshake handlers run
        # directly, not through a subscriber Queue, so slow-TTS drop-oldest
        # cannot drop the handshake frame. Snapshot under the lock so a
        # concurrent clear/ set cannot change the set mid-dispatch.
        handlers: list[object] = []
        async with self._lock:
            if payload.get("event") == "speech.asr" and payload.get("state") == "final" and self._asr_final_handlers:
                handlers = list(self._asr_final_handlers.values())
        for handler in handlers:
            try:
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("asr final handler failed")
        async with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except asyncio.QueueEmpty:
                    pass

    def set_asr_final_handler(self, handler) -> int:
        # Returns a token; pass it to clear_asr_final_handler to remove ONLY
        # this handler. A second wake gets a distinct token so the first's
        # finally cannot wipe the second's handler (R3 residual race).
        self._next_token += 1
        token = self._next_token
        self._asr_final_handlers[token] = handler
        return token

    def clear_asr_final_handler(self, token: int | None = None) -> None:
        if token is None:
            self._asr_final_handlers.clear()
        else:
            self._asr_final_handlers.pop(token, None)
