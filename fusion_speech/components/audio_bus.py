from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AudioBus:
    # Centralizes mic ownership so arm_wake and start_listening cannot both
    # spawn an AudioSource over the same ring (H3 SPSC violation). The bus holds
    # ONE canonical AudioSource; the first acquirer builds it (with its own
    # ring + stream factory) and registers it, later acquirers BORROW that same
    # source instead of peeking another service's internals. Coexistence is
    # single-reader at a time: _on_wake pauses wake before arming ASR, and speak
    # stops ASR before TTS, so two consumers never drain the ring concurrently.
    # A refcount defers the real close until the last borrower releases (audit
    # A3: a sharing ASR listener never tears down a borrowed wake ring).
    # Engines still accept an injected audio_source — tests bypass the bus by
    # assigning daemon._listener with a prebuilt source, which is why the
    # production borrow path is exercised only by integration, not unit tests.
    def __init__(self, config):
        self.config = config
        self._source = None
        self._borrowers: set[str] = set()

    @property
    def source(self):
        return self._source

    def register(self, source, owner: str) -> None:
        # First acquirer installs the canonical source. A later acquirer that
        # somehow built its own (e.g. wake armed standalone then ASR started)
        # keeps the existing one and discards the new — single source of truth.
        if self._source is None:
            self._source = source
            logger.info("audio bus registered source for owner=%s", owner)
        self._borrowers.add(owner)

    def borrow(self, owner: str):
        # Return the canonical source for a second consumer, or None if none is
        # registered (caller must build + register its own).
        if self._source is not None:
            self._borrowers.add(owner)
        return self._source

    def release(self, owner: str) -> None:
        self._borrowers.discard(owner)
        if not self._borrowers and self._source is not None:
            try:
                stop = getattr(self._source, "stop", None)
                if stop is not None:
                    stop()
            except Exception:
                logger.exception("audio bus source stop failed")
            self._source = None
            logger.info("audio bus released last borrower, closed source")
