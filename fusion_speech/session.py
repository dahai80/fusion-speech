from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

SESSION_IDLE = "idle"
SESSION_LISTENING = "listening"
SESSION_SPEAKING = "speaking"
SESSION_INTERRUPTED = "interrupted"
SESSION_CLOSED = "closed"


class VoiceSession:
    # M-D (issue #2) reusable state machine: listening -> speaking ->
    # interrupted -> listening. Barge-in: a VAD onset while speaking
    # (full-duplex — the browser owns AEC, so server-side VAD hears the user,
    # not the playback) latches the interrupt; end_speak then lands back in
    # listening, ready for the interrupting utterance's ASR final. A
    # low-confidence / empty ASR final emits speech.reprompt so the caller can
    # re-ask instead of acting on garbage. Importable standalone (drive via
    # begin_speak / on_vad_onset / on_asr_final) and wired by the daemon's
    # WS ingress.
    def __init__(
        self,
        session_id: str,
        *,
        barge_in: bool = True,
        reprompt_min_confidence: float = 0.0,
        on_event: Callable[[dict], Awaitable[None]] | None = None,
    ):
        self.session_id = session_id
        self.barge_in = barge_in
        self.reprompt_min_confidence = reprompt_min_confidence
        self._on_event = on_event
        self._state = SESSION_IDLE
        self._barge_in_count = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def speaking(self) -> bool:
        return self._state == SESSION_SPEAKING

    @property
    def barge_in_count(self) -> int:
        return self._barge_in_count

    async def _emit(self, payload: dict) -> None:
        if self._on_event is None:
            return
        try:
            await self._on_event(payload)
        except Exception:
            logger.exception("session %s event handler failed", self.session_id)

    async def _transition(self, new_state: str) -> None:
        old = self._state
        self._state = new_state
        logger.info("session %s: %s -> %s", self.session_id, old, new_state)
        await self._emit({"event": "speech.session", "session": self.session_id, "state": new_state, "previous": old})

    async def start(self) -> None:
        if self._state == SESSION_CLOSED:
            raise RuntimeError(f"session {self.session_id} already closed")
        if self._state != SESSION_IDLE:
            return
        await self._transition(SESSION_LISTENING)

    async def begin_speak(self) -> bool:
        # listening -> speaking. False (no transition) when not listening —
        # callers use the return to decide whether to start synthesis at all.
        if self._state != SESSION_LISTENING:
            logger.warning("session %s begin_speak rejected in state %s", self.session_id, self._state)
            return False
        await self._transition(SESSION_SPEAKING)
        return True

    async def on_vad_onset(self) -> dict | None:
        # Speech onset while TTS plays. Only meaningful in full-duplex with
        # barge_in armed; returns the barge-in event (None when ignored) so
        # the caller learns whether to cancel playback.
        if self._state != SESSION_SPEAKING or not self.barge_in:
            return None
        self._barge_in_count += 1
        event = {
            "event": "speech.barge_in",
            "session": self.session_id,
            "count": self._barge_in_count,
            "ts": time.time(),
        }
        await self._emit(event)
        await self._transition(SESSION_INTERRUPTED)
        return event

    async def end_speak(self, *, cancelled: bool = False) -> str:
        # speaking/interrupted -> listening. cancelled=True means playback
        # was aborted (tts cancel or barge-in); the state already reflects
        # the interrupt, so both paths land in listening either way.
        if self._state in (SESSION_SPEAKING, SESSION_INTERRUPTED):
            await self._transition(SESSION_LISTENING)
        return self._state

    async def on_asr_final(self, payload: dict) -> dict | None:
        # Low-confidence branch (M-D): an empty transcript or an
        # engine-reported confidence below the threshold emits speech.reprompt
        # so the caller can re-ask. None = final accepted.
        if self._state == SESSION_CLOSED:
            return None
        text = (payload.get("text") or "").strip()
        confidence = payload.get("confidence")
        low = not text
        if confidence is not None and self.reprompt_min_confidence > 0:
            low = low or float(confidence) < self.reprompt_min_confidence
        if not low:
            return None
        event = {
            "event": "speech.reprompt",
            "session": self.session_id,
            "reason": "empty_transcript" if not text else "low_confidence",
            "confidence": confidence,
            "ts": time.time(),
        }
        await self._emit(event)
        return event

    async def stop(self) -> None:
        if self._state == SESSION_CLOSED:
            return
        await self._transition(SESSION_CLOSED)
