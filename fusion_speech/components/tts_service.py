from __future__ import annotations

import asyncio
import logging

from fusion_speech.metrics import metrics
from fusion_speech.tts_playback import StreamingPlayer

logger = logging.getLogger(__name__)


class TtsService:
    # Owns the TTS playback player. _play is the leaf operation (build player,
    # stream+play); speak orchestration (wake pause/resume, ASR stop/resume)
    # stays on the facade so test patches on daemon.speak keep hitting the real
    # facade path. stop_tts cancels an in-flight player.
    def __init__(self, config, notifier):
        self.config = config
        self._notifier = notifier
        self.player = None
        self.speaking = False
        # R5: latched when a speak's wake/ASR resume fails. A resume failure
        # leaves wake/ASR in a half-restored state; degraded=True warns clients
        # (and the next speak) that the pipeline is not cleanly armed. Cleared
        # on a successful resume. Not blocking — the TTS itself played.
        self.degraded = False
        # H5: guard the check-then-set of player/speaking so two concurrent
        # _play calls cannot both pass the "already speaking" check and start
        # overlapping players. The lock wraps only the guard + assignment, NOT
        # the long player.speak stream — otherwise stop_tts could not cancel a
        # playing stream (it would block on this lock until speak finished).
        self._lock = asyncio.Lock()

    async def _play(self, text, voice="", model="", *, synthesize=None, stream_factory=None) -> dict:
        async with self._lock:
            if self.player is not None and self.player.active:
                logger.warning("speak rejected: already speaking")
                return {"played": False, "error": "already speaking"}
            player = StreamingPlayer(
                self.config, notifier=self._notifier, synthesize=synthesize, stream_factory=stream_factory
            )
            self.player = player
            self.speaking = True
            metrics.speaking.set(1)
            metrics.tts_speak_total.inc()
        try:
            result = await player.speak(text, voice=voice, model=model)
        finally:
            async with self._lock:
                self.speaking = False
                self.player = None
                metrics.speaking.set(0)
        return result

    async def stop_tts(self) -> dict:
        # No lock: must be able to interrupt a playing stream immediately. Reads
        # self.player (single attr read is atomic under GIL) and cancels.
        player = self.player
        logger.info("speech.stop_tts active=%s", player is not None)
        if player is None:
            return {"stopped": False}
        return await player.cancel()
