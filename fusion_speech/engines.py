from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from fusion_speech.mlx_bridge import synthesize_stream, transcribe_stream
from fusion_speech.stt_stream import pcm_to_wav

logger = logging.getLogger(__name__)


class AsrEngineError(RuntimeError):
    pass


class ASREngine(ABC):
    # M-C (issue #2) streaming ASR contract: start_stream opens a recognition
    # stream, feed pushes 16 kHz mono float32 samples, finish flushes and
    # returns the final result dict, cancel aborts without emitting. Cloud
    # engines (e.g. iFlytek WS ASR) implement the same four calls + on_event,
    # so the session layer never branches on engine kind.
    name = "asr"

    def __init__(self, *, on_event: Callable[[dict], Awaitable[None]] | None = None):
        self.on_event = on_event

    @abstractmethod
    async def start_stream(self, *, language: str = "") -> None: ...

    @abstractmethod
    async def feed(self, pcm: np.ndarray) -> None: ...

    @abstractmethod
    async def finish(self) -> dict: ...

    @abstractmethod
    async def cancel(self) -> None: ...

    async def _emit(self, payload: dict) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(payload)
        except Exception:
            logger.exception("asr engine event handler failed")


class TTSEngine(ABC):
    # M-C streaming TTS contract. ASR streams IN (feed), TTS streams OUT, so
    # the shared shape is start/cancel + a stream: synthesize yields a WAV
    # byte stream (RIFF header first, PCM body after) chunk by chunk; the
    # consumer aborts by closing the iterator (aclose / break + GC). Cloud
    # engines implement the same call. ref_audio/ref_text carry zero-shot
    # voice cloning where the backend supports it (fusion-mlx: /audio/speech
    # does not accept them yet — forwarded only when set, see upstream issue).
    name = "tts"

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        model: str = "",
        ref_audio: str = "",
        ref_text: str = "",
    ) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def cancel(self) -> None: ...


class LocalASREngine(ASREngine):
    # Batch adapter over the fusion-mlx delegation bridge: feed buffers PCM,
    # finish encodes one 16k mono WAV and posts /audio/transcriptions. The
    # daemon's VAD-segmented mic path (StreamingListener) predates the engine
    # layer and stays as-is; this adapter is the pluggable one-shot contract.
    name = "local"

    def __init__(self, config, *, transcribe: Callable[..., Awaitable[dict]] | None = None, on_event=None):
        super().__init__(on_event=on_event)
        self.config = config
        self._transcribe = transcribe or transcribe_stream
        self._chunks: list[np.ndarray] = []
        self._started = False
        self._language = ""

    async def start_stream(self, *, language: str = "") -> None:
        self._chunks = []
        self._language = language or self.config.language
        self._started = True
        logger.info("local asr engine stream start lang=%s", self._language)
        await self._emit({"event": "speech.asr.engine", "engine": self.name, "state": "listening"})

    async def feed(self, pcm: np.ndarray) -> None:
        if not self._started:
            raise AsrEngineError("asr engine stream not started")
        if pcm.size:
            self._chunks.append(np.asarray(pcm, dtype=np.float32).reshape(-1))

    async def finish(self) -> dict:
        if not self._started:
            raise AsrEngineError("asr engine stream not started")
        self._started = False
        pcm = np.concatenate(self._chunks) if self._chunks else np.empty(0, dtype=np.float32)
        self._chunks = []
        duration = round(pcm.size / self.config.sample_rate, 3)
        if pcm.size == 0:
            logger.info("local asr engine finish: empty stream")
            return {"text": "", "language": self._language, "duration": 0.0}
        wav = pcm_to_wav(pcm, self.config.sample_rate)
        await self._emit(
            {"event": "speech.asr.engine", "engine": self.name, "state": "transcribing", "duration": duration}
        )
        result = await self._transcribe(self.config, wav, filename="engine.wav", language=self._language)
        logger.info("local asr engine finish: %d samples -> %r", pcm.size, (result.get("text") or "")[:80])
        return result

    async def cancel(self) -> None:
        self._started = False
        self._chunks = []
        logger.info("local asr engine stream cancelled")


class LocalTTSEngine(TTSEngine):
    # Wraps mlx_bridge.synthesize_stream (fusion-mlx /audio/speech, WAV out).
    # ref_audio/ref_text are forwarded only when set — fusion-mlx currently
    # ignores/rejects unknown body fields, so defaults must stay clean.
    name = "local"

    def __init__(self, config, *, synthesize=None):
        self.config = config
        self._synthesize = synthesize or synthesize_stream
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def synthesize(self, text: str, *, voice: str = "", model: str = "", ref_audio: str = "", ref_text: str = ""):
        return self._run(text, voice, model, ref_audio, ref_text)

    async def _run(self, text: str, voice: str, model: str, ref_audio: str, ref_text: str):
        self._active = True
        try:
            async for chunk in self._synthesize(
                self.config,
                text,
                voice=voice,
                model=model,
                ref_audio=ref_audio,
                ref_text=ref_text,
            ):
                yield chunk
        finally:
            self._active = False

    async def cancel(self) -> None:
        # Real abort is the consumer closing the iterator (aclose propagates
        # into the bridge stream). This marks intent for parity with ASREngine.
        self._active = False
        logger.info("local tts engine cancel requested")


def tts_engine_adapter(engine: TTSEngine, *, ref_audio: str = "", ref_text: str = ""):
    # StreamingPlayer calls synthesize(config, text, voice=..., model=...,
    # stream=True) and iterates the RESULT with `async for` — so the adapter
    # is a plain def returning the engine's async generator, not an async def
    # (that would hand the player a coroutine). Any registered engine drops
    # into the existing player without changes.
    def _synthesize(config, text, *, voice: str = "", model: str = "", stream: bool = True):
        return engine.synthesize(text, voice=voice, model=model, ref_audio=ref_audio, ref_text=ref_text)

    return _synthesize


_ASR_FACTORIES: dict[str, Callable[..., ASREngine]] = {}
_TTS_FACTORIES: dict[str, Callable[..., TTSEngine]] = {}


def register_asr_engine(name: str, factory: Callable[..., ASREngine]) -> None:
    # Extension point for cloud engines (iFlytek WS ASR etc.): downstream
    # projects register a factory(config, **kwargs) -> ASREngine and select it
    # per session by name. No fork needed — the base admits the cloud path.
    _ASR_FACTORIES[name] = factory
    logger.info("asr engine registered: %s", name)


def register_tts_engine(name: str, factory: Callable[..., TTSEngine]) -> None:
    _TTS_FACTORIES[name] = factory
    logger.info("tts engine registered: %s", name)


def registered_asr_engines() -> list[str]:
    return sorted(_ASR_FACTORIES)


def registered_tts_engines() -> list[str]:
    return sorted(_TTS_FACTORIES)


def make_asr_engine(name: str, config, **kwargs) -> ASREngine:
    factory = _ASR_FACTORIES.get(name)
    if factory is None:
        raise AsrEngineError(f"unknown asr engine {name!r} (registered: {sorted(_ASR_FACTORIES)})")
    return factory(config, **kwargs)


def make_tts_engine(name: str, config, **kwargs) -> TTSEngine:
    factory = _TTS_FACTORIES.get(name)
    if factory is None:
        raise AsrEngineError(f"unknown tts engine {name!r} (registered: {sorted(_TTS_FACTORIES)})")
    return factory(config, **kwargs)


register_asr_engine("local", LocalASREngine)
register_tts_engine("local", LocalTTSEngine)
