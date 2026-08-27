from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable

import numpy as np

from fusion_speech.audio_source import AudioSource
from fusion_speech.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

_RING_CAPACITY = 16000 * 5  # 5s @ 16k, drop-oldest on overrun
_CHUNK_16K = 1280  # 80ms @ 16k — openwakeword native frame size

# pretrained openwakeword wake words shipped as release assets (name -> model file).
# All available at <release_base>/<name>_v0.1.onnx. Source: openwakeword.MODELS.
PRETRAINED_WAKE_WORDS = {
    "alexa": "alexa_v0.1",
    "hey_mycroft": "hey_mycroft_v0.1",
    "hey_jarvis": "hey_jarvis_v0.1",
    "hey_rhasspy": "hey_rhasspy_v0.1",
    "timer": "timer_v0.1",
    "weather": "weather_v0.1",
}

# feature models every wakeword needs (melspectrogram + embedding). Same release.
_FEATURE_MODELS = ("melspectrogram", "embedding_model")


class WakeModelError(RuntimeError):
    pass


class WakeModelDownloader:
    # Fetches openwakeword ONNX models (wakeword + the two feature models) into a
    # local cache dir via httpx, idempotent (skip if file exists). The base URL is
    # configurable so a mirror can be used; openwakeword's own download_models()
    # has no URL-override hook (hardcoded GitHub via requests), so we roll our own.

    @staticmethod
    def ensure_models(
        words: list[str],
        cache_dir: str,
        base_url: str,
        *,
        client_factory: Callable[[], object] | None = None,
    ) -> list[str]:
        # Download every wake word's ONNX model + the shared feature models into
        # cache_dir. Returns the on-disk paths of the wake-word models (in order).
        os.makedirs(cache_dir, exist_ok=True)
        paths: list[str] = []
        for word in words:
            model_file = PRETRAINED_WAKE_WORDS.get(word)
            if model_file is None:
                logger.warning("wake word %r has no pretrained model, skip", word)
                continue
            WakeModelDownloader._ensure_one(f"{model_file}.onnx", cache_dir, base_url, client_factory)
            paths.append(os.path.join(cache_dir, f"{model_file}.onnx"))
        for feat in _FEATURE_MODELS:
            WakeModelDownloader._ensure_one(f"{feat}.onnx", cache_dir, base_url, client_factory)
        return paths

    @staticmethod
    def _ensure_one(filename: str, cache_dir: str, base_url: str, client_factory) -> None:
        target = os.path.join(cache_dir, filename)
        if os.path.exists(target):
            logger.debug("wake model cached: %s", target)
            return
        url = f"{base_url.rstrip('/')}/{filename}"
        logger.info("downloading wake model %s -> %s", url, target)
        if client_factory is not None:
            client = client_factory()
            try:
                _download_with(client, url, target)
            finally:
                close = getattr(client, "aclose", None) or getattr(client, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        close()
            return
        # default: a short-lived httpx client. 120s timeout — GitHub release
        # downloads can be slow/flaky from some networks; the wake engine is
        # fail-soft so a timeout degrades to wake-disabled, not a crash.
        import httpx

        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            _download_with(client, url, target)


def _download_with(client, url: str, target: str) -> None:
    # works for both httpx.Client (sync .get/stream) and a duck-typed client
    get = getattr(client, "get", None)
    if get is None:
        raise WakeModelError(f"client {type(client)!r} has no .get")
    resp = get(url)
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    content = getattr(resp, "content", None)
    if content is None and hasattr(resp, "iter_bytes"):
        content = b"".join(resp.iter_bytes())
    if not content:
        raise WakeModelError(f"empty download from {url}")
    tmp = target + ".part"
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, target)
    logger.info("wake model saved: %s (%d bytes)", target, len(content))


class WakeWordEngine:
    # Always-on wake-word detector. Producer = AudioSource writing PCM into the
    # ring; consumer = the drain task, which reads 1280-sample (80ms) chunks and
    # calls openwakeword.Model.predict. A score >= threshold fires on_wake(word)
    # once, then a cooldown (debounce) blocks repeats. pause()/resume() let the
    # daemon suspend detection during active ASR (so your own speech does not
    # self-trigger). When paused the drain stops reading the ring entirely, so a
    # sharing ASR listener becomes the sole SPSC consumer (no read race).
    # Fail-soft: if openwakeword or the model is unavailable, `enabled` is False
    # and the engine is a no-op (daemon still runs).
    def __init__(
        self,
        config,
        *,
        on_wake: Callable[[str], object] | None = None,
        audio_source: AudioSource | None = None,
        model_builder: Callable[..., object] | None = None,
        downloader: type[WakeModelDownloader] | None = None,
    ):
        self.config = config
        self._on_wake = on_wake
        self._own_source = audio_source is None
        if audio_source is not None:
            self.ring = audio_source.ring
        else:
            self.ring = RingBuffer(_RING_CAPACITY)
        self.source = audio_source or AudioSource(
            self.ring,
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self._model_builder = model_builder
        self._downloader = downloader or WakeModelDownloader
        self.words: list[str] = [config.wake_default_word]
        self.model = None
        self.enabled = False
        self.threshold = config.wake_threshold
        self._debounce_s = config.wake_debounce_s
        self._cooldown_until: float = 0.0
        self._paused = False
        self._active = False
        self._drain_task: asyncio.Task | None = None
        self._wake_tasks: set = set()
        self._last_scores: dict[str, float] = {}
        self._fires = 0
        try:
            self._load_model()
        except Exception as e:
            logger.warning("wake engine unavailable, disabled: %s", e)
            self.model = None
            self.enabled = False

    def _load_model(self) -> None:
        if self._model_builder is not None:
            # test injection: build a fake openwakeword-like Model
            self.model = self._model_builder(self.config, self.words)
            self.enabled = True
            logger.info("wake engine armed (injected model) words=%s", self.words)
            return
        import openwakeword

        cache = self.config.wake_model_dir
        paths = self._downloader.ensure_models(self.words, cache, self.config.wake_model_url)
        if not paths:
            raise WakeModelError(f"no wake-word models available for {self.words}")
        feat = {f: os.path.join(cache, f"{f}.onnx") for f in _FEATURE_MODELS}
        self.model = openwakeword.Model(
            wakeword_models=paths,
            inference_framework="onnx",
            melspec_model_path=feat["melspectrogram"],
            embedding_model_path=feat["embedding_model"],
        )
        self.enabled = True
        logger.info("wake engine armed words=%s threshold=%.2f", self.words, self.threshold)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def own_source(self) -> bool:
        return self._own_source

    def set_hotwords(self, words: list[str]) -> list[str]:
        # Swap the active wake words at runtime. Only pretrained names are usable;
        # unknown words are dropped (logged). Rebuilds the openwakeword Model.
        known = [w for w in words if w in PRETRAINED_WAKE_WORDS]
        dropped = [w for w in words if w not in PRETRAINED_WAKE_WORDS]
        if dropped:
            logger.warning("unknown wake words have no pretrained model, dropped: %s", dropped)
        if not known:
            logger.warning("set_hotwords: no known words, keep current %s", self.words)
            return list(self.words)
        self.words = known
        if self.enabled and self._model_builder is None:
            try:
                self._load_model()
            except Exception as e:
                logger.warning("set_hotwords reload failed, wake disabled: %s", e)
                self.model = None
                self.enabled = False
        return list(self.words)

    def pause(self) -> None:
        if not self._paused:
            self._paused = True
            logger.info("wake engine paused")

    def resume(self) -> None:
        if self._paused:
            self._paused = False
            if self.model is not None and hasattr(self.model, "reset"):
                with contextlib.suppress(Exception):
                    self.model.reset()
            logger.info("wake engine resumed")

    async def start(self) -> dict:
        if self._active:
            logger.warning("wake engine already active")
            return {"armed": True, "already": True}
        if not self.enabled or self.model is None:
            logger.warning("wake engine not enabled, start is a no-op")
            return {"armed": False, "enabled": False}
        self._paused = False
        self._cooldown_until = 0.0
        self._fires = 0
        if self.model is not None and hasattr(self.model, "reset"):
            with contextlib.suppress(Exception):
                self.model.reset()
        try:
            self.source.start()
        except Exception as e:
            logger.error("cannot start audio source for wake: %s", e)
            raise WakeModelError(f"cannot start audio source: {e}") from e
        self._active = True
        self._drain_task = asyncio.create_task(self._drain_loop())
        logger.info("wake drain started words=%s", self.words)
        return {"armed": True, "words": list(self.words)}

    async def stop(self) -> dict:
        if not self._active:
            return {"armed": False}
        self._active = False
        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None
        for t in list(self._wake_tasks):
            t.cancel()
        self._wake_tasks.clear()
        if self._own_source:
            self.source.stop()
        logger.info("wake engine stopped fires=%d", self._fires)
        return {"armed": False, "fires": self._fires}

    async def _drain_loop(self) -> None:
        # Consumer: pull 1280-sample frames and predict. While paused the drain
        # does NOT read the ring (a sharing ASR listener is sole consumer then).
        # Fires on_wake on threshold cross, guarded by a cooldown so one
        # utterance = one wake.
        try:
            while self._active:
                if self._paused:
                    await asyncio.sleep(0.02)
                    continue
                available = self.ring.available()
                if available < _CHUNK_16K:
                    await asyncio.sleep(0.02)
                    continue
                chunk = self.ring.read(available - (available % _CHUNK_16K))
                if chunk.size == 0:
                    await asyncio.sleep(0.02)
                    continue
                self._predict(chunk)
        except asyncio.CancelledError:
            raise

    def _predict(self, chunk: np.ndarray) -> None:
        if self.model is None:
            return
        try:
            scores = self.model.predict(chunk)
        except Exception:
            logger.exception("wake predict failed")
            return
        if not isinstance(scores, dict):
            return
        self._last_scores = scores
        now = time.monotonic()
        if now < self._cooldown_until:
            return
        for word in self.words:
            key = self._score_key(word)
            score = float(scores.get(key, 0.0))
            if score >= self.threshold:
                self._fires += 1
                self._cooldown_until = now + self._debounce_s
                logger.info("wake fired word=%s score=%.3f", word, score)
                self._fire_callback(word)
                return

    def _fire_callback(self, word: str) -> None:
        if self._on_wake is None:
            return
        try:
            result = self._on_wake(word)
        except Exception:
            logger.exception("wake on_wake callback failed")
            return
        if asyncio.iscoroutine(result):
            task = asyncio.ensure_future(result)
            self._wake_tasks.add(task)
            task.add_done_callback(self._on_wake_done)

    def _on_wake_done(self, task: asyncio.Task) -> None:
        self._wake_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("wake on_wake task failed: %r", exc)

    @staticmethod
    def _score_key(word: str) -> str:
        # openwakeword model name == basename without extension (e.g. hey_jarvis_v0.1)
        model_file = PRETRAINED_WAKE_WORDS.get(word, word)
        return os.path.splitext(model_file)[0]
