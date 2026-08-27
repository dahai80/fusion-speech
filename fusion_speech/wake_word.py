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
        # R2: model load (and the 120s httpx download it triggers) is deferred
        # out of __init__ to async load(); enabled stays None (undecided) until
        # then, so constructing the engine never blocks the event loop.
        self.enabled = None
        self.threshold = config.wake_threshold
        self._debounce_s = config.wake_debounce_s
        # R7: openwakeword Model.predict contracts a fixed 1280-sample (80ms)
        # frame. Read exactly self._chunk per drain pass; leftover stays in the
        # ring for next pass. Variable-length reads (available % chunk) fed the
        # model 2560/3840-sample frames and silently corrupted wake scores.
        self._chunk = config.wake_chunk_samples
        self._cooldown_until: float = 0.0
        self._paused = False
        self._active = False
        self._drain_task: asyncio.Task | None = None
        self._wake_tasks: set = set()
        self._last_scores: dict[str, float] = {}
        self._fires = 0
        self._score_keys: dict[str, str] = {}

    async def load(self) -> None:
        # R2: load the model (and download wake ONNX on first arm) off the event
        # loop. Idempotent: a no-op once enabled is decided. Fail-soft: on
        # failure the engine stays disabled (enabled=False), daemon unharmed.
        if self.enabled is not None:
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            self._score_keys = self._resolve_score_keys()
            logger.info("wake score keys resolved: %s", self._score_keys)
        except Exception as e:
            logger.warning("wake engine unavailable, disabled: %s", e)
            self.model = None
            self.enabled = False

    async def _reload(self) -> None:
        # R2: rebuild the model off the event loop (set_hotwords swap). The
        # download/rebuild is sync and long; run it in an executor so the loop
        # is not blocked. _load_model sets self.enabled on success.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self._score_keys = self._resolve_score_keys()

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

    async def set_hotwords(self, words: list[str]) -> list[str]:
        # Swap the active wake words at runtime. Only pretrained names are usable;
        # unknown words are dropped (logged). Rebuilds the openwakeword Model off
        # the event loop (R2 _reload). On reload failure roll back to the previous
        # model/words/enabled rather than silently half-disabling the engine (A4).
        known = [w for w in words if w in PRETRAINED_WAKE_WORDS]
        dropped = [w for w in words if w not in PRETRAINED_WAKE_WORDS]
        if dropped:
            logger.warning("unknown wake words have no pretrained model, dropped: %s", dropped)
        if not known:
            logger.warning("set_hotwords: no known words, keep current %s", self.words)
            return list(self.words)
        prev_words = self.words
        prev_model = self.model
        prev_enabled = self.enabled
        prev_keys = self._score_keys
        self.words = known
        if self.enabled and self._model_builder is None:
            try:
                await self._reload()
            except Exception as e:
                logger.error("set_hotwords reload failed, rolling back: %s", e)
                self.words = prev_words
                self.model = prev_model
                self.enabled = prev_enabled
                self._score_keys = prev_keys
                raise WakeModelError(f"set_hotwords reload failed: {e}") from e
        else:
            self._score_keys = self._resolve_score_keys()
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
        if self.enabled is None:
            # R2: lazy load on first start (download off the event loop). Fail-soft:
            # a load failure sets enabled=False -> armed stays False below.
            await self.load()
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
                if available < self._chunk:
                    await asyncio.sleep(0.02)
                    continue
                chunk = self.ring.read(self._chunk)
                if chunk.size == 0:
                    await asyncio.sleep(0.02)
                    continue
                await self._predict(chunk)
        except asyncio.CancelledError:
            raise

    async def _predict(self, chunk: np.ndarray) -> None:
        # R1: ONNX model.predict blocks the event loop (~5-15ms/frame). Run the
        # pure-sync predict+score in an executor thread so other coroutines (ASR
        # final push, WS write) are not starved. openwakeword Model.predict is
        # not thread-safe (internal ONNX session state), but the drain task is a
        # single serial awaiter — only one frame is in flight at a time, so the
        # default thread pool never contends on the session.
        if self.model is None:
            return
        loop = asyncio.get_event_loop()
        fired_word = await loop.run_in_executor(None, self._predict_sync, chunk)
        if fired_word is not None:
            self._fire_callback(fired_word)

    def _predict_sync(self, chunk: np.ndarray) -> str | None:
        # Pure sync: predict + threshold/cooldown logic. Returns the word that
        # fired (or None). Runs in an executor off the event loop.
        if self.model is None:
            return None
        try:
            scores = self.model.predict(chunk)
        except Exception:
            logger.exception("wake predict failed")
            return None
        if not isinstance(scores, dict):
            return None
        self._last_scores = scores
        now = time.monotonic()
        if now < self._cooldown_until:
            return None
        for word in self.words:
            key = self._score_keys.get(word, self._fallback_score_key(word))
            score = float(scores.get(key, 0.0))
            if score >= self.threshold:
                self._fires += 1
                self._cooldown_until = now + self._debounce_s
                logger.info("wake fired word=%s score=%.3f", word, score)
                return word
        return None

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

    def _resolve_score_keys(self) -> dict[str, str]:
        # openwakeword.Model.predict() keys its score dict by the model name it
        # derives internally as splitext(basename(path))[0] — e.g. file
        # "hey_jarvis_v0.1.onnx" -> key "hey_jarvis_v0.1". Older code wrongly did
        # splitext("hey_jarvis_v0.1")[0] -> "hey_jarvis_v0" (stripped the ".1"),
        # so scores never matched and wake never fired. Read the real keys from
        # the model's class_mapping / models attrs; fall back to the same
        # derivation openwakeword uses if those attrs are absent.
        if self.model is None:
            return {}
        real_keys: list[str] = []
        class_mapping = getattr(self.model, "class_mapping", None)
        if isinstance(class_mapping, dict):
            real_keys = [str(k) for k in class_mapping]
        if not real_keys:
            models = getattr(self.model, "models", None)
            if isinstance(models, dict):
                real_keys = [str(k) for k in models]
        result: dict[str, str] = {}
        for word in self.words:
            if not real_keys:
                result[word] = self._fallback_score_key(word)
                continue
            model_file = PRETRAINED_WAKE_WORDS.get(word, word)
            match = next((k for k in real_keys if k == model_file or k.startswith(model_file)), None)
            result[word] = match if match is not None else self._fallback_score_key(word)
        return result

    @staticmethod
    def _fallback_score_key(word: str) -> str:
        # mirror openwakeword's own derivation: splitext(basename(path))[0]
        model_file = PRETRAINED_WAKE_WORDS.get(word, word)
        return os.path.splitext(os.path.basename(model_file + ".onnx"))[0]
