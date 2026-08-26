from __future__ import annotations

import asyncio
import importlib.util

import numpy as np
import pytest

from fusion_speech.config import load_config
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.wake_word import PRETRAINED_WAKE_WORDS, WakeWordEngine


class _FakeSource:
    # Stand-in for AudioSource: owns a ring, start/stop are no-ops, exposes the
    # ring + .source/.ring attrs the engine reads.
    def __init__(self, ring):
        self.ring = ring
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class _FakeOwwModel:
    # openwakeword.Model stand-in: predict() returns a scheduled score for the
    # active word, keyed the same way openwakeword does (splitext of basename ->
    # "hey_jarvis_v0" for hey_jarvis_v0.1.onnx). The engine's _score_key uses the
    # same derivation, so the keys line up.
    def __init__(self, config, words):
        self.words = list(words)
        self.scores: list[float] = []
        self._i = 0
        self.resets = 0

    def set_scores(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self._i = 0

    def predict(self, chunk: np.ndarray) -> dict:
        import os

        s = self.scores[min(self._i, len(self.scores) - 1)] if self.scores else 0.0
        self._i += 1
        key = os.path.splitext(PRETRAINED_WAKE_WORDS[self.words[0]])[0]
        return {key: s}

    def reset(self):
        self.resets += 1


def _make_engine(scores=None, *, words=None, source=None, threshold=0.5, debounce=0.0):
    cfg = load_config()
    cfg.wake_default_word = words or "hey_jarvis"
    cfg.wake_threshold = threshold
    cfg.wake_debounce_s = debounce
    fired = []
    model_holder = {}

    def builder(c, w):
        model_holder["model"] = _FakeOwwModel(c, w)
        if scores is not None:
            model_holder["model"].set_scores(scores)
        return model_holder["model"]

    eng = WakeWordEngine(cfg, on_wake=lambda word: fired.append(word), audio_source=source, model_builder=builder)
    return eng, fired, model_holder


@pytest.mark.asyncio
async def test_engine_fires_on_wake_above_threshold():
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    # score jumps to 0.9 (>0.5) after the first chunk -> one wake fire. debounce
    # 5s blocks the remaining frames from re-firing (one utterance = one wake).
    eng, fired, holder = _make_engine(scores=[0.9, 0.9, 0.9], source=src, debounce=5.0)
    assert eng.enabled is True
    res = await eng.start()
    assert res["armed"] is True
    # feed enough 1280-sample frames to trigger predict
    for _ in range(4):
        ring.write(np.zeros(1280, dtype=np.float32))
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)
    await eng.stop()
    assert len(fired) == 1
    assert fired[0] == "hey_jarvis"
    assert eng._fires == 1


@pytest.mark.asyncio
async def test_engine_no_fire_below_threshold():
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    eng, fired, holder = _make_engine(scores=[0.1, 0.1, 0.1], source=src)
    await eng.start()
    for _ in range(4):
        ring.write(np.zeros(1280, dtype=np.float32))
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)
    await eng.stop()
    assert len(fired) == 0
    assert eng._fires == 0


@pytest.mark.asyncio
async def test_cooldown_blocks_repeat_fires():
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    # threshold met every frame but debounce=10s -> only one fire total.
    eng, fired, holder = _make_engine(scores=[0.9] * 12, source=src, debounce=10.0)
    await eng.start()
    for _ in range(12):
        ring.write(np.zeros(1280, dtype=np.float32))
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    await eng.stop()
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_pause_stops_detection():
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    eng, fired, holder = _make_engine(scores=[0.9] * 8, source=src)
    await eng.start()
    eng.pause()
    assert eng.paused is True
    # feed audio while paused: drain skips the ring, no predict, no fire
    for _ in range(6):
        ring.write(np.zeros(1280, dtype=np.float32))
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert len(fired) == 0
    eng.resume()
    assert eng.paused is False
    await eng.stop()


@pytest.mark.asyncio
async def test_resume_resets_model():
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    eng, fired, holder = _make_engine(scores=[0.0], source=src)
    await eng.start()
    eng.pause()
    eng.resume()
    assert holder["model"].resets >= 1
    await eng.stop()


@pytest.mark.asyncio
async def test_fail_soft_when_model_build_raises():
    cfg = load_config()
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)

    def bad_builder(c, w):
        raise RuntimeError("boom")

    eng = WakeWordEngine(cfg, on_wake=lambda w: None, audio_source=src, model_builder=bad_builder)
    assert eng.enabled is False
    assert eng.model is None
    res = await eng.start()
    assert res["armed"] is False


@pytest.mark.asyncio
async def test_set_hotwords_rebuilds_with_known_words():
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    eng, fired, holder = _make_engine(scores=[0.0], source=src)
    accepted = eng.set_hotwords(["hey_mycroft", "alexa"])
    assert accepted == ["hey_mycroft", "alexa"]
    assert eng.words == ["hey_mycroft", "alexa"]


@pytest.mark.asyncio
async def test_set_hotwords_drops_unknown_words():
    eng, fired, holder = _make_engine(scores=[0.0])
    accepted = eng.set_hotwords(["hey_jarvis", "nonexistent_word"])
    assert accepted == ["hey_jarvis"]
    assert "nonexistent_word" not in eng.words


@pytest.mark.asyncio
async def test_async_on_wake_is_awaited():
    # on_wake may return a coroutine (the daemon's _on_wake does); the engine
    # must schedule it, not drop it.
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    cfg = load_config()
    cfg.wake_threshold = 0.5
    cfg.wake_debounce_s = 0.0
    done = asyncio.Event()

    async def on_wake(word):
        done.set()

    def builder(c, w):
        m = _FakeOwwModel(c, w)
        m.set_scores([0.9])
        return m

    eng = WakeWordEngine(cfg, on_wake=on_wake, audio_source=src, model_builder=builder)
    await eng.start()
    for _ in range(4):
        ring.write(np.zeros(1280, dtype=np.float32))
        await asyncio.sleep(0.02)
    await asyncio.wait_for(done.wait(), timeout=2.0)
    await eng.stop()
    assert done.is_set()


OWW_PRESENT = importlib.util.find_spec("openwakeword") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not OWW_PRESENT, reason="openwakeword not installed")
async def test_real_openwakeword_loads_and_predicts_silence():
    # Real-load gate (project rule: really load the model). Constructs a real
    # openwakeword.Model downloading hey_jarvis + feature models to the cache
    # dir, feeds silence -> scores stay low (no false wake). Skips if the
    # download fails (network/mirror). Models are kept (final output).
    cfg = load_config()
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    eng = WakeWordEngine(cfg, on_wake=lambda w: None, audio_source=src)
    if not eng.enabled:
        pytest.skip("openwakeword model download failed or model unavailable")
    await eng.start()
    silence = np.zeros(1280, dtype=np.float32)
    max_score = 0.0
    for _ in range(25):  # 2s of silence in 80ms frames
        ring.write(silence)
        await asyncio.sleep(0.02)
        if eng._last_scores:
            max_score = max(max_score, max(eng._last_scores.values()))
    await eng.stop()
    # silence must not cross the wake threshold
    assert max_score < cfg.wake_threshold, f"false wake on silence: max={max_score}"
