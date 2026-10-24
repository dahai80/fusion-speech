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
    # openwakeword.Model stand-in. Exposes class_mapping the way the real one
    # does (key = splitext(basename(model_file + ".onnx"))[0] -> "hey_jarvis_v0.1"
    # for hey_jarvis_v0.1, NOT the old wrong splitext(word)[0] = "hey_jarvis_v0").
    # The engine's _resolve_score_keys reads class_mapping; predict() must key
    # with the same real key or the score is never read (audit F1).
    def __init__(self, config, words):
        import os

        self.words = list(words)
        self.scores: list[float] = []
        self._i = 0
        self.resets = 0
        self._keys: list[str] = []
        for w in self.words:
            model_file = PRETRAINED_WAKE_WORDS.get(w, w)
            self._keys.append(os.path.splitext(os.path.basename(model_file + ".onnx"))[0])
        self.class_mapping = {k: {"0": "0"} for k in self._keys}
        self.models = {k: object() for k in self._keys}

    def set_scores(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self._i = 0

    def predict(self, chunk: np.ndarray) -> dict:
        s = self.scores[min(self._i, len(self.scores) - 1)] if self.scores else 0.0
        self._i += 1
        return {self._keys[0]: s}

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
    # R2: construction no longer loads; explicit load() decides enabled.
    await eng.load()
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
    # R2: enabled is None (undecided) at construction; load() runs the failing
    # builder off the loop and fail-softs to enabled=False.
    assert eng.enabled is None
    await eng.load()
    assert eng.enabled is False
    assert eng.model is None
    res = await eng.start()
    assert res["armed"] is False


@pytest.mark.asyncio
async def test_set_hotwords_rebuilds_with_known_words():
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)
    eng, fired, holder = _make_engine(scores=[0.0], source=src)
    accepted = await eng.set_hotwords(["hey_mycroft", "alexa"])
    assert accepted == ["hey_mycroft", "alexa"]
    assert eng.words == ["hey_mycroft", "alexa"]


@pytest.mark.asyncio
async def test_set_hotwords_drops_unknown_words():
    eng, fired, holder = _make_engine(scores=[0.0])
    accepted = await eng.set_hotwords(["hey_jarvis", "nonexistent_word"])
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
    # R2: load() downloads + builds off the event loop (fail-soft if the mirror
    # is unreachable). start() would also load, but we load first so a download
    # failure skips cleanly instead of surfacing as armed=False mid-test.
    await eng.load()
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


@pytest.mark.asyncio
async def test_r1_predict_not_blocking():
    # R1: model.predict must run off the event loop. A predict that sleeps 50ms
    # in the executor must NOT block a concurrent asyncio.sleep coroutine from
    # completing within that window — if predict were inline (sync), the loop
    # would be held and the sleep could not progress.
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)

    class _SlowModel(_FakeOwwModel):
        def predict(self, chunk):
            # blocks the executor thread ~200ms; the event loop must stay free
            import time as _t

            _t.sleep(0.20)
            return {self._keys[0]: 0.0}

    cfg = load_config()
    cfg.wake_threshold = 0.5
    cfg.wake_debounce_s = 0.0

    def builder(c, w):
        return _SlowModel(c, w)

    eng = WakeWordEngine(cfg, on_wake=lambda w: None, audio_source=src, model_builder=builder)
    await eng.start()
    ring.write(np.zeros(1280, dtype=np.float32))
    # let the drain pull the chunk and dispatch the slow predict to the executor
    await asyncio.sleep(0.02)
    # while predict blocks in the executor, a loop coroutine must complete:
    # three 20ms sleeps (~60ms) run concurrently with the 50ms predict and
    # should finish in ~60ms, not 50+60ms (inline predict would serialize).
    t0 = asyncio.get_event_loop().time()
    await asyncio.sleep(0.02)
    await asyncio.sleep(0.02)
    await asyncio.sleep(0.02)
    elapsed = asyncio.get_event_loop().time() - t0
    await eng.stop()
    # the loop was free to schedule the sleeps while predict ran in the
    # executor: total elapsed is ~60ms (concurrent), not 200ms predict +
    # 60ms sleeps = 260ms (inline predict would serialize). Threshold 150ms
    # cleanly separates concurrent (<150ms) from blocked (>=200ms) even on a
    # loaded machine where asyncio.sleep overshoots.
    assert elapsed < 0.15, f"predict blocked the event loop: elapsed={elapsed:.3f}s"


@pytest.mark.asyncio
async def test_r2_download_off_loop():
    # R2: the 120s model download must run off the event loop, and __init__ must
    # NOT trigger any load (construction is cheap, enabled stays None). We
    # monkeypatch the downloader's ensure_models to sleep 50ms, then start() the
    # engine (which loads on demand). A concurrent coroutine must progress while
    # the download runs in the executor.
    cfg = load_config()
    cfg.wake_default_word = "hey_jarvis"
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)

    import time as _t

    from fusion_speech import wake_word as ww

    called = {"ensure": 0}

    class _SlowDownloader:
        @staticmethod
        def ensure_models(words, cache_dir, base_url, *, client_factory=None):
            called["ensure"] += 1
            # simulate a slow download in the executor thread (300ms gives a
            # wide margin vs the ~50ms of measured sleeps, so a loaded machine
            # cannot false-fail; inline download would serialize to >=300ms).
            _t.sleep(0.30)
            # return dummy paths so _load_model proceeds to build the real model
            paths = []
            for w in words:
                f = ww.PRETRAINED_WAKE_WORDS.get(w)
                if f is not None:
                    paths.append(f"{cache_dir}/{f}.onnx")
            return paths

    # construction must be cheap and not call the downloader
    eng = WakeWordEngine(
        cfg,
        on_wake=lambda w: None,
        audio_source=src,
        downloader=_SlowDownloader,
    )
    assert eng.enabled is None, "construction triggered a load (R2 regression)"
    assert called["ensure"] == 0, "construction called ensure_models (R2 regression)"

    # start() loads on demand in the executor; the loop stays free. But the real
    # openwakeword build needs real ONNX files — without them _load_model raises
    # and load() fail-softs to enabled=False. We assert the load ran off the
    # loop (a concurrent coroutine progressed during it) regardless of outcome.
    t0 = asyncio.get_event_loop().time()
    # run start() and a concurrent sleep together
    start_task = asyncio.create_task(eng.start())
    await asyncio.sleep(0.01)  # let start() reach the load() path
    await asyncio.sleep(0.02)
    await asyncio.sleep(0.02)
    elapsed = asyncio.get_event_loop().time() - t0
    # safety timeout only — the concurrency assertion is `elapsed` above. The
    # executor load includes a COLD import of openwakeword+onnxruntime (first
    # test in the suite to touch them, ~7s standalone, more under load), so
    # 10s false-failed; 60s bounds a hung load without racing a slow import.
    res = await asyncio.wait_for(start_task, timeout=60.0)
    # the downloader ran (load was attempted off the loop)
    assert called["ensure"] >= 1, "start() did not trigger a download"
    # the loop stayed free during the executor download: elapsed ~50ms
    # (concurrent), not 300ms download + 50ms sleeps = 350ms (inline download
    # would block). Threshold 200ms separates concurrent (<200ms) from blocked
    # (>=300ms) even on a loaded machine.
    assert elapsed < 0.20, f"download blocked the event loop: elapsed={elapsed:.3f}s"
    # without real ONNX files the build fails -> fail-soft armed False (acceptable)
    assert res["armed"] in (True, False)


@pytest.mark.asyncio
async def test_r7_fixed_chunk_read():
    # R7: drain must read a FIXED 1280-sample frame per predict pass, not the
    # variable `available - (available % 1280)` (which fed 2560/3840-sample
    # frames and corrupted openwakeword scores). Write 2560 into the ring; the
    # first predict sees exactly 1280, the leftover 1280 stays for the next
    # pass. Captures the size each predict() received.
    ring = RingBuffer(16000 * 5)
    src = _FakeSource(ring)

    seen_sizes: list[int] = []

    class _SizeCapturingModel(_FakeOwwModel):
        def predict(self, chunk: np.ndarray) -> dict:
            seen_sizes.append(int(chunk.size))
            return {self._keys[0]: 0.0}

    cfg = load_config()
    cfg.wake_threshold = 0.5
    cfg.wake_debounce_s = 0.0

    def builder(c, w):
        return _SizeCapturingModel(c, w)

    eng = WakeWordEngine(cfg, on_wake=lambda w: None, audio_source=src, model_builder=builder)
    await eng.start()
    # write 2560 samples = exactly two 1280 frames
    ring.write(np.zeros(2560, dtype=np.float32))
    # let two drain passes run
    await asyncio.sleep(0.08)
    await eng.stop()
    # each predict got a fixed 1280 (never 2560), leftover retained not dropped
    assert seen_sizes, "drain never ran a predict"
    assert all(s == 1280 for s in seen_sizes), f"non-fixed frame sizes: {seen_sizes}"
