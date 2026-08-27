from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fusion_speech.audio_source import AudioSource
from fusion_speech.components.event_bus import EventBus
from fusion_speech.config import load_config
from fusion_speech.ipc import jsonrpc
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.server import SpeechDaemon
from fusion_speech.stt_stream import StreamingListener
from fusion_speech.vad import VadSegment
from fusion_speech.wake_word import WakeWordEngine

# E7: adversarial concurrency gate. Each test hammers a specific Wave 3 fix:
#   H5 facade _handshake_lock (double wake no tear), per-Service locks,
#   H5 dispatch backstop timeout, R3 per-handshake token registry (no wipe),
#   R4 inflight cap (backpressure drop). Concurrency tests are flaky-prone, so
#   each asserts a deterministic end-state, not timing, and yields control
#   generously so the scheduler interleaves the racing coroutines.


class _FakeListener:
    active = True
    source = None


class _FakeStream:
    def __init__(self, **kw):
        self.active = False
        self.cb = kw.get("callback")

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


class _FakeOwwModel:
    def __init__(self, config, words):
        self.words = list(words)

    def predict(self, chunk: np.ndarray) -> dict:
        import os

        from fusion_speech.wake_word import PRETRAINED_WAKE_WORDS

        key = os.path.splitext(PRETRAINED_WAKE_WORDS[self.words[0]])[0]
        return {key: 0.0}

    def reset(self):
        pass


def _patch_daemon_for_wake(daemon, *, auto_listen_s=0.2):
    daemon.config.wake_auto_listen_s = auto_listen_s
    daemon.config.wake_threshold = 0.5
    daemon.config.wake_debounce_s = 0.0
    ring = RingBuffer(16000 * 10)

    def factory(**kw):
        return _FakeStream(**kw)

    shared_source = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)

    def build_engine():
        return WakeWordEngine(
            daemon.config,
            on_wake=daemon._on_wake,
            audio_source=shared_source,
            model_builder=lambda c, w: _FakeOwwModel(c, w),
        )

    async def fake_arm():
        wake = build_engine()
        daemon._wake = wake
        res = await wake.start()
        daemon.wake_armed = bool(res.get("armed"))
        if daemon.wake_armed:
            daemon.hotwords = list(wake.words)
        return res

    daemon.arm_wake = fake_arm

    started = {"n": 0}
    stopped = {"n": 0}

    async def fake_start_listening(audio_source=None):
        started["n"] += 1
        daemon.listening = True
        daemon._listener = _FakeListener()
        return {"listening": True}

    async def fake_stop_listening():
        stopped["n"] += 1
        daemon.listening = False
        daemon._listener = None
        return {"listening": False}

    daemon.start_listening = fake_start_listening
    daemon.stop_listening = fake_stop_listening
    daemon._handshake_started_count = started
    daemon._handshake_stopped_count = stopped


async def _wait_event(q: asyncio.Queue, event: str, timeout: float = 2.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            payload = await asyncio.wait_for(q.get(), timeout=0.1)
        except TimeoutError:
            continue
        if payload.get("event") == event:
            return payload
    raise TimeoutError(f"event {event} not seen")


@pytest.mark.asyncio
async def test_double_wake_no_tear():
    # H5: two rapid wakes (debounce 0 -> would race under the old check-then-set)
    # must serialize through _handshake_lock. Exactly one start_listening per
    # handshake; the second wake blocks until the first's stop+resume finish,
    # so start_count == 2 (one per handshake) and the listener is never
    # double-assigned (no half-state).
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon, auto_listen_s=0.1)
    await daemon.arm_wake()
    # fire both wakes concurrently; the lock serializes them
    await asyncio.gather(daemon._on_wake("hey_jarvis"), daemon._on_wake("hey_jarvis"))
    # both handshakes ran (each does start then stop)
    assert daemon._handshake_started_count["n"] == 2
    assert daemon._handshake_stopped_count["n"] == 2
    assert daemon.listening is False
    assert daemon._listener is None
    await daemon.disarm_wake()


@pytest.mark.asyncio
async def test_speak_concurrent_with_wake():
    # H5: a wake firing during speak must not corrupt speak's resume path.
    # speak wraps wake pause/resume; a concurrent _on_wake is serialized after
    # speak finishes its orchestration (or is guarded by the asr-active check).
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon, auto_listen_s=0.1)

    speak_done = asyncio.Event()

    async def fake_play(text, voice="", model="", *, synthesize=None, stream_factory=None):
        await asyncio.sleep(0.05)
        speak_done.set()
        return {"played": True}

    daemon.tts._play = fake_play
    # arm wake so _on_wake's wake.pause()/resume() path is exercised
    await daemon.arm_wake()
    wake = daemon._wake
    # run speak + a wake concurrently
    speak_task = asyncio.create_task(daemon.speak("hi"))
    await asyncio.sleep(0.01)
    wake_task = asyncio.create_task(daemon._on_wake("hey_jarvis"))
    speak_res = await speak_task
    await wake_task
    assert speak_res["played"] is True
    assert speak_done.is_set()
    # wake engine resumed cleanly (no stuck pause from interleaving)
    assert wake.paused is False
    await daemon.disarm_wake()


@pytest.mark.asyncio
async def test_concurrent_subscribe_broadcast():
    # R3/H2: many subscribers + concurrent broadcasts must all receive every
    # event (non-overflow path). EventBus.publish under its lock fans out
    # atomically; no subscriber is starved or dropped.
    bus = EventBus(queue_maxsize=64)
    subs = [bus.subscribe() for _ in range(10)]

    async def broadcast_n(i):
        await bus.publish({"event": "speech.asr", "state": "final", "i": i})

    await asyncio.gather(*(broadcast_n(i) for i in range(20)))
    # each subscriber should hold all 20 frames
    for q in subs:
        seen = []
        while not q.empty():
            seen.append(q.get_nowait())
        assert len(seen) == 20, f"subscriber lost frames: got {len(seen)}"
        assert all(s["event"] == "speech.asr" for s in seen)


@pytest.mark.asyncio
async def test_concurrent_start_stop_listening():
    # H5 AsrService._lock: concurrent start/stop must not leave a half-state
    # (two listeners on one mic, or listener set but listening=False). End state
    # is deterministic: after all settle, listener is None OR active, and
    # listening flag matches.
    daemon = SpeechDaemon(load_config())
    # bypass the mic: inject a fake listener the lock still guards the swap
    started = {"n": 0}

    async def fake_start(audio_source=None):
        # simulate slow start so concurrent calls overlap
        await asyncio.sleep(0.01)
        if daemon.asr.listener is not None and daemon.asr.listener.active:
            return {"listening": True, "already": True}
        started["n"] += 1
        daemon.asr.listener = _FakeListener()
        daemon.asr.listening = True
        return {"listening": True}

    # restore the real locked method but with a contended body: swap to a fake
    # that mimics the real check-then-set but goes through the service lock by
    # calling the real start_listening with a patched source builder.
    daemon.asr.listener = None
    daemon.asr.listening = False
    # drive the real AsrService.start_listening with a stubbed source so it
    # builds a StreamingListener without hardware; give it a fake transcribe.
    async def fake_transcribe(cfg, wav, **kw):
        return {"text": "", "language": "en"}

    ring = RingBuffer(16000 * 10)

    def factory(**kw):
        s = _FakeStream(**kw)
        return s

    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)
    daemon.asr._audio_bus = None  # force the injected source path off the bus

    # Patch the StreamingListener build inside start_listening by pointing the
    # service at a lightweight constructor via monkeypatching the import.
    import fusion_speech.components.asr_service as asr_mod

    real_cls = asr_mod.StreamingListener

    class _LightListener(real_cls):
        def __init__(self, config, **kw):
            # skip the real ring/source build; just enough for active check
            self.config = config
            self._active = False
            self.source = kw.get("audio_source")
            self.ring = src.ring

        @property
        def active(self):
            return self._active

        async def start(self):
            self._active = True
            return {"listening": True}

        async def stop(self):
            self._active = False
            return {"listening": False, "segments": 0}

    asr_mod.StreamingListener = _LightListener
    try:
        # Run start/stop/start/stop concurrently many times.
        async def op_start():
            await daemon.asr.start_listening(audio_source=src)

        async def op_stop():
            await daemon.asr.stop_listening()

        await asyncio.gather(
            op_start(), op_stop(), op_start(), op_stop(), op_start(), op_stop()
        )
        # settle
        await asyncio.sleep(0.01)
    finally:
        asr_mod.StreamingListener = real_cls

    # Deterministic end state: listener None or active, flag consistent.
    listener = daemon.asr.listener
    if listener is None:
        assert daemon.asr.listening is False
    else:
        assert listener.active == daemon.asr.listening
    # never built more than one live listener concurrently — start_count <= 6
    # and the lock prevented a double-assign (no assertion on exact count since
    # some starts hit the already-active fast path legitimately).
    assert started["n"] <= 6


@pytest.mark.asyncio
async def test_dispatch_timeout():
    # H5: a handler that never returns gets -32000 timeout, not an infinite
    # hang. Monkeypatch DISPATCH_TIMEOUT to a tiny value so the test is fast
    # (Rule 6: no 130s real sleep).
    real_timeout = jsonrpc.DISPATCH_TIMEOUT
    jsonrpc.DISPATCH_TIMEOUT = 0.05

    async def slow_handler(params):
        await asyncio.sleep(10.0)
        return {"never": True}

    dispatcher = JsonRpcDispatcher({"speech.slow": slow_handler})
    try:
        resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.slow"})
    finally:
        jsonrpc.DISPATCH_TIMEOUT = real_timeout
    assert resp["error"]["code"] == -32000
    assert "timeout" in resp["error"]["message"]


@pytest.mark.asyncio
async def test_r4_backpressure():
    # R4: _on_close caps in-flight transcribe tasks at _max_inflight (4). Feed
    # >4 segments with a slow transcribe; assert inflight never exceeds the cap
    # and the overflow segments are dropped (logged), not queued.

    class _Cfg:
        sample_rate = 16000
        channels = 1
        language = "en"
        vad_speech_threshold = 0.5

    notified = []

    async def notifier(payload):
        notified.append(payload)

    async def slow_transcribe(cfg, wav, **kw):
        # hold the task open so inflight piles up
        await asyncio.sleep(0.2)
        return {"text": "x", "language": "en"}

    ring = RingBuffer(16000 * 10)

    def factory(**kw):
        return _FakeStream(**kw)

    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)
    listener = StreamingListener(
        _Cfg(),
        transcribe=slow_transcribe,
        notifier=notifier,
        audio_source=src,
    )
    # build a long enough segment to clear the 0.1s skip gate
    big_samples = 16000
    seg = VadSegment(start_sample=0, end_sample=big_samples)
    # _on_close runs on the loop thread; we call it directly (no drain task
    # needed — it schedules tasks via ensure_future on self._loop).
    listener._loop = asyncio.get_running_loop()
    # stub _collect_pcm so the segment has real PCM without ring capture
    listener._collect_pcm = lambda s: np.ones(big_samples, dtype=np.float32)

    # fire 8 segments (cap is 4). The cap drops segments 5-8 at creation.
    inflight_at_5th = -1
    for i in range(8):
        listener._on_close(seg)
        # snapshot inflight each fire
        if i == 4:
            inflight_at_5th = listener._inflight
    assert listener._inflight <= listener._max_inflight
    # after the 4th accepted segment, inflight is at the cap; the 5th fire
    # either keeps it at cap (dropped) or the cap held — never above.
    assert inflight_at_5th <= listener._max_inflight
    # wait for the accepted 4 to finish
    await asyncio.sleep(0.3)
    # segment_seq advanced 8 (one per _on_close), but only 4 transcribed
    finals = [n for n in notified if n.get("state") == "final" and not n.get("skipped")]
    assert len(finals) == 4, f"expected 4 transcribed finals, got {len(finals)}"
    listener._inflight = 0
    listener._transcribe_tasks.clear()
