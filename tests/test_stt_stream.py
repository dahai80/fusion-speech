from __future__ import annotations

import asyncio
import wave
from io import BytesIO

import numpy as np
import pytest

from fusion_speech.audio_source import AudioSource
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.stt_stream import StreamingListener, pcm_to_wav
from fusion_speech.vad import SileroVadEngine


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


def _fake_source(ring):
    captured = {}

    def factory(**kwargs):
        s = _FakeStream(**kwargs)
        captured["cb"] = kwargs["callback"]
        captured["stream"] = s
        return s

    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)
    return src, captured


def _make_listener(transcribe, notifier, *, vad_model=None, min_silence_ms=100):
    ring = RingBuffer(16000 * 10)
    src, captured = _fake_source(ring)
    vad = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=0,
        model=vad_model,
        on_open=None,
        on_close=None,
    )
    # rebind callbacks onto the listener via the engine the listener owns
    listener = StreamingListener(
        _Cfg(),
        transcribe=transcribe,
        notifier=notifier,
        audio_source=src,
        vad=vad,
    )
    # point vad callbacks at listener methods (listener.vad is the same obj)
    vad._on_open = listener._on_open
    vad._on_close = listener._on_close
    return listener, captured


class _Cfg:
    sample_rate = 16000
    channels = 1
    language = "en"
    vad_speech_threshold = 0.5


class _FakeVadModel:
    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0

    def reset_state(self):
        pass

    def __call__(self, window):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p


def _feed_window(cb, amp=0.3):
    t = np.arange(512) / 16000
    sig = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(-1, 1)
    cb(sig, 512, None, 0)


async def _collect(notified, want_state, timeout=2.0):
    # wait for a notification whose state == want_state
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for n in list(notified):
            if n.get("state") == want_state:
                return n
        await asyncio.sleep(0.01)
    return None


@pytest.mark.asyncio
async def test_listener_start_stop_lifecycle():
    transcribed = []

    async def transcribe(cfg, wav, **kw):
        transcribed.append(wav)
        return {"text": "hi", "language": "en"}

    notified = []

    async def notifier(payload):
        notified.append(payload)

    model = _FakeVadModel([0.0] * 40)  # all silence, no segment
    listener, captured = _make_listener(transcribe, notifier, vad_model=model)
    res = await listener.start()
    assert res["listening"] is True
    assert listener.active is True
    assert captured["stream"].active is True
    # a listening notification should fire on start
    assert any(n.get("state") == "listening" for n in notified)
    res = await listener.stop()
    assert res["listening"] is False
    assert listener.active is False


@pytest.mark.asyncio
async def test_segment_close_triggers_transcribe_and_final():
    transcribed = []

    async def transcribe(cfg, wav, **kw):
        transcribed.append(wav)
        return {"text": "hello world", "language": "en", "duration": 0.5}

    notified = []

    async def notifier(payload):
        notified.append(payload)

    # schedule: 2 silence, 4 speech, then long silence -> 1 segment
    probs = [0.0, 0.0] + [0.9] * 4 + [0.0] * 20
    model = _FakeVadModel(probs)
    listener, captured = _make_listener(transcribe, notifier, vad_model=model, min_silence_ms=200)
    await listener.start()
    cb = captured["cb"]
    for _ in range(len(probs)):
        _feed_window(cb)
        await asyncio.sleep(0.001)
    # wait for the transcribe + final notification
    final = await _collect(notified, "final", timeout=3.0)
    await listener.stop()
    assert final is not None, f"no final notification; got {notified}"
    assert final["text"] == "hello world"
    assert final["seq"] == 1
    assert len(transcribed) == 1
    # the transcribed payload must be a valid WAV
    with wave.open(BytesIO(transcribed[0]), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2


@pytest.mark.asyncio
async def test_short_segment_is_skipped():
    transcribed = []

    async def transcribe(cfg, wav, **kw):
        transcribed.append(wav)
        return {"text": ""}

    notified = []

    async def notifier(payload):
        notified.append(payload)

    # 1 speech window then silence. VAD min_speech_ms=0 so it still fires a
    # close for the short segment; the listener's 0.1s STT gate then skips it
    # and emits a final(skipped) notification without calling transcribe.
    probs = [0.9] + [0.0] * 20
    model = _FakeVadModel(probs)
    listener, captured = _make_listener(transcribe, notifier, vad_model=model, min_silence_ms=200)
    listener.vad.min_speech_samples = 0  # let VAD fire on the 1-window segment
    await listener.start()
    cb = captured["cb"]
    for _ in range(len(probs)):
        _feed_window(cb)
        await asyncio.sleep(0.001)
    final = await _collect(notified, "final", timeout=3.0)
    await listener.stop()
    assert final is not None
    assert final.get("skipped") is True
    assert final["text"] == ""
    assert transcribed == []


@pytest.mark.asyncio
async def test_transcribe_error_emits_error_notification():
    notified = []

    async def transcribe(cfg, wav, **kw):
        raise RuntimeError("mlx down")

    async def notifier(payload):
        notified.append(payload)

    probs = [0.0, 0.0] + [0.9] * 4 + [0.0] * 20
    model = _FakeVadModel(probs)
    listener, captured = _make_listener(transcribe, notifier, vad_model=model, min_silence_ms=200)
    await listener.start()
    cb = captured["cb"]
    for _ in range(len(probs)):
        _feed_window(cb)
        await asyncio.sleep(0.001)
    err = await _collect(notified, "error", timeout=3.0)
    await listener.stop()
    assert err is not None
    assert "mlx down" in err["error"]


def test_pcm_to_wav_roundtrip():
    pcm = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    wav = pcm_to_wav(pcm, 16000)
    with wave.open(BytesIO(wav), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        frames = w.readframes(w.getnframes())
    out = np.frombuffer(frames, dtype=np.int16)
    # check values are monotonic in the right direction (1.0 -> +32767)
    assert out[3] == 32767
    assert out[4] == -32767


@pytest.mark.asyncio
async def test_double_start_is_idempotent():
    async def transcribe(cfg, wav, **kw):
        return {"text": ""}

    async def notifier(payload):
        pass

    model = _FakeVadModel([0.0] * 5)
    listener, _ = _make_listener(transcribe, notifier, vad_model=model)
    await listener.start()
    r2 = await listener.start()
    assert r2.get("already") is True
    await listener.stop()
