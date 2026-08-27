from __future__ import annotations

import os

import numpy as np
import pytest

from fusion_speech.vad import (
    DEFAULT_SILERO_PATH,
    SileroVadEngine,
    SileroVadModel,
    VadSegment,
)

SILERO_AVAILABLE = os.path.exists(DEFAULT_SILERO_PATH)


class FakeModel:
    # deterministic speech-prob schedule for unit tests (no onnxruntime needed)
    def __init__(self, probs: list[float]):
        self.probs = list(probs)
        self.calls = 0
        self.resets = 0

    def reset_state(self):
        self.resets += 1

    def __call__(self, window):
        p = self.probs[min(self.calls, len(self.probs) - 1)]
        self.calls += 1
        return p


def _tone(n, freq=440, sr=16000, amp=0.3):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(n):
    return np.zeros(n, dtype=np.float32)


def test_segment_open_close_events_with_fake_model():
    opens = []
    closes = []

    # window 512@16k. 2 silence, 4 speech (2048 >= min_speech 1600),
    # 8 silence (4096 >= min_silence 3200) -> one segment closes.
    probs = [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    model = FakeModel(probs)
    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=200,
        speech_pad_ms=0,
        on_open=lambda s: opens.append(s),
        on_close=lambda seg: closes.append(seg),
        model=model,
    )
    for _ in range(len(probs)):
        eng.feed(_tone(512))
    assert opens == [2 * 512]
    assert len(closes) == 1
    seg = closes[0]
    assert isinstance(seg, VadSegment)
    assert seg.start_sample == 2 * 512
    assert seg.end_sample >= 6 * 512


def test_short_speech_below_min_is_dropped():
    closes = []
    # 1 speech window then long silence -> segment too short (< min_speech_ms)
    probs = [0.9, 0.0, 0.0, 0.0, 0.0]
    model = FakeModel(probs)
    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=1000,  # 16000 samples = 31 windows; 1 window < min
        min_silence_ms=100,
        speech_pad_ms=0,
        on_close=lambda seg: closes.append(seg),
        model=model,
    )
    for _ in range(len(probs)):
        eng.feed(_tone(512))
    assert closes == []


def test_energy_only_fallback_when_model_none():
    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=100,
        speech_pad_ms=0,
        model=None,
        model_path="/nonexistent/silero.onnx",
        on_close=lambda seg: None,
    )
    assert eng.energy_only is True


def test_energy_gate_skips_silence_windows():
    calls = []

    class CountingModel:
        def reset_state(self):
            pass

        def __call__(self, window):
            calls.append(1)
            return 0.0

    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=100,
        model=CountingModel(),
        on_open=lambda s: None,
        on_close=lambda seg: None,
    )
    eng.feed(_silence(512))  # rms ~0 < gate -> model NOT called
    assert calls == []


def test_reset_clears_state():
    opens = []
    model = FakeModel([0.9])
    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=100,
        speech_pad_ms=0,
        on_open=lambda s: opens.append(s),
        on_close=lambda seg: None,
        model=model,
    )
    eng.feed(_tone(512))
    assert eng._in_speech is True
    eng.reset()
    assert eng._in_speech is False
    assert eng._len == 0
    assert model.resets >= 2


def test_flush_closes_open_segment():
    closes = []
    model = FakeModel([0.9])
    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=10000,  # never auto-close
        speech_pad_ms=0,
        on_close=lambda seg: closes.append(seg),
        model=model,
    )
    eng.feed(_tone(512))
    assert eng._in_speech is True
    eng.flush()
    assert len(closes) == 1


def test_subwindow_buffering_accumulates():
    model = FakeModel([0.0, 0.0])
    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=100,
        model=model,
        on_open=lambda s: None,
        on_close=lambda seg: None,
    )
    eng.feed(_tone(256))  # half a window -> no processing yet
    assert model.calls == 0
    eng.feed(_tone(256))  # now a full window
    assert model.calls == 1


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "assistant_bank_en.wav")


def _load_speech_fixture() -> np.ndarray:
    import wave

    import scipy.signal as ss

    with wave.open(_FIXTURE) as w:
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x[:: w.getnchannels()]
        sr_in = w.getframerate()
    if sr_in != 16000:
        x = ss.resample_poly(x, 16000, sr_in).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


_FIXTURE_AVAILABLE = os.path.exists(_FIXTURE)


@pytest.mark.skipif(not (SILERO_AVAILABLE and _FIXTURE_AVAILABLE), reason="silero onnx or speech fixture missing")
def test_real_silero_model_loads_and_classifies():
    # real-load checkpoint: load actual onnx, feed real speech vs silence.
    speech = _load_speech_fixture()
    model = SileroVadModel(DEFAULT_SILERO_PATH)
    speech_prob = model(speech[10 * 512 : 11 * 512])  # mid-utterance
    model.reset_state()
    silence_prob = model(_silence(512))
    assert 0.0 <= speech_prob <= 1.0
    assert 0.0 <= silence_prob <= 1.0
    assert speech_prob > silence_prob


@pytest.mark.skipif(not (SILERO_AVAILABLE and _FIXTURE_AVAILABLE), reason="silero onnx or speech fixture missing")
def test_real_silero_engine_segment_on_speech():
    opens = []
    closes = []
    eng = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=300,
        speech_pad_ms=0,
        on_open=lambda s: opens.append(s),
        on_close=lambda seg: closes.append(seg),
        model_path=DEFAULT_SILERO_PATH,
    )
    assert eng.energy_only is False
    # real 4s speech utterance then 0.5s silence -> expect >=1 segment
    speech = _load_speech_fixture()
    eng.feed(speech)
    eng.feed(_silence(8000))
    eng.flush()
    assert len(opens) >= 1
    assert len(closes) >= 1
