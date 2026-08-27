from __future__ import annotations

import numpy as np

from fusion_speech.audio_source import AudioSource, AudioSourceError
from fusion_speech.ring_buffer import RingBuffer


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True
        self.active = False


def _fake_factory():
    captured: list[FakeStream] = []

    def factory(**kwargs):
        s = FakeStream(**kwargs)
        captured.append(s)
        return s

    return factory, captured


def test_start_opens_and_starts_stream():
    ring = RingBuffer(1024)
    factory, captured = _fake_factory()
    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)
    src.start()
    assert src.is_active is True
    assert len(captured) == 1
    assert captured[0].started is True


def test_callback_writes_mono_into_ring():
    ring = RingBuffer(1024)
    factory, captured = _fake_factory()
    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)
    src.start()
    cb = captured[0].kwargs["callback"]
    cb(np.arange(8, dtype=np.float32).reshape(-1, 1), 8, None, 0)
    out = ring.drain()
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_callback_downmixes_multichannel():
    ring = RingBuffer(1024)
    factory, captured = _fake_factory()
    src = AudioSource(ring, sample_rate=16000, channels=2, stream_factory=factory)
    src.start()
    cb = captured[0].kwargs["callback"]
    stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    cb(stereo, 3, None, 0)
    out = ring.drain()
    assert out.tolist() == [1.0, 3.0, 5.0]


def test_stop_closes_stream_and_ring():
    ring = RingBuffer(1024)
    factory, captured = _fake_factory()
    src = AudioSource(ring, stream_factory=factory)
    src.start()
    src.stop()
    assert src.is_active is False
    assert captured[0].closed is True
    assert ring.stats()["closed"] is True


def test_double_start_is_noop():
    ring = RingBuffer(1024)
    factory, captured = _fake_factory()
    src = AudioSource(ring, stream_factory=factory)
    src.start()
    src.start()
    assert len(captured) == 1


def test_open_failure_raises_audio_source_error():
    ring = RingBuffer(1024)

    def boom(**kwargs):
        raise OSError("no device")

    src = AudioSource(ring, stream_factory=boom)
    raised = False
    try:
        src.start()
    except AudioSourceError as e:
        raised = True
        assert "no device" in str(e)
    assert raised
    assert src.is_active is False


def test_status_flag_increments_xruns():
    ring = RingBuffer(1024)
    factory, captured = _fake_factory()
    src = AudioSource(ring, stream_factory=factory)
    src.start()
    cb = captured[0].kwargs["callback"]
    cb(np.zeros(4, dtype=np.float32).reshape(-1, 1), 4, None, "input overflow")
    assert src.xruns >= 1
