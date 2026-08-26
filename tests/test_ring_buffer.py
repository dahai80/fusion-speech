from __future__ import annotations

import numpy as np

from fusion_speech.ring_buffer import RingBuffer


def _data(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float32)


def test_write_then_read_roundtrip():
    rb = RingBuffer(8)
    assert rb.write(_data(5)) == 5
    out = rb.read(5)
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert rb.available() == 0


def test_write_wraps_around():
    rb = RingBuffer(4)
    rb.write(_data(3))
    rb.read(3)
    rb.write(_data(3))
    out = rb.drain()
    assert out.tolist() == [0.0, 1.0, 2.0]


def test_overflow_drops_oldest():
    rb = RingBuffer(4)
    rb.write(_data(4))
    dropped = rb.write(_data(2))
    assert dropped == 2
    assert rb.overruns == 1
    assert rb.total_dropped == 2
    out = rb.drain()
    assert out.tolist() == [2.0, 3.0, 0.0, 1.0]


def test_partial_read_keeps_remaining():
    rb = RingBuffer(8)
    rb.write(_data(6))
    out = rb.read(3)
    assert out.tolist() == [0.0, 1.0, 2.0]
    assert rb.available() == 3
    rest = rb.drain()
    assert rest.tolist() == [3.0, 4.0, 5.0]


def test_notifier_called_on_write():
    hits = []
    rb = RingBuffer(4)
    rb.set_notifier(lambda: hits.append(1))
    rb.write(_data(1))
    assert hits == [1]


def test_close_stops_writes():
    rb = RingBuffer(4)
    rb.close()
    assert rb.write(_data(2)) == 0
    assert rb.stats()["closed"] is True


def test_empty_read_returns_empty():
    rb = RingBuffer(4)
    assert rb.read(4).size == 0
    assert rb.drain().size == 0


def test_reset_clears_state():
    rb = RingBuffer(4)
    rb.write(_data(4))
    rb.write(_data(1))
    rb.reset()
    assert rb.available() == 0
    assert rb.overruns == 0
    assert rb.total_written == 0
