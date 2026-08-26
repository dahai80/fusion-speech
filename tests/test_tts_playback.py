from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest

from fusion_speech.config import load_config
from fusion_speech.tts_playback import StreamingPlayer


def _wav_header(sample_rate: int, num_samples: int) -> bytes:
    data = num_samples * 2
    return (
        b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", data)
    )


class _FakeOutStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False
        self.started = False
        self.written: list[np.ndarray] = []

    def start(self):
        self.active = True
        self.started = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False

    def write(self, data: np.ndarray):
        self.written.append(np.asarray(data, dtype=np.float32))


def _make_player(synth, *, buffer_s=2.0, rate=16000, captured=None):
    cfg = load_config()
    cfg.tts_playback_buffer_s = buffer_s

    def factory(**kwargs):
        s = _FakeOutStream(**kwargs)
        if captured is not None:
            captured["stream"] = s
            captured["kwargs"] = kwargs
        return s

    return StreamingPlayer(cfg, synthesize=synth, stream_factory=factory)


@pytest.mark.asyncio
async def test_speak_plays_chunks_and_returns_done():
    pcm = (np.sin(np.linspace(0, 100, 1600)) * 0.3).astype(np.float32)
    i16 = (pcm * 32767).astype(np.int16).tobytes()

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, pcm.size)
        yield i16
        yield i16

    captured = {}
    player = _make_player(synth, captured=captured)
    res = await player.speak("hi")
    assert res["played"] is True
    assert res["chars"] == 2
    assert res["sample_rate"] == 16000
    # output stream started (TTFA on first chunk)
    assert captured["stream"].started is True
    # all PCM written (2 chunks worth)
    total = sum(w.size for w in captured["stream"].written)
    assert total == pcm.size * 2
    assert player.active is False


@pytest.mark.asyncio
async def test_speak_fail_soft_when_output_open_raises():
    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 4)
        yield (np.zeros(4, dtype=np.int16)).tobytes()

    cfg = load_config()

    def bad_factory(**kwargs):
        raise OSError("no device")

    player = StreamingPlayer(cfg, synthesize=synth, stream_factory=bad_factory)
    res = await player.speak("hi")
    assert res["played"] is False
    assert "no device" in res["error"]
    assert player.active is False


@pytest.mark.asyncio
async def test_speak_resamples_when_rate_mismatch():
    src_rate = 24000
    pcm = (np.sin(np.linspace(0, 100, 2400)) * 0.3).astype(np.float32)
    i16 = (pcm * 32767).astype(np.int16).tobytes()

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(src_rate, pcm.size)
        yield i16

    captured = {}
    player = _make_player(synth, rate=16000, captured=captured)
    res = await player.speak("hi")
    assert res["played"] is True
    assert res["sample_rate"] == src_rate
    # daemon rate is 16000; output opened at 16000
    assert captured["kwargs"]["samplerate"] == 16000
    # written samples resampled down: 2400 -> 1600 (approx)
    total = sum(w.size for w in captured["stream"].written)
    assert 1500 <= total <= 1700


@pytest.mark.asyncio
async def test_cancel_stops_playback():
    big = (np.zeros(16000 * 10, dtype=np.float32))

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, big.size)
        # yield many small frames so the writer is mid-playback
        for _ in range(20):
            if player._cancelled:
                break
            yield (big[:1000] * 32767).astype(np.int16).tobytes()
            await asyncio.sleep(0.01)

    captured = {}
    player = _make_player(synth, captured=captured)
    task = asyncio.create_task(player.speak("hi"))
    await asyncio.sleep(0.05)
    res = await player.cancel()
    assert res["stopped"] is True
    speak_res = await asyncio.wait_for(task, timeout=3.0)
    assert speak_res["played"] is False
    assert speak_res["cancelled"] is True
    assert player.active is False


@pytest.mark.asyncio
async def test_bounded_queue_backpressures_fetch():
    # The player builds a bounded asyncio.Queue(maxsize=cap) where
    # cap = tts_playback_buffer_s * sample_rate. Backpressure = the fetch
    # task's `await queue.put(frame)` blocks once the queue is full, so the
    # producer cannot race unbounded ahead of the consumer. We verify the
    # mechanism directly: the queue is bounded, and a put past the cap blocks
    # (its coroutine does not resume) until a consumer drains.
    cfg = load_config()
    cfg.tts_playback_buffer_s = 0.05
    cfg.sample_rate = 16000
    expected_cap = max(1, int(cfg.tts_playback_buffer_s * cfg.sample_rate))

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 1600)
        # never yield PCM: keeps fetch idle so we can drive the queue ourselves

    player = StreamingPlayer(cfg, synthesize=synth, stream_factory=lambda **k: _FakeOutStream())
    # speak starts fetch + write tasks; fetch yields only a header (empty PCM
    # after the 44-byte strip), so nothing is queued. We piggyback on the live
    # queue the player created to assert boundedness + blocking put behavior.
    task = asyncio.create_task(player.speak("hi"))
    # let the tasks start and build the queue
    await asyncio.sleep(0.05)
    queue = player._queue
    assert queue is not None, "player did not build a queue"
    assert queue.maxsize == expected_cap, f"queue unbounded: maxsize={queue.maxsize}"
    # fill the queue to its cap with put_nowait (must succeed up to maxsize)
    for _ in range(expected_cap):
        queue.put_nowait(np.zeros(1, dtype=np.float32))
    assert queue.qsize() == expected_cap
    # one more put MUST block (backpressure): a coroutine attempting it does
    # not complete until a consumer drains a slot.
    blocked = asyncio.Event()
    put_done = asyncio.Event()

    async def _blocked_put():
        blocked.set()
        await queue.put(np.zeros(1, dtype=np.float32))
        put_done.set()

    put_task = asyncio.create_task(_blocked_put())
    await asyncio.wait_for(blocked.wait(), timeout=1.0)
    # give the event loop a chance to run the put; it must stay blocked
    await asyncio.sleep(0.02)
    assert not put_done.is_set(), "put did NOT block on a full bounded queue (no backpressure)"
    # drain one slot -> the blocked put completes
    queue.get_nowait()
    await asyncio.wait_for(put_done.wait(), timeout=1.0)
    put_task.cancel()
    # stop the speak we started (fetch is idle on empty synth, cancel cleanly)
    await player.cancel()
    await asyncio.wait_for(task, timeout=3.0)
    assert player.active is False

