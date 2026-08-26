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
