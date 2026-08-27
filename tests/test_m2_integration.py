from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fusion_speech.audio_source import AudioSource
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.server import SpeechDaemon
from fusion_speech.stt_stream import StreamingListener
from fusion_speech.vad import SileroVadEngine


class _FakeStream:
    def __init__(self, **kwargs):
        self.active = False
        self.cb = kwargs.get("callback")

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


class _FakeVadModel:
    # Deterministic prob schedule: silence -> speech -> silence -> 1 segment.
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


@pytest.mark.asyncio
async def test_start_listening_capture_transcribe_pushes_final_to_subscriber():
    # End-to-end M2 path: RPC start -> fake mic capture -> VAD segment ->
    # mock transcribe -> final speech.asr notification delivered to a WS
    # subscriber queue. No real hardware, no real mlx call.
    daemon = SpeechDaemon()
    transcribed = []

    async def transcribe(cfg, wav, **kw):
        transcribed.append(wav)
        return {"text": "integration ok", "language": "en", "duration": 0.5}

    class _Cfg:
        sample_rate = 16000
        channels = 1
        language = "en"
        vad_speech_threshold = 0.5

    ring = RingBuffer(16000 * 10)

    def factory(**kwargs):
        return _FakeStream(**kwargs)

    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)
    vad = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=200,
        speech_pad_ms=0,
        model=_FakeVadModel([0.0, 0.0] + [0.9] * 4 + [0.0] * 20),
        on_open=None,
        on_close=None,
    )
    listener = StreamingListener(
        _Cfg(),
        transcribe=transcribe,
        notifier=daemon.broadcast,
        audio_source=src,
        vad=vad,
    )
    vad._on_open = listener._on_open
    vad._on_close = listener._on_close
    daemon._listener = listener

    # subscriber queue = what a /stream/stt WS client would read
    q = daemon.subscribe()
    await listener.start()
    cb = src._stream.cb if hasattr(src, "_stream") else None
    # drive the fake mic: feed the scheduled windows through the source callback
    for _ in range(26):
        _feed_window(cb)
        await asyncio.sleep(0.001)

    final = await asyncio.wait_for(_wait_for_state(q, "final"), timeout=4.0)
    await listener.stop()
    daemon.unsubscribe(q)

    assert final["state"] == "final"
    assert final["text"] == "integration ok"
    assert final["seq"] == 1
    assert len(transcribed) == 1
    assert daemon.listening is False


async def _wait_for_state(q: asyncio.Queue, state: str) -> dict:
    while True:
        payload = await q.get()
        if payload.get("state") == state:
            return payload
