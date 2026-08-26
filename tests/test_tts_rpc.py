from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest

from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher
from fusion_speech.ipc.methods import build_methods
from fusion_speech.server import SpeechDaemon


def _wav_header(sample_rate: int, num_samples: int) -> bytes:
    data = num_samples * 2
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data)
    )


class _FakeOutStream:
    def __init__(self, **kwargs):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False

    def write(self, data):
        pass


def _patch_daemon_tts(daemon):
    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 4)
        yield (np.zeros(4, dtype=np.int16)).tobytes()

    def factory(**kwargs):
        return _FakeOutStream(**kwargs)

    real_speak = daemon.speak

    async def patched_speak(text, voice="", model="", synthesize=None, stream_factory=None):
        return await real_speak(
            text,
            voice=voice,
            model=model,
            synthesize=synthesize or synth,
            stream_factory=stream_factory or factory,
        )

    daemon.speak = patched_speak


@pytest.mark.asyncio
async def test_speak_rpc_round_trip():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_tts(daemon)
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    res = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hello"}})
    assert res["result"]["played"] is True


@pytest.mark.asyncio
async def test_status_reports_speaking():
    daemon = SpeechDaemon(load_config())
    release = asyncio.Event()
    seen_mid_flight = []

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 4)
        yield (np.zeros(4, dtype=np.int16)).tobytes()
        await release.wait()

    def factory(**kwargs):
        return _FakeOutStream(**kwargs)

    real_speak = daemon.speak

    async def patched_speak(text, voice="", model="", synthesize=None, stream_factory=None):
        return await real_speak(
            text,
            voice=voice,
            model=model,
            synthesize=synthesize or synth,
            stream_factory=stream_factory or factory,
        )

    daemon.speak = patched_speak
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    task = asyncio.create_task(
        dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hi"}})
    )
    await asyncio.sleep(0.05)
    s_mid = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 2, "method": "speech.status"})
    seen_mid_flight.append(s_mid["result"]["speaking"])
    release.set()
    await asyncio.wait_for(task, timeout=3.0)
    s_done = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 3, "method": "speech.status"})
    assert seen_mid_flight == [True], "speaking was not True mid-flight"
    assert s_done["result"]["speaking"] is False


class _FakeWake:
    def __init__(self):
        self.active = True
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


@pytest.mark.asyncio
async def test_speak_pauses_wake_during_playback():
    daemon = SpeechDaemon(load_config())
    paused_during = []

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        paused_during.append(daemon._wake.paused)
        yield _wav_header(16000, 4)
        yield (np.zeros(4, dtype=np.int16)).tobytes()

    def factory(**kwargs):
        return _FakeOutStream(**kwargs)

    real_speak = daemon.speak

    async def patched_speak(text, voice="", model="", synthesize=None, stream_factory=None):
        return await real_speak(
            text,
            voice=voice,
            model=model,
            synthesize=synthesize or synth,
            stream_factory=stream_factory or factory,
        )

    daemon.speak = patched_speak

    wake = _FakeWake()
    daemon._wake = wake
    daemon.wake_armed = True
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hi"}})
    assert paused_during and paused_during[0] is True
    assert wake.paused is False  # resumed after


@pytest.mark.asyncio
async def test_speak_stops_and_restarts_asr():
    daemon = SpeechDaemon(load_config())

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 4)
        yield (np.zeros(4, dtype=np.int16)).tobytes()

    def factory(**kwargs):
        return _FakeOutStream(**kwargs)

    real_speak = daemon.speak

    async def patched_speak(text, voice="", model="", synthesize=None, stream_factory=None):
        return await real_speak(
            text,
            voice=voice,
            model=model,
            synthesize=synthesize or synth,
            stream_factory=stream_factory or factory,
        )

    daemon.speak = patched_speak

    class _FakeListener:
        active = True

    daemon._listener = _FakeListener()
    listen_calls = []

    async def fake_start(audio_source=None):
        listen_calls.append("start")
        daemon._listener = _FakeListener()
        return {"listening": True}

    async def fake_stop():
        listen_calls.append("stop")
        daemon._listener = None
        return {"listening": False}

    daemon.start_listening = fake_start
    daemon.stop_listening = fake_stop
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hi"}})
    assert listen_calls == ["stop", "start"]


async def _wait_event(q: asyncio.Queue, event: str, state: str, timeout=2.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            payload = await asyncio.wait_for(q.get(), timeout=deadline - asyncio.get_event_loop().time())
        except TimeoutError:
            break
        if payload.get("event") == event and payload.get("state") == state:
            return payload
    return None


@pytest.mark.asyncio
async def test_speak_emits_tts_events():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_tts(daemon)
    q = daemon.subscribe()
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hi"}})
    done = await _wait_event(q, "speech.tts", "done")
    assert done is not None
    assert done["state"] == "done"
    daemon.unsubscribe(q)
