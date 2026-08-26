from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fusion_speech.audio_source import AudioSource
from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher
from fusion_speech.ipc.methods import build_methods
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.server import SpeechDaemon
from fusion_speech.wake_word import WakeWordEngine


class _FakeOwwModel:
    def __init__(self, config, words):
        self.words = list(words)
        self._i = 0

    def predict(self, chunk: np.ndarray) -> dict:
        import os

        from fusion_speech.wake_word import PRETRAINED_WAKE_WORDS

        key = os.path.splitext(PRETRAINED_WAKE_WORDS[self.words[0]])[0]
        return {key: 0.0}

    def reset(self):
        pass


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


def _patch_daemon_for_wake(daemon, *, auto_listen_s=0.2):
    # Replace the real engine build + real ASR so the handshake runs without
    # hardware or mlx. arm_wake() builds a fake WakeWordEngine; _on_wake runs the
    # real auto-listen handshake, but start_listening uses a fake source.
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

    daemon._wake_factory = build_engine

    async def fake_arm():
        wake = build_engine()
        daemon._wake = wake
        if not wake.enabled:
            return {"armed": False, "enabled": False}
        res = await wake.start()
        daemon.wake_armed = bool(res.get("armed"))
        if daemon.wake_armed:
            daemon.hotwords = list(wake.words)
        return res

    daemon.arm_wake = fake_arm

    async def fake_start_listening(audio_source=None):
        # pretend ASR arms instantly; track the handshake did call us.
        daemon._handshake_started = True
        daemon.listening = True
        daemon._listener = _FakeListener()
        return {"listening": True}

    async def fake_stop_listening():
        daemon.listening = False
        daemon._handshake_stopped = True
        daemon._listener = None
        return {"listening": False}

    daemon.start_listening = fake_start_listening
    daemon.stop_listening = fake_stop_listening
    daemon._handshake_started = False
    daemon._handshake_stopped = False


class _FakeListener:
    active = True
    source = None


@pytest.mark.asyncio
async def test_wake_rpc_arm_disarm():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon)
    dispatcher = JsonRpcDispatcher(build_methods(daemon))

    arm = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.wake", "params": {"arm": True}})
    assert arm["result"]["armed"] is True
    assert daemon.wake_armed is True
    assert daemon._wake is not None and daemon._wake.active

    disarm = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 2, "method": "speech.wake", "params": {"arm": False}})
    assert disarm["result"]["armed"] is False
    assert daemon.wake_armed is False
    assert daemon._wake is None


@pytest.mark.asyncio
async def test_set_hotwords_rpc():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon)
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.wake", "params": {"arm": True}})

    resp = await dispatcher.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "speech.set_hotwords", "params": {"hotwords": ["alexa", "hey_mycroft"]}}
    )
    assert resp["result"]["hotwords"] == ["alexa", "hey_mycroft"]
    assert daemon.hotwords == ["alexa", "hey_mycroft"]


@pytest.mark.asyncio
async def test_set_hotwords_drops_unknown_via_rpc():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon)
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.wake", "params": {"arm": True}})

    resp = await dispatcher.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "speech.set_hotwords", "params": {"hotwords": ["hey_jarvis", "bogus"]}}
    )
    assert resp["result"]["hotwords"] == ["hey_jarvis"]


@pytest.mark.asyncio
async def test_status_reports_wake_armed():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon)
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    s0 = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.status"})
    assert s0["result"]["wake_armed"] is False

    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 2, "method": "speech.wake", "params": {"arm": True}})
    s1 = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 3, "method": "speech.status"})
    assert s1["result"]["wake_armed"] is True


@pytest.mark.asyncio
async def test_on_wake_handshake_auto_listens_then_resumes():
    # The chosen coexistence model: wake fires -> broadcast speech.wake -> pause
    # wake -> start ASR (auto-listen window) -> stop ASR -> resume wake.
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon, auto_listen_s=0.1)
    q = daemon.subscribe()
    await daemon.arm_wake()
    wake = daemon._wake

    # drive a wake fire directly through the engine's callback path
    await daemon._on_wake("hey_jarvis")

    # the speech.wake notification must land on the subscriber queue
    wake_evt = await asyncio.wait_for(_wait_event(q, "speech.wake"), timeout=2.0)
    assert wake_evt["word"] == "hey_jarvis"

    # handshake ran start+stop listening, then resumed wake
    await asyncio.sleep(0.2)
    assert daemon._handshake_started is True
    assert daemon._handshake_stopped is True
    assert wake.paused is False
    assert daemon.listening is False
    daemon.unsubscribe(q)
    await daemon.disarm_wake()


@pytest.mark.asyncio
async def test_on_wake_skips_handshake_if_asr_already_active():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon)
    await daemon.arm_wake()
    # mark ASR active beforehand
    daemon._listener = _FakeListener()
    daemon._handshake_started = False
    q = daemon.subscribe()
    await daemon._on_wake("hey_jarvis")
    evt = await asyncio.wait_for(_wait_event(q, "speech.wake"), timeout=2.0)
    assert evt["word"] == "hey_jarvis"
    await asyncio.sleep(0.05)
    # handshake start NOT called (already listening)
    assert daemon._handshake_started is False
    daemon.unsubscribe(q)
    await daemon.disarm_wake()


@pytest.mark.asyncio
async def test_stop_daemon_stops_wake_engine():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_for_wake(daemon)
    await daemon.arm_wake()
    assert daemon._wake is not None
    await daemon.stop()
    assert daemon._wake is None
    assert daemon.wake_armed is False


async def _wait_event(q: asyncio.Queue, event: str) -> dict:
    while True:
        payload = await q.get()
        if payload.get("event") == event:
            return payload
