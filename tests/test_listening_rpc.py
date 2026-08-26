from __future__ import annotations

import asyncio

import pytest

from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher
from fusion_speech.ipc.methods import build_methods
from fusion_speech.server import SpeechDaemon


@pytest.mark.asyncio
async def test_start_stop_listening_rpc_with_fake_mic():
    # Avoid touching real audio hardware: patch daemon.start_listening to use
    # a StreamingListener whose AudioSource is a fake stream factory.
    from fusion_speech.audio_source import AudioSource
    from fusion_speech.ring_buffer import RingBuffer
    from fusion_speech.stt_stream import StreamingListener
    from fusion_speech.vad import SileroVadEngine

    daemon = SpeechDaemon(load_config())

    class _FakeStream:
        def __init__(self, **kw):
            self.active = False

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

    async def transcribe(cfg, wav, **kw):
        return {"text": "x", "language": "en"}

    async def fake_start():
        ring = RingBuffer(16000 * 10)

        def fact(**kw):
            return _FakeStream(**kw)

        src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=fact)
        vad = SileroVadEngine(
            sample_rate=16000,
            threshold=0.5,
            min_speech_ms=100,
            min_silence_ms=100,
            speech_pad_ms=0,
            model=None,
            model_path="/nonexistent/onnx",
            on_open=None,
            on_close=None,
        )
        listener = StreamingListener(
            daemon.config,
            transcribe=transcribe,
            notifier=daemon.broadcast,
            audio_source=src,
            vad=vad,
        )
        vad._on_open = listener._on_open
        vad._on_close = listener._on_close
        daemon._listener = listener
        res = await listener.start()
        daemon.listening = True
        return res

    daemon.start_listening = fake_start
    dispatcher = JsonRpcDispatcher(build_methods(daemon))

    start_resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.start_listening"})
    assert start_resp["result"]["listening"] is True
    assert daemon.listening is True

    stop_resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 2, "method": "speech.stop_listening"})
    assert stop_resp["result"]["listening"] is False
    assert daemon.listening is False
    assert daemon._listener is None


@pytest.mark.asyncio
async def test_status_reports_listening_state():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.status"})
    assert resp["result"]["listening"] is False
    assert "hotwords" in resp["result"]


@pytest.mark.asyncio
async def test_broadcast_delivers_to_subscribers():
    daemon = SpeechDaemon(load_config())
    q1 = daemon.subscribe()
    q2 = daemon.subscribe()
    await daemon.broadcast({"event": "speech.asr", "state": "final", "text": "hi"})
    assert q1.get_nowait()["text"] == "hi"
    assert q2.get_nowait()["text"] == "hi"
    daemon.unsubscribe(q1)
    daemon.unsubscribe(q2)
    assert len(daemon._subscribers) == 0


@pytest.mark.asyncio
async def test_broadcast_drops_oldest_on_full_queue():
    daemon = SpeechDaemon(load_config())
    daemon.subscribe()  # normal subscriber, should be untouched
    # force a small queue to exercise the drop-oldest path
    small: asyncio.Queue = asyncio.Queue(maxsize=2)
    daemon._subscribers.add(small)
    await daemon.broadcast({"seq": 1})
    await daemon.broadcast({"seq": 2})
    await daemon.broadcast({"seq": 3})  # should drop seq 1
    got = []
    while not small.empty():
        got.append(small.get_nowait()["seq"])
    assert got == [2, 3]
    daemon._subscribers.clear()


def test_stream_stt_websocket_accepts_and_closes():
    # WS handler is thin glue (accept -> read subscriber queue -> send_json).
    # The broadcast->subscriber fan-out is covered by the unit tests above; here
    # we only confirm the endpoint accepts a connection and tears down cleanly.
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    daemon = SpeechDaemon(load_config())
    app = daemon.build_app()
    client = TestClient(app)
    # The handler subscribes on accept; when the client context closes, the
    # server's finally block unsubscribes. Starlette's TestClient re-raises the
    # 1000 close code on context exit — catch it so the teardown assertion runs.
    try:
        with client.websocket_connect("/stream/stt"):
            pass
    except WebSocketDisconnect:
        pass
    assert len(daemon._subscribers) == 0


@pytest.mark.asyncio
async def test_stop_listening_when_none_is_noop():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.stop_listening"})
    assert resp["result"]["listening"] is False
