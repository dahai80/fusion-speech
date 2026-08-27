from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile

import numpy as np
import pytest

from fusion_speech.audio_source import AudioSource
from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher, handle_client
from fusion_speech.ipc.methods import MAX_AUDIO_B64, MAX_TEXT, build_methods
from fusion_speech.ring_buffer import RingBuffer
from fusion_speech.server import SpeechDaemon


@pytest.fixture
def short_sock_dir():
    # macOS caps AF_UNIX path length (~104 chars); pytest's tmp_path is far
    # deeper. Bind under a short /tmp dir instead.
    d = tempfile.mkdtemp(prefix="fs_", dir="/tmp")
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


class _FakeStream:
    def __init__(self, **kwargs):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


# F2: a peer-uid check that fails (or has no socket) must fail CLOSED — the
# connection is rejected and nothing is dispatched. verify_fn injection lets us
# exercise the fail-closed path without a real UDS peer-credential handshake.
@pytest.mark.asyncio
async def test_f2_fail_closed_rejects_unverified_peer(short_sock_dir):
    sock_path = str(short_sock_dir + "/fail.sock")
    dispatched: list[dict] = []
    dispatcher = JsonRpcDispatcher({"speech.ping": lambda p: _record_ping(dispatched, p)})

    async def _cb(reader, writer):
        await handle_client(
            reader,
            writer,
            dispatcher,
            lambda: True,
            verify_fn=lambda w: False,  # always reject
        )

    server = await asyncio.start_unix_server(_cb, path=sock_path)
    try:
        reader, writer = await asyncio.open_unix_connection(sock_path)
        writer.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "speech.ping"}).encode() + b"\n")
        await writer.drain()
        # server should close the connection without responding
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        except TimeoutError:
            data = b""
        assert data == b"", f"unverified peer got a response: {data!r}"
        assert dispatched == [], "handler ran despite failed peer check"
    finally:
        server.close()
        await server.wait_closed()


# F2: when the socket object is missing (get_extra_info returns None) the real
# _verify_peer_uid must also reject, not the old fail-open return True.
@pytest.mark.asyncio
async def test_f2_verify_peer_uid_rejects_when_no_socket():
    from fusion_speech.ipc.jsonrpc import _verify_peer_uid

    class _NoSockWriter:
        def get_extra_info(self, name):
            return None

    assert _verify_peer_uid(_NoSockWriter()) is False


# F3: a client streaming > 1 MiB with no newline must not grow the buffer
# unbounded — the connection is closed and the daemon stays alive.
@pytest.mark.asyncio
async def test_f3_huge_unterminated_line_closes_connection(short_sock_dir):
    sock_path = str(short_sock_dir + "/dos.sock")
    dispatched: list[dict] = []
    dispatcher = JsonRpcDispatcher({"speech.ping": lambda p: _record_ping(dispatched, p)})

    async def _cb(reader, writer):
        await handle_client(reader, writer, dispatcher, lambda: True, verify_fn=lambda w: True)

    server = await asyncio.start_unix_server(_cb, path=sock_path)
    try:
        reader, writer = await asyncio.open_unix_connection(sock_path)
        # > _MAX_LINE (1 MiB) with no newline anywhere
        writer.write(b"x" * ((1 << 20) + 4096))
        await writer.drain()
        # server should close the connection; read returns EOF (b"") quickly
        data = await asyncio.wait_for(reader.read(4096), timeout=3.0)
        assert data == b"", f"client not closed after line-limit breach: {data!r}"
        assert dispatched == [], "handler ran on the oversized line"
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


# F3: a well-formed message under the limit still dispatches normally (guard
# does not break the happy path).
@pytest.mark.asyncio
async def test_f3_normal_message_under_limit_still_dispatches(short_sock_dir):
    sock_path = str(short_sock_dir + "/ok.sock")
    dispatched: list[dict] = []
    dispatcher = JsonRpcDispatcher({"speech.ping": lambda p: _record_ping(dispatched, p)})

    async def _cb(reader, writer):
        await handle_client(reader, writer, dispatcher, lambda: True, verify_fn=lambda w: True)

    server = await asyncio.start_unix_server(_cb, path=sock_path)
    try:
        reader, writer = await asyncio.open_unix_connection(sock_path)
        writer.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "speech.ping"}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        resp = json.loads(line)
        assert resp["result"]["pong"] is True
        assert dispatched, "handler did not run for a normal message"
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


# M2: oversized audio base64 -> -32000 error, not an unbounded decode.
@pytest.mark.asyncio
async def test_m2_transcribe_rejects_oversized_audio():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "speech.transcribe",
            "params": {"audio": "x" * (MAX_AUDIO_B64 + 1)},
        }
    )
    assert resp["error"]["code"] == -32000
    assert "too large" in resp["error"]["message"]


# M2: oversized speak text -> -32000 error.
@pytest.mark.asyncio
async def test_m2_speak_rejects_oversized_text():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "speech.speak",
            "params": {"text": "x" * (MAX_TEXT + 1)},
        }
    )
    assert resp["error"]["code"] == -32000
    assert "too long" in resp["error"]["message"]


# L3: the ASR subscriber queue is bounded (maxsize=64); overflow drops oldest.
@pytest.mark.asyncio
async def test_l3_subscriber_queue_is_bounded_and_drops_oldest():
    daemon = SpeechDaemon(load_config())
    q = daemon.subscribe()
    assert q.maxsize == 64
    for i in range(100):
        await daemon.broadcast({"event": "speech.asr", "state": "listening", "seq": i})
    assert q.qsize() <= 64, f"queue grew past cap: {q.qsize()}"
    # the oldest frames were dropped; the head is NOT seq 0
    head = await q.get()
    assert head["seq"] > 0, "drop-oldest did not evict the head"
    daemon.unsubscribe(q)


# A1: in-flight segment transcribe tasks are cancelled and cleared on stop() —
# no orphan tasks outlive the session. Builds the listener with a stub VAD
# model so on_close fires immediately and schedules a long-running transcribe
# task that is still in-flight when stop() runs.
@pytest.mark.asyncio
async def test_a1_stop_clears_inflight_transcribe_tasks():
    from fusion_speech.stt_stream import StreamingListener
    from fusion_speech.vad import SileroVadEngine

    class _StubVadModel:
        # speech on every window so _close_segment fires after min_silence
        def __call__(self, window):
            return 0.9

        def reset_state(self):
            pass

    class _Cfg:
        sample_rate = 16000
        channels = 1
        language = "en"
        vad_speech_threshold = 0.5

    started = asyncio.Event()

    async def transcribe(cfg, wav, **kw):
        started.set()
        await asyncio.sleep(5.0)
        return {"text": "x"}

    notified: list[dict] = []

    async def notifier(payload):
        notified.append(payload)

    ring = RingBuffer(16000 * 10)
    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=lambda **kw: _FakeStream())
    vad = SileroVadEngine(
        sample_rate=16000,
        threshold=0.5,
        min_speech_ms=100,
        min_silence_ms=100,
        speech_pad_ms=0,
        model=_StubVadModel(),
    )
    listener = StreamingListener(_Cfg(), transcribe=transcribe, notifier=notifier, audio_source=src, vad=vad)
    vad._on_open = listener._on_open
    vad._on_close = listener._on_close
    await listener.start()
    # feed speech then silence so a segment closes and schedules a transcribe
    cb = src._on_audio
    sig = np.ones(512, dtype=np.float32).reshape(-1, 1)
    sil = np.zeros(512, dtype=np.float32).reshape(-1, 1)
    for _ in range(5):
        cb(sig, 512, None, 0)
        await asyncio.sleep(0.001)
    for _ in range(25):
        cb(sil, 512, None, 0)
        await asyncio.sleep(0.001)
    await asyncio.wait_for(started.wait(), timeout=3.0)
    assert listener._transcribe_tasks, "no transcribe task was scheduled"
    await listener.stop()
    assert not listener._transcribe_tasks, "orphan transcribe tasks survived stop()"


# A2: the realtime capture callback must not allocate per frame on the mono
# path — the array pushed into the ring shares memory with the input (a view),
# not a contiguous copy.
@pytest.mark.asyncio
async def test_a2_mono_callback_writes_view_not_copy():
    ring = RingBuffer(16000 * 5)
    written: list[np.ndarray] = []
    orig_write = ring.write

    def spy(samples):
        written.append(samples)
        return orig_write(samples)

    ring.write = spy  # type: ignore[assignment]

    def factory(**kwargs):
        return _FakeStream(**kwargs)

    src = AudioSource(ring, sample_rate=16000, channels=1, stream_factory=factory)
    src.start()
    cb = src._on_audio
    indata = np.zeros(512, dtype=np.float32).reshape(-1, 1)
    cb(indata, 512, None, 0)
    assert written, "callback did not push into the ring"
    pushed = written[0]
    assert np.shares_memory(pushed, indata), "mono path copied instead of writing a view"
    src.stop()


async def _record_ping(dispatched, params):
    dispatched.append(params)
    return {"pong": True}
