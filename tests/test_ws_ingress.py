from __future__ import annotations

import asyncio
import time
import wave
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import fusion_speech.components.asr_service as asr_service_module
import fusion_speech.stt_stream as stt_stream_module
from fusion_speech.engines import TTSEngine, register_tts_engine, tts_engine_adapter
from fusion_speech.server import SpeechDaemon
from fusion_speech.tts_playback import StreamingPlayer
from fusion_speech.vad import SileroVadEngine
from fusion_speech.ws_ingress import StreamAudioSource, WsTtsSink, decode_pcm

# Deterministic energy-only VAD: the listener builds its own SileroVadEngine,
# so patch the name stt_stream resolved at import time. Forces the
# energy-only fallback (nonexistent model path) and fast segment params.
_FAST_VAD_KW = {"min_speech_ms": 100, "min_silence_ms": 150, "speech_pad_ms": 0}

# TestClient's ws.receive() blocks forever; a missed event must fail the test,
# not hang the suite.
pytestmark = pytest.mark.timeout(60)


@pytest.fixture()
def ws_daemon(monkeypatch):
    async def fake_transcribe(config, wav, **kw):
        with wave.open(BytesIO(wav)) as w:
            frames = w.getnframes()
        return {"text": f"ws heard {frames} frames", "language": "en", "duration": frames / 16000}

    monkeypatch.setattr(asr_service_module, "transcribe_stream", fake_transcribe)

    real_vad = SileroVadEngine

    def fast_vad(**kwargs):
        kwargs.update(_FAST_VAD_KW)
        kwargs["model_path"] = "/nonexistent/silero_vad.onnx"
        return real_vad(**kwargs)

    monkeypatch.setattr(stt_stream_module, "SileroVadEngine", fast_vad)
    return SpeechDaemon()


class _SlowTTSEngine(TTSEngine):
    # Cloud-engine stand-in: yields a valid WAV stream slowly so playback is
    # observable and cancellable mid-stream.
    name = "slow_stub"

    def __init__(self, config, **kwargs):
        self.config = config

    def synthesize(self, text, *, voice="", model="", ref_audio="", ref_text=""):
        return self._gen()

    async def _gen(self):
        buf = BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())  # 1s
        payload = buf.getvalue()
        for i in range(0, len(payload), 1600):
            await asyncio.sleep(0.02)
            yield payload[i : i + 1600]

    async def cancel(self):
        pass


register_tts_engine("slow_stub", _SlowTTSEngine)


def _sine_pcm(seconds: float, amp: float = 0.4) -> bytes:
    t = np.arange(int(16000 * seconds)) / 16000
    return (amp * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()


def _silence_pcm(seconds: float) -> bytes:
    return np.zeros(int(16000 * seconds), dtype="<i2").tobytes()


def _recv_until(ws, predicate, timeout: float = 8.0, collect: list | None = None):
    # receive JSON frames until one matches; binary frames counted, not parsed
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = ws.receive()
        if "text" in frame and frame["text"] is not None:
            import json

            payload = json.loads(frame["text"])
            if collect is not None:
                collect.append(payload)
            if predicate(payload):
                return payload
        # binary frames: ignore here (audio out)
    raise TimeoutError("expected event not received")


# --- unit: StreamAudioSource / decode_pcm / WsTtsSink ---


def test_decode_pcm_int16_and_float32():
    i16 = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    dec = decode_pcm(i16, "int16")
    assert np.allclose(dec, [0.0, 0.5, -0.5], atol=1e-4)
    f32 = np.array([0.0, 0.5, -0.5], dtype="<f4").tobytes()
    assert np.allclose(decode_pcm(f32, "float32"), [0.0, 0.5, -0.5])
    with pytest.raises(ValueError):
        decode_pcm(b"\x01", "int16")
    with pytest.raises(ValueError):
        decode_pcm(b"\x01\x02\x03", "float32")


def test_stream_audio_source_ring_feed_and_stop():
    src = StreamAudioSource(sample_rate=16000, capacity_s=1)
    src.start()
    assert src.active
    assert src.feed(np.ones(1000, dtype=np.float32)) == 1000
    assert src.ring.available() == 1000
    src.stop()
    assert not src.active
    assert src.feed(np.ones(10, dtype=np.float32)) == 0  # closed: drop


class _FakeWs:
    def __init__(self):
        self.sent: list[bytes] = []

    async def send_bytes(self, data):
        self.sent.append(data)


async def test_ws_tts_sink_sends_int16_and_flush_drops():
    ws = _FakeWs()
    sink = WsTtsSink(ws)
    sink.start()
    sink.write(np.full(160, 0.5, dtype=np.float32))
    sink.flush()
    sink.write(np.full(160, -0.5, dtype=np.float32))
    sink.stop()
    await asyncio.sleep(0.05)
    assert len(ws.sent) == 1  # pre-flush frame dropped, post-flush frame sent
    assert len(ws.sent[0]) == 320  # 160 samples int16


class _TimedOutStream:
    # OutputStream stand-in recording write timestamps so the cancel-latency
    # budget can be asserted: no write may land >100ms after cancel() returns.
    def __init__(self, **kwargs):
        self.writes: list[float] = []

    def start(self):
        pass

    def write(self, frame):
        self.writes.append(time.monotonic())

    def stop(self):
        pass

    def close(self):
        pass


async def test_tts_cancel_stops_audio_under_100ms():
    # Acceptance #2 M-B: tts.cancel stops audio with bounded server-side
    # latency (<100 ms). Slow synthesis + fake out stream; after cancel, no
    # further writes within the budget and the player settles cancelled.
    cfg = ws_daemon_config()
    stream_holder = {}

    def factory(**kwargs):
        stream = _TimedOutStream(**kwargs)
        stream_holder["stream"] = stream
        return stream

    player = StreamingPlayer(cfg, synthesize=tts_engine_adapter(_SlowTTSEngine(cfg)), stream_factory=factory)
    speak_task = asyncio.get_running_loop().create_task(player.speak("long utterance"))
    for _ in range(200):
        if player.speaking:
            break
        await asyncio.sleep(0.01)
    assert player.speaking, "player never started speaking"
    t_cancel = time.monotonic()
    await player.cancel()
    result = await asyncio.wait_for(speak_task, timeout=5.0)
    late_writes = [t for t in stream_holder["stream"].writes if t > t_cancel + 0.1]
    assert not late_writes, f"{len(late_writes)} writes landed >100ms after cancel"
    assert result.get("cancelled") is True


def ws_daemon_config():
    # minimal config shape StreamingPlayer needs, isolated from env
    from fusion_speech.config import load_config

    return load_config()


# --- integration: /stream/audio over TestClient ---


def test_ws_session_full_flow_asr_final(ws_daemon):
    # Acceptance #2 M-A: WS-fed PCM -> VAD endpoint -> final ASR text — same
    # StreamingListener pipeline as the mic path (parity by construction).
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws:
        ws.send_json({"type": "start", "mode": "full_duplex", "encoding": "int16"})
        info = ws.receive_json()
        assert info["event"] == "speech.session.info"
        assert info["sample_rate"] == 16000
        # feed 0.4s speech then 0.8s silence -> VAD opens + closes a segment
        ws.send_bytes(_sine_pcm(0.4))
        ws.send_bytes(_silence_pcm(0.8))
        final = _recv_until(ws, lambda p: p.get("event") == "speech.asr" and p.get("state") == "final")
        assert final["text"].startswith("ws heard")
        ws.send_json({"type": "stop"})
    assert ws_daemon.listening is False
    assert ws_daemon.asr.listener is None


def test_ws_session_barge_in_cancels_tts(ws_daemon):
    # Acceptance #2 M-B: during TTS playback, injected speech triggers
    # speech.barge_in and playback stops (tts cancelled, session back to
    # listening).
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws:
        ws.send_json({"type": "start", "mode": "full_duplex", "tts_engine": "slow_stub"})
        info = ws.receive_json()  # session info
        ws.send_json({"type": "speak", "text": "playing while you talk"})
        _recv_until(ws, lambda p: p.get("event") == "speech.tts" and p.get("state") == "speaking")
        # user interrupts: loud speech into the (still-live) ASR stream
        ws.send_bytes(_sine_pcm(0.5))
        barge = _recv_until(ws, lambda p: p.get("event") == "speech.barge_in")
        assert barge["session"] == info["session"]
        cancelled = _recv_until(ws, lambda p: p.get("event") == "speech.tts" and p.get("state") == "cancelled")
        assert cancelled is not None
        listening = _recv_until(ws, lambda p: p.get("event") == "speech.session" and p.get("state") == "listening")
        assert listening is not None
        ws.send_json({"type": "stop"})


def test_ws_session_half_duplex_pauses_asr_during_speak(ws_daemon):
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws:
        ws.send_json({"type": "start", "mode": "half_duplex", "tts_engine": "slow_stub"})
        ws.receive_json()
        ws.send_json({"type": "speak", "text": "half duplex blocks asr"})
        _recv_until(ws, lambda p: p.get("event") == "speech.tts" and p.get("state") == "speaking")
        # ASR must be paused during playback (mic-era half-duplex behavior)
        assert ws_daemon.listening is False
        ws.send_json({"type": "tts_cancel"})
        _recv_until(ws, lambda p: p.get("event") == "speech.tts" and p.get("state") == "cancelled")
        # after cancel, half-duplex restarts ASR on the same WS source — poll
        # the daemon state (bounded), THEN read the guaranteed-queued event
        # (the listening event is broadcast before the flag flips, so once the
        # flag is True the event is already on the wire).
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not ws_daemon.listening:
            time.sleep(0.05)
        assert ws_daemon.listening, "half-duplex did not restart ASR after tts_cancel"
        _recv_until(ws, lambda p: p.get("event") == "speech.asr" and p.get("state") == "listening")
        ws.send_json({"type": "stop"})


def test_ws_session_second_stream_rejected_while_active(ws_daemon):
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws1:
        ws1.send_json({"type": "start"})
        ws1.receive_json()
        with client.websocket_connect("/stream/audio") as ws2:
            ws2.send_json({"type": "start"})
            err = ws2.receive_json()
            assert err["event"] == "speech.session.error"
            assert "ASR already active" in err["error"]
        ws1.send_json({"type": "stop"})


def test_ws_audio_before_start_rejected(ws_daemon):
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws:
        ws.send_bytes(_silence_pcm(0.1))
        err = ws.receive_json()
        assert err["event"] == "speech.session.error"
        assert "session not started" in err["error"]


def test_ws_oversize_frame_rejected(ws_daemon):
    ws_daemon.config.ws_max_frame_bytes = 1024
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"\x00" * 2048)
        err = ws.receive_json()
        assert "too large" in err["error"]
        ws.send_json({"type": "stop"})


def test_ws_ingress_disabled_closes_4404(ws_daemon):
    ws_daemon.config.ws_ingress_enabled = False
    client = TestClient(ws_daemon.build_app())
    with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect("/stream/audio"):
        pass
    assert exc_info.value.code == 4404


def test_ws_token_required(ws_daemon):
    ws_daemon.config.ws_ingress_token = "secret-token"
    client = TestClient(ws_daemon.build_app())
    with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect("/stream/audio"):
        pass
    assert exc_info.value.code == 4401
    with client.websocket_connect("/stream/audio?token=secret-token") as ws:
        ws.send_json({"type": "start"})
        info = ws.receive_json()
        assert info["event"] == "speech.session.info"
        ws.send_json({"type": "stop"})


def test_ws_unknown_control_frame(ws_daemon):
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws:
        ws.send_json({"type": "explode"})
        err = ws.receive_json()
        assert "unknown control type" in err["error"]


def test_ws_bad_json_frame(ws_daemon):
    client = TestClient(ws_daemon.build_app())
    with client.websocket_connect("/stream/audio") as ws:
        ws.send_text("{not json")
        err = ws.receive_json()
        assert "bad json" in err["error"]
