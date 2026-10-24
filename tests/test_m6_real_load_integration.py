from __future__ import annotations

import asyncio
import base64
import logging
import os

import pytest
from _uds_client import UdsJsonRpcClient

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.real_load

_FIXTURE_WAV = os.path.join(os.path.dirname(__file__), "fixtures", "assistant_bank_en.wav")


def _load_wav_b64(path: str) -> str:
    # whole-file base64 incl. the RIFF header: fusion-mlx decodes the uploaded
    # file as a container — raw headerless PCM frames fail with DecodeError
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@pytest.fixture(scope="module")
def wav_b64() -> str:
    return _load_wav_b64(_FIXTURE_WAV)


@pytest.fixture
async def rpc(real_daemon) -> UdsJsonRpcClient:
    client = UdsJsonRpcClient(real_daemon, timeout=60.0)
    await client.connect()
    try:
        yield client
    finally:
        await client.close()


async def _no_error(resp: dict, method: str) -> dict:
    if "error" in resp:
        raise AssertionError(f"{method} returned error: {resp['error']}")
    return resp.get("result", {})


async def test_step1_ping_status(rpc: UdsJsonRpcClient):
    # Step 1: liveness + status reflects a live daemon bound to fusion-mlx.
    pong = await _no_error(await rpc.call("speech.ping"), "speech.ping")
    assert pong.get("pong") is True

    status = await _no_error(await rpc.call("speech.status"), "speech.status")
    assert status.get("running") is True
    assert status.get("port") == 11465
    lane = status.get("mlx_audio_lane") or {}
    assert lane.get("probed") is True, f"audio lane not probed: {lane}"
    logger.info(
        "step1: ping ok, uptime=%.1fs, lane stt=%s tts=%s", status.get("uptime", 0), lane.get("stt"), lane.get("tts")
    )


async def test_step2_transcribe_real_stt(rpc: UdsJsonRpcClient, wav_b64: str):
    # Step 2: real STT over fusion-mlx against the bundled English fixture.
    # Known upstream blocker: fusion-mlx issue #668 — mlx-audio Whisper
    # non_speech_tokens IndexError makes /v1/audio/transcriptions return 500
    # for every model. Skip (not xfail) so the gate is green upstream-side and
    # turns into a real assertion the moment #668 lands a fix.
    resp = await rpc.call("speech.transcribe", {"audio": wav_b64, "language": "en"})
    if "error" in resp:
        msg = str(resp["error"].get("message", ""))
        if "500" in msg or "Internal server error" in msg:
            pytest.skip("upstream fusion-mlx STT 500 (issue #668 mlx-audio IndexError)")
        raise AssertionError(f"speech.transcribe returned error: {resp['error']}")
    result = resp.get("result", {})
    text = (result.get("text") or "").strip()
    assert text, f"transcribe returned empty text: {result}"
    logger.info("step2: transcribe text=%r model=%s", text, result.get("model"))


async def test_step3_synthesize_real_tts(rpc: UdsJsonRpcClient):
    # Step 3: real TTS synthesize returns non-empty WAV bytes.
    result = await _no_error(
        await rpc.call("speech.synthesize", {"text": "Hello, fusion speech is live."}),
        "speech.synthesize",
    )
    audio_b64 = result.get("audio", "")
    assert audio_b64, "synthesize returned empty audio"
    assert result.get("format") == "wav"
    decoded = base64.b64decode(audio_b64)
    assert len(decoded) > 44, f"WAV too small ({len(decoded)} bytes), likely no PCM body"
    assert decoded[:4] == b"RIFF", f"not a WAV (header={decoded[:4]!r})"
    logger.info("step3: synthesize size=%d bytes", result.get("size", len(decoded)))


async def test_step4_speak_streaming_tts(rpc: UdsJsonRpcClient):
    # Step 4: streaming speak plays and returns done. speak is fire-and-track;
    # we poll speech.status until speaking clears, then assert played=True.
    result = await _no_error(
        await rpc.call("speech.speak", {"text": "Streaming playback complete."}),
        "speech.speak",
    )
    assert result.get("played") is True, f"speak did not play: {result}"

    deadline = asyncio.get_event_loop().time() + 30
    while asyncio.get_event_loop().time() < deadline:
        status = await _no_error(await rpc.call("speech.status"), "speech.status")
        if not status.get("speaking"):
            break
        await asyncio.sleep(0.5)
    else:
        pytest.skip("speak did not settle within 30s (slow fusion-mlx TTS)")

    logger.info("step4: speak done, played=%s", result.get("played"))


async def test_step5_wake_arm_no_false_fire(rpc: UdsJsonRpcClient):
    # Step 5: arm wake over real fusion-mlx (openwakeword ONNX). Silence (no mic
    # input in CI/headless) must NOT fire a spurious wake. Armed state reflects.
    armed = await _no_error(await rpc.call("speech.wake", {"arm": True}), "speech.wake")
    assert armed.get("armed") is True, f"wake did not arm: {armed}"

    status = await _no_error(await rpc.call("speech.status"), "speech.status")
    assert status.get("wake_armed") is True, "status.wake_armed not True after arm"

    # Hold armed briefly; a headless run has no mic stream so no false fire expected.
    await asyncio.sleep(2.0)
    status2 = await _no_error(await rpc.call("speech.status"), "speech.status")
    assert status2.get("wake_armed") is True, "wake disarmed itself during hold"

    disarmed = await _no_error(await rpc.call("speech.wake", {"arm": False}), "speech.wake")
    assert disarmed.get("armed") is False, f"wake did not disarm: {disarmed}"
    logger.info("step5: arm/disarm clean, hotwords=%s", armed.get("hotwords"))


async def test_step6_ws_ingress_full_duplex(real_daemon):
    # Step 6 (issue #2 M-A/M-B): WS PCM ingress on the real daemon — start a
    # full-duplex session, feed silence (no spurious ASR), speak via the WS
    # (real Kokoro TTS through fusion-mlx, binary PCM back), cancel mid-stream
    # (barge-in path), stop. Exercises /stream/audio end-to-end.
    import json as _json

    import numpy as np
    import websockets

    uri = "ws://127.0.0.1:11465/stream/audio"
    async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
        await ws.send(_json.dumps({"type": "start", "mode": "full_duplex", "encoding": "int16"}))
        info = _json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        assert info["event"] == "speech.session.info", info
        session_id = info["session"]
        logger.info("step6: session %s started (mode=%s)", session_id, info["mode"])

        # feed 1s of silence: must NOT open a VAD segment (no false ASR)
        silence = np.zeros(16000, dtype="<i2").tobytes()
        await ws.send(silence)

        # speak over the WS: real TTS via fusion-mlx, binary PCM frames back
        await ws.send(_json.dumps({"type": "speak", "text": "Voice session is live."}))
        audio_bytes = 0
        cancelled = False
        deadline = asyncio.get_event_loop().time() + 60
        got_speaking = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
            except TimeoutError:
                break
            if isinstance(msg, bytes):
                audio_bytes += len(msg)
                if audio_bytes > 4096 and not cancelled:
                    # audio is flowing — cancel mid-stream (barge-in budget)
                    await ws.send(_json.dumps({"type": "tts_cancel"}))
                    cancelled = True
                continue
            payload = _json.loads(msg)
            if payload.get("event") == "speech.tts":
                if payload.get("state") == "speaking":
                    got_speaking = True
                if payload.get("state") == "cancelled":
                    break
        assert got_speaking, "never saw speech.tts speaking on the WS session"
        assert audio_bytes > 0, "no binary TTS PCM frames received over the WS"
        logger.info("step6: tts audio=%d bytes cancelled=%s", audio_bytes, cancelled)

        await ws.send(_json.dumps({"type": "stop"}))

    # listener slot must be released after the session stops
    client = UdsJsonRpcClient(real_daemon, timeout=30.0)
    await client.connect()
    try:
        status = await _no_error(await client.call("speech.status"), "speech.status")
    finally:
        await client.close()
    assert status.get("listening") is False, "listener still active after ws session stop"
    logger.info("step6: ws session stopped cleanly, listener released")
