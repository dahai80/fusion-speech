from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_speech.config import SpeechConfig
from fusion_speech.mlx_bridge import AudioBridgeError, synthesize_stream, transcribe_stream


def _config():
    return SpeechConfig()


def _wav_header(sample_rate: int = 24000) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        0xFFFFFFFF,
    )


# E3: a 200 with a non-JSON body must raise an explicit AudioBridgeError
# carrying the status, not a bare JSONDecodeError with no context.
@pytest.mark.asyncio
async def test_e3_transcribe_malformed_200_raises_with_status():
    cfg = _config()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.side_effect = ValueError("not json")

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)

    fake_wrapper = MagicMock()
    fake_wrapper.client = fake_client
    fake_wrapper.close = AsyncMock()

    with (
        patch("fusion_speech.mlx_bridge._build_client", return_value=fake_wrapper),
        pytest.raises(AudioBridgeError) as exc,
    ):
        await transcribe_stream(cfg, b"pcm")

    assert exc.value.status == 200
    assert "malformed response" in str(exc.value)


# E3: a 200 with a malformed WAV (sample_rate=0) must abort the stream and
# raise rather than silently piping zero-rate PCM to the player.
@pytest.mark.asyncio
async def test_e3_synthesize_malformed_wav_aborts():
    cfg = _config()

    class _Resp:
        status_code = 200

        async def aiter_bytes(self):
            # valid RIFF/WAVE header but sample_rate field = 0
            yield struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                0xFFFFFFFF,
                b"WAVE",
                b"fmt ",
                16,
                1,
                1,
                0,
                0,
                2,
                16,
                b"data",
                0xFFFFFFFF,
            )

        async def aread(self):
            return b""

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *exc):
            return False

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_Ctx())

    fake_wrapper = MagicMock()
    fake_wrapper.client = fake_client
    fake_wrapper.close = AsyncMock()

    with (
        patch("fusion_speech.mlx_bridge._build_client", return_value=fake_wrapper),
        pytest.raises(AudioBridgeError) as exc,
    ):
        async for _ in synthesize_stream(cfg, "hi"):
            pass

    assert exc.value.status == 200
    assert "sample_rate=0" in str(exc.value)


# E1: speech.reprobe RPC refreshes audio_lane. Additive non-breaking method.
@pytest.mark.asyncio
async def test_e1_reprobe_refreshes_audio_lane(tmp_path, monkeypatch):
    from fusion_speech import server as server_mod

    monkeypatch.setattr(server_mod, "DRAIN_TIMEOUT", 0.5)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sock = str(run_dir / "fusion-speech.sock")

    cfg = SpeechConfig()
    cfg.socket_path = sock
    daemon = server_mod.SpeechDaemon(cfg)

    async def fake_probe():
        daemon.audio_lane = {"tts": True, "stt": True, "probed": True, "version": "B"}
        return daemon.audio_lane

    monkeypatch.setattr(daemon, "probe_audio_lane", fake_probe)
    from fusion_speech.ipc.methods import handle_reprobe

    result = await handle_reprobe({}, daemon)
    assert result["audio_lane"]["version"] == "B"


# R5: a wake/ASR resume failure after speak must broadcast a speech.tts error
# event with degraded=True and latch tts.degraded.
@pytest.mark.asyncio
async def test_r5_resume_failure_broadcasts_error_and_degraded(tmp_path, monkeypatch):
    from fusion_speech import server as server_mod

    monkeypatch.setattr(server_mod, "DRAIN_TIMEOUT", 0.5)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sock = str(run_dir / "fusion-speech.sock")

    cfg = SpeechConfig()
    cfg.socket_path = sock
    daemon = server_mod.SpeechDaemon(cfg)
    daemon._subscribers.clear()
    q = daemon.subscribe()

    # a wake engine that is active + paused so the resume path runs.
    # resume is a SYNC method (wake_word.WakeWordEngine.resume returns None),
    # so the failure must raise synchronously, not as an un-awaited coroutine.
    wake = MagicMock()
    wake.active = True
    wake.paused = True
    wake.resume = MagicMock(side_effect=RuntimeError("resume boom"))
    daemon._wake = wake
    daemon._listener = None  # was_listening False -> only wake resume path

    async def fake_play(*a, **k):
        return {"played": True}

    monkeypatch.setattr(daemon.tts, "_play", fake_play)

    await daemon.speak("hello")

    payload = await asyncio.wait_for(q.get(), timeout=1.0)
    assert payload["event"] == "speech.tts"
    assert payload["state"] == "error"
    assert payload["degraded"] is True
    assert "resume boom" in payload["error"]
    assert daemon.tts.degraded is True


# R5: a clean resume clears degraded (no false-sticky flag).
@pytest.mark.asyncio
async def test_r5_clean_resume_clears_degraded(tmp_path, monkeypatch):
    from fusion_speech import server as server_mod

    monkeypatch.setattr(server_mod, "DRAIN_TIMEOUT", 0.5)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sock = str(run_dir / "fusion-speech.sock")

    cfg = SpeechConfig()
    cfg.socket_path = sock
    daemon = server_mod.SpeechDaemon(cfg)
    daemon.tts.degraded = True  # pre-latched from a prior failure

    wake = MagicMock()
    wake.active = True
    wake.paused = True
    wake.resume = AsyncMock()
    daemon._wake = wake
    daemon._listener = None

    async def fake_play(*a, **k):
        return {"played": True}

    monkeypatch.setattr(daemon.tts, "_play", fake_play)

    await daemon.speak("hello")
    assert daemon.tts.degraded is False
