from __future__ import annotations

import asyncio
import wave
from io import BytesIO

import numpy as np
import pytest

from fusion_speech.engines import (
    AsrEngineError,
    LocalASREngine,
    LocalTTSEngine,
    TTSEngine,
    make_asr_engine,
    make_tts_engine,
    register_tts_engine,
    registered_asr_engines,
    registered_tts_engines,
    tts_engine_adapter,
)
from fusion_speech.tts_playback import StreamingPlayer


class _Cfg:
    sample_rate = 16000
    channels = 1
    language = "en"
    tts_model = "kokoro"
    tts_voice = "af_heart"
    tts_playback_buffer_s = 1.0
    tts_output_device = ""


async def _fake_transcribe(config, wav, **kw):
    # returns text derived from the PCM length so tests can assert the payload
    with wave.open(BytesIO(wav)) as w:
        frames = w.getnframes()
    return {"text": f"heard {frames} frames", "language": kw.get("language", "en"), "duration": frames / 16000}


async def test_local_asr_engine_feed_finish_roundtrip():
    engine = LocalASREngine(_Cfg(), transcribe=_fake_transcribe)
    await engine.start_stream(language="en")
    pcm = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3200) / 16000)).astype(np.float32)
    await engine.feed(pcm)
    result = await engine.finish()
    assert result["text"] == "heard 3200 frames"
    assert result["language"] == "en"


async def test_local_asr_engine_rejects_feed_before_start():
    engine = LocalASREngine(_Cfg(), transcribe=_fake_transcribe)
    with pytest.raises(AsrEngineError):
        await engine.feed(np.zeros(16, dtype=np.float32))


async def test_local_asr_engine_finish_empty():
    engine = LocalASREngine(_Cfg(), transcribe=_fake_transcribe)
    await engine.start_stream()
    result = await engine.finish()
    assert result == {"text": "", "language": "en", "duration": 0.0}


async def test_local_asr_engine_cancel_discards():
    engine = LocalASREngine(_Cfg(), transcribe=_fake_transcribe)
    await engine.start_stream()
    await engine.feed(np.zeros(1600, dtype=np.float32))
    await engine.cancel()
    with pytest.raises(AsrEngineError):
        await engine.finish()


async def test_local_asr_engine_emits_events():
    events = []

    async def on_event(payload):
        events.append(payload)

    engine = LocalASREngine(_Cfg(), transcribe=_fake_transcribe, on_event=on_event)
    await engine.start_stream()
    await engine.feed(np.zeros(1600, dtype=np.float32))
    await engine.finish()
    states = [e["state"] for e in events if e["event"] == "speech.asr.engine"]
    assert states == ["listening", "transcribing"]


def _wav_bytes(rate: int, samples: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


async def _fake_bridge(config, text, *, voice="", model="", ref_audio="", ref_text=""):
    # mirrors mlx_bridge.synthesize_stream's extended signature (incl. cloning
    # params) so LocalTTSEngine is exercised against the real call shape
    payload = _wav_bytes(config.sample_rate, 1600)
    for i in range(0, len(payload), 800):
        yield payload[i : i + 800]


async def test_local_tts_engine_streams_wav_and_forwards_cloning_params():
    captured = {}

    async def bridge(config, text, *, voice="", model="", ref_audio="", ref_text=""):
        captured["ref_audio"] = ref_audio
        captured["ref_text"] = ref_text
        async for chunk in _fake_bridge(config, text, voice=voice, model=model):
            yield chunk

    engine = LocalTTSEngine(_Cfg(), synthesize=bridge)
    chunks = [c async for c in engine.synthesize("hi", ref_audio="ref.wav", ref_text="hello there")]
    assert chunks and chunks[0][:4] == b"RIFF"
    assert captured == {"ref_audio": "ref.wav", "ref_text": "hello there"}


async def test_local_tts_engine_default_has_no_cloning_params():
    # empty cloning params must not leak into the bridge call (fusion-mlx
    # rejects/ignores unknown body fields — defaults stay clean)
    captured = {}

    async def bridge(config, text, *, voice="", model="", ref_audio="", ref_text=""):
        captured.update(ref_audio=ref_audio, ref_text=ref_text)
        async for chunk in _fake_bridge(config, text, voice=voice, model=model):
            yield chunk

    engine = LocalTTSEngine(_Cfg(), synthesize=bridge)
    _ = [c async for c in engine.synthesize("hi")]
    assert captured == {"ref_audio": "", "ref_text": ""}


def test_registry_unknown_engine_rejected():
    with pytest.raises(AsrEngineError, match="unknown asr engine"):
        make_asr_engine("nope", _Cfg())
    with pytest.raises(AsrEngineError, match="unknown tts engine"):
        make_tts_engine("nope", _Cfg())


def test_registry_lists_registered():
    assert "local" in registered_asr_engines()
    assert "local" in registered_tts_engines()


class _FakeOutStream:
    def __init__(self, **kwargs):
        self.writes: list[float] = []

    def start(self):
        pass

    def write(self, frame):
        import time

        self.writes.append(time.monotonic())

    def stop(self):
        pass

    def close(self):
        pass


async def _player_conformance(engine: TTSEngine) -> dict:
    # Acceptance (#2): a cloud TTSEngine stub must pass the same suite as the
    # local engine — drive StreamingPlayer through tts_engine_adapter, the
    # exact wiring the WS session uses.
    cfg = _Cfg()
    player = StreamingPlayer(
        cfg,
        synthesize=tts_engine_adapter(engine),
        stream_factory=lambda **kw: _FakeOutStream(**kw),
    )
    return await player.speak("conformance", voice="af_heart")


async def test_tts_engine_conformance_local_and_stub():
    # stub = the shape a cloud engine (iFlytek WS etc.) would register
    class StubCloudTTSEngine(TTSEngine):
        name = "stub_cloud"

        def synthesize(self, text, *, voice="", model="", ref_audio="", ref_text=""):
            return self._gen(text)

        async def _gen(self, text):
            payload = _wav_bytes(16000, 4800)
            for i in range(0, len(payload), 1200):
                await asyncio.sleep(0.001)
                yield payload[i : i + 1200]

        async def cancel(self):
            pass

    register_tts_engine("stub_cloud", lambda config, **kw: StubCloudTTSEngine())
    assert "stub_cloud" in registered_tts_engines()
    local = LocalTTSEngine(_Cfg(), synthesize=_fake_bridge)
    stub = make_tts_engine("stub_cloud", _Cfg())
    for engine in (local, stub):
        result = await _player_conformance(engine)
        assert result["played"] is True, engine.name
        assert result["chars"] == len("conformance")


async def test_adapter_shape_matches_player_call():
    # player calls synthesize(config, text, voice=..., model=..., stream=True);
    # the adapter must translate that to the engine contract
    seen = {}

    class _Probe(TTSEngine):
        name = "probe"

        def synthesize(self, text, *, voice="", model="", ref_audio="", ref_text=""):
            seen["text"] = text
            seen["voice"] = voice
            seen["ref_audio"] = ref_audio
            return self._gen()

        async def _gen(self):
            yield _wav_bytes(16000, 512)

        async def cancel(self):
            pass

    adapter = tts_engine_adapter(_Probe(), ref_audio="r.wav")
    chunks = [c async for c in adapter(None, "probe text", voice="v1", model="m1", stream=True)]
    assert chunks and chunks[0][:4] == b"RIFF"
    assert seen == {"text": "probe text", "voice": "v1", "ref_audio": "r.wav"}
