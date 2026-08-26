from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_speech.config import SpeechConfig
from fusion_speech.mlx_bridge import (
    AudioBridgeError,
    _filter_audio_models,
    _parse_wav_sample_rate,
    synthesize_full,
    synthesize_stream,
    transcribe_stream,
)


def _wav_header(sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    block_align = channels * sample_width
    byte_rate = sample_rate * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b"data",
        0xFFFFFFFF,
    )


def _config():
    return SpeechConfig()


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeStreamResponse:
    def __init__(self, status_code: int, chunks: list[bytes], error_body: bytes = b""):
        self.status_code = status_code
        self._chunks = chunks
        self._error_body = error_body

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    async def aread(self):
        return self._error_body


@pytest.mark.asyncio
async def test_transcribe_stream_posts_multipart_and_returns_text():
    cfg = _config()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"text": "你好", "language": "zh", "duration": 1.2}

    fake_client = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_resp)
    fake_client.aclose = AsyncMock()

    fake_wrapper = MagicMock()
    fake_wrapper.client = fake_client
    fake_wrapper.close = AsyncMock()

    with patch("fusion_speech.mlx_bridge._build_client", return_value=fake_wrapper):
        result = await transcribe_stream(cfg, b"RIFF...pcm", language="zh")

    assert result["text"] == "你好"
    fake_client.post.assert_awaited_once()
    args, kwargs = fake_client.post.call_args
    assert args[0] == "/audio/transcriptions"
    assert kwargs["data"]["model"] == cfg.stt_model
    assert kwargs["data"]["language"] == "zh"
    assert "file" in kwargs["files"]
    fake_wrapper.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe_stream_raises_on_non_200():
    cfg = _config()
    fake_resp = MagicMock()
    fake_resp.status_code = 404
    fake_resp.json.return_value = {"error": {"message": "Model not found"}}

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

    assert exc.value.status == 404
    assert "Model not found" in str(exc.value)


@pytest.mark.asyncio
async def test_synthesize_stream_yields_wav_header_then_pcm():
    cfg = _config()
    header = _wav_header(24000)
    pcm1 = b"\x00\x01" * 100
    pcm2 = b"\x02\x03" * 50
    fake_resp = _FakeStreamResponse(200, [header, pcm1, pcm2])
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_FakeStreamCtx(fake_resp))

    fake_wrapper = MagicMock()
    fake_wrapper.client = fake_client
    fake_wrapper.close = AsyncMock()

    with patch("fusion_speech.mlx_bridge._build_client", return_value=fake_wrapper):
        chunks = []
        async for c in synthesize_stream(cfg, "你好"):
            chunks.append(c)

    assert chunks[0] == header
    assert b"".join(chunks) == header + pcm1 + pcm2
    fake_client.stream.assert_called_once()
    _, kwargs = fake_client.stream.call_args
    assert kwargs["json"]["model"] == cfg.tts_model
    assert kwargs["json"]["input"] == "你好"
    assert kwargs["json"]["stream"] is True


@pytest.mark.asyncio
async def test_synthesize_stream_raises_on_error_status():
    cfg = _config()
    fake_resp = _FakeStreamResponse(503, [], error_body=b'{"error":{"message":"busy"}}')
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_FakeStreamCtx(fake_resp))

    fake_wrapper = MagicMock()
    fake_wrapper.client = fake_client
    fake_wrapper.close = AsyncMock()

    with (
        patch("fusion_speech.mlx_bridge._build_client", return_value=fake_wrapper),
        pytest.raises(AudioBridgeError) as exc,
    ):
        async for _ in synthesize_stream(cfg, "x"):
            pass

    assert exc.value.status == 503


@pytest.mark.asyncio
async def test_synthesize_full_collects_all_chunks():
    cfg = _config()
    header = _wav_header(24000)
    pcm = b"\x10\x20" * 10
    fake_resp = _FakeStreamResponse(200, [header, pcm])
    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=_FakeStreamCtx(fake_resp))

    fake_wrapper = MagicMock()
    fake_wrapper.client = fake_client
    fake_wrapper.close = AsyncMock()

    with patch("fusion_speech.mlx_bridge._build_client", return_value=fake_wrapper):
        wav = await synthesize_full(cfg, "test")

    assert wav == header + pcm


def test_parse_wav_sample_rate_reads_24000():
    assert _parse_wav_sample_rate(_wav_header(24000)) == 24000
    assert _parse_wav_sample_rate(_wav_header(16000)) == 16000


def test_parse_wav_sample_rate_returns_zero_on_bad_header():
    assert _parse_wav_sample_rate(b"not a wav header at all...........") == 0
    assert _parse_wav_sample_rate(b"") == 0


def test_filter_audio_models_splits_stt_and_tts():
    models = [
        {"id": "mlx-community--whisper-large-v3-mlx", "modality": "audio"},
        {"id": "mlx-community--Kokoro-82M-bf16", "modality": "audio"},
        {"id": "Qwen3-TTS-12Hz-1.7B-Base-8bit", "modality": "audio"},
        {"id": "mlx-community--kitten-tts-nano-0.8", "modality": "audio"},
        {"id": "Llama-3.2-1B", "modality": "text"},
    ]
    stt, tts = _filter_audio_models(models)
    assert "mlx-community--whisper-large-v3-mlx" in stt
    assert "mlx-community--Kokoro-82M-bf16" in tts
    assert "Qwen3-TTS-12Hz-1.7B-Base-8bit" in tts
    assert "mlx-community--kitten-tts-nano-0.8" in tts
    assert "Llama-3.2-1B" not in stt and "Llama-3.2-1B" not in tts
