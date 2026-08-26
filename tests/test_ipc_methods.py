from __future__ import annotations

import pytest

from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher
from fusion_speech.ipc.methods import build_methods
from fusion_speech.server import SpeechDaemon


@pytest.mark.asyncio
async def test_ping_returns_pong():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.ping"})
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["pong"] is True
    assert "timestamp" in resp["result"]


@pytest.mark.asyncio
async def test_status_returns_lane_and_models():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 2, "method": "speech.status"})
    result = resp["result"]
    assert result["stt_model"] == "openai/whisper-tiny"
    assert result["tts_model"] == "mlx-community/Kokoro-82M-bf16"
    assert result["port"] == daemon.port
    assert "mlx_audio_lane" in result


@pytest.mark.asyncio
async def test_unknown_method_returns_32601():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 3, "method": "speech.nope"})
    assert resp["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_missing_jsonrpc_returns_32600():
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch({"id": 4, "method": "speech.ping"})
    assert resp["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_str_params_normalized(tmp_socket_dir):
    daemon = SpeechDaemon(load_config())
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    resp = await dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "speech.ping",
            "params": '{"k": 1}',
        }
    )
    assert resp["result"]["pong"] is True


@pytest.mark.asyncio
async def test_handler_error_has_type_name_not_empty():
    class EmptyTimeout(Exception):
        pass

    handlers = {"speech.boom": lambda params: _raise(EmptyTimeout())}
    dispatcher = JsonRpcDispatcher(handlers)
    resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 6, "method": "speech.boom"})
    assert resp["error"]["code"] == -32000
    assert resp["error"]["message"] == "EmptyTimeout"


async def _raise(exc):
    raise exc


@pytest.mark.asyncio
async def test_handler_error_includes_status_for_audio_bridge_error():
    from fusion_speech.mlx_bridge import AudioBridgeError

    handlers = {"speech.boom": lambda params: _raise(AudioBridgeError("no model", status=404))}
    dispatcher = JsonRpcDispatcher(handlers)
    resp = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 7, "method": "speech.boom"})
    assert resp["error"]["code"] == -32000
    assert resp["error"]["message"] == "AudioBridgeError (status 404): no model"


def test_resolve_tts_model_falls_back_when_configured_missing():
    from fusion_speech.ipc.methods import _resolve_tts_model

    daemon = SpeechDaemon(load_config())
    daemon.audio_lane = {"tts": True, "tts_models": ["Qwen3-TTS-12Hz-1.7B-Base-8bit"]}
    assert _resolve_tts_model(daemon) == "Qwen3-TTS-12Hz-1.7B-Base-8bit"


def test_resolve_tts_model_keeps_configured_when_present():
    from fusion_speech.ipc.methods import _resolve_tts_model

    daemon = SpeechDaemon(load_config())
    daemon.audio_lane = {"tts": True, "tts_models": [daemon.config.tts_model, "other"]}
    assert _resolve_tts_model(daemon) == daemon.config.tts_model


def test_resolve_stt_model_keeps_configured_when_no_probed_set():
    from fusion_speech.ipc.methods import _resolve_stt_model

    daemon = SpeechDaemon(load_config())
    daemon.audio_lane = {"stt": False, "stt_models": []}
    assert _resolve_stt_model(daemon) == daemon.config.stt_model
