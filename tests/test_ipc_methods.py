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
    assert result["stt_model"] == "mlx-community/whisper-large-v3-mlx"
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
