from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_SOCKET_DIR", str(tmp_path / "sock"))
    monkeypatch.setenv("FUSION_SPEECH_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("FUSION_SPEECH_PORT", "0")
    yield


@pytest.fixture
def tmp_socket_dir(tmp_path):
    d = tmp_path / "uds"
    d.mkdir(mode=0o700)
    return str(d)
