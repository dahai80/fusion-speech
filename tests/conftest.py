from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time

import httpx
import pytest

logger = logging.getLogger(__name__)

_ISOLATE_VARS = ("FUSION_SOCKET_DIR", "FUSION_SPEECH_LOG_DIR", "FUSION_SPEECH_PORT")
_MLX_URL = "http://127.0.0.1:11434"
_SPEECH_HEALTH = "http://127.0.0.1:11465/health"
_REAL_SOCK = os.path.expanduser("~/.fusion-speech/run/fusion-speech.sock")


def _mlx_reachable() -> bool:
    # 200 = healthy; 401 = up but needs key (still reachable). 5xx/conn-error = down.
    headers = {}
    key = os.environ.get("FUSION_MLX_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = httpx.get(f"{_MLX_URL}/v1/models", headers=headers, timeout=2.0)
        return r.status_code in (200, 401)
    except Exception:
        return False


def _speech_reachable() -> bool:
    try:
        r = httpx.get(_SPEECH_HEALTH, timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _real_load_enabled() -> bool:
    return os.environ.get("FUSION_SPEECH_REAL_LOAD") == "1"


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


@pytest.fixture(scope="session")
def real_daemon():
    # Session fixture: ensures the REAL fusion-speech daemon is up on port 11465
    # with the real socket (~/.fusion-speech/run/fusion-speech.sock), backed by a
    # REAL fusion-mlx on 11434. Opt-in only (FUSION_SPEECH_REAL_LOAD=1 + mlx up).
    # Spawns via start.sh with the autouse isolation vars stripped so the daemon
    # binds its real defaults, not the per-test tmp socket/port. If a daemon was
    # already running it is left alone (not torn down).
    if not _real_load_enabled():
        pytest.skip("FUSION_SPEECH_REAL_LOAD=1 not set; skipping real-load integration")
    if not _mlx_reachable():
        pytest.skip("fusion-mlx not reachable on 11434 (start ~/fusion/fusion-mlx/start.sh start)")

    started_here = False
    if _speech_reachable():
        logger.info("real_daemon: already running, reusing")
    else:
        start_sh = os.path.join(os.path.dirname(__file__), "..", "start.sh")
        start_sh = os.path.abspath(start_sh)
        if not shutil.which(start_sh) and not os.path.isfile(start_sh):
            pytest.skip(f"start.sh not found at {start_sh}")
        # Strip the autouse isolation env so start.sh binds real port 11465 + real sock.
        env = {k: v for k, v in os.environ.items() if k not in _ISOLATE_VARS}
        logger.info("real_daemon: starting via %s", start_sh)
        proc = subprocess.run(
            ["bash", start_sh, "start"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            pytest.skip(f"start.sh start failed: {proc.stdout!r} {proc.stderr!r}")
        started_here = True

    # Poll /health (boot can take a few seconds for the mlx audio-lane probe).
    deadline = time.time() + 30
    while time.time() < deadline:
        if _speech_reachable():
            break
        time.sleep(0.5)
    else:
        if started_here:
            subprocess.run(["bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "start.sh")), "stop"],
                           capture_output=True, timeout=30)
        pytest.skip("fusion-speech /health not reachable within 30s")

    if not os.path.exists(_REAL_SOCK):
        pytest.skip(f"real socket missing: {_REAL_SOCK}")

    yield _REAL_SOCK

    if started_here:
        logger.info("real_daemon: stopping (this fixture started it)")
        subprocess.run(
            ["bash", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "start.sh")), "stop"],
            capture_output=True,
            timeout=30,
        )


def _socket_exists(path: str) -> bool:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(path)
        s.close()
        return True
    except Exception:
        return False

