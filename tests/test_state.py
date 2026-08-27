from __future__ import annotations

import asyncio
import json
import os

import pytest

# E6: DaemonState persists restart_count + uptime across restarts. A first boot
# writes restart_count=1; a second boot (simulated crash recovery, no graceful
# stop in between) bumps to 2 and the facade broadcasts speech.restarted. A
# graceful stop records uptime_at_stop. Corrupt JSON -> fresh start, no crash.


def _tmp_socket(tmp_path) -> str:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return str(run_dir / "fusion-speech.sock")


def test_state_first_start_writes_restart_count_1(tmp_path):
    from fusion_speech.state import DaemonState

    sock = _tmp_socket(tmp_path)
    st = DaemonState(sock)
    n = st.enter_start()
    assert n == 1
    assert st.restart_count == 1
    assert os.path.exists(st.path)
    with open(st.path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["restart_count"] == 1
    assert data["start_time"] > 0


def test_state_second_start_bumps_restart_count(tmp_path):
    from fusion_speech.state import DaemonState

    sock = _tmp_socket(tmp_path)
    st1 = DaemonState(sock)
    st1.enter_start()
    # simulate crash: no enter_stop, new process builds a new DaemonState
    st2 = DaemonState(sock)
    n = st2.enter_start()
    assert n == 2
    assert st2.restart_count == 2


def test_state_stop_records_uptime(tmp_path):
    from fusion_speech.state import DaemonState

    sock = _tmp_socket(tmp_path)
    st = DaemonState(sock)
    st.enter_start()
    st.enter_stop()
    with open(st.path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["uptime_at_stop"] >= 0.0
    # restart_count preserved across graceful stop
    assert data["restart_count"] == 1


def test_state_corrupt_json_starts_fresh(tmp_path):
    from fusion_speech.state import DaemonState

    sock = _tmp_socket(tmp_path)
    run_dir = os.path.dirname(sock)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "state.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")
    st = DaemonState(sock)
    st.load()
    assert st.restart_count == 0
    n = st.enter_start()
    assert n == 1


@pytest.mark.asyncio
async def test_facade_broadcasts_restarted_on_second_start(tmp_path, monkeypatch):
    # Point the socket (and thus state.json) into tmp_path so the test does
    # not touch the real ~/.fusion-speech/run. Use a daemon that never binds
    # a real UDS/port — we only exercise enter_start + the broadcast path.
    from fusion_speech import server as server_mod
    from fusion_speech.config import SpeechConfig

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sock = str(run_dir / "fusion-speech.sock")
    monkeypatch.setattr(server_mod, "DRAIN_TIMEOUT", 0.5)

    # Build a daemon with a config pointing at tmp socket; skip start() (which
    # binds UDS + uvicorn). Drive enter_start + broadcast manually so the test
    # isolates the E6 behavior without a live server.
    cfg = SpeechConfig()
    cfg.socket_path = sock
    daemon = server_mod.SpeechDaemon(cfg)
    daemon._subscribers.clear()
    q = daemon.subscribe()

    # first boot
    daemon._restart_count = daemon.state.enter_start()
    assert daemon._restart_count == 1
    # count == 1 -> _announce_restart is a no-op (first boot, not a restart)
    await daemon._announce_restart()
    assert q.empty()
    # second boot (crash recovery): bump + broadcast speech.restarted
    daemon._restart_count = daemon.state.enter_start()
    assert daemon._restart_count == 2
    await daemon._announce_restart()
    payload = await asyncio.wait_for(q.get(), timeout=1.0)
    assert payload["event"] == "speech.restarted"
    assert payload["restart_count"] == 2
