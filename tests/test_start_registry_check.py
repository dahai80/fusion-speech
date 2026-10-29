from __future__ import annotations

import os
import subprocess

# Issue #7 / config Phase 3.3: start.sh must refuse to serve on a port that is
# not registered in port-registry.yaml. Semantics: refuse on unregistered port;
# FUSION_SKIP_REGISTRY_CHECK=1 bypasses; missing registry file = warn + continue.
# These tests exercise start.sh's start path WITHOUT spawning a daemon: the
# self-check runs before any state mutation, and assertions are message-based
# so a live daemon ("already running") still proves the check passed.

_START_SH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "start.sh"))

_GOOD_REGISTRY = """services:
  fusion-speech:
    port: 11465
    purpose: test copy
"""


def _run_start(tmp_path, registry_content=None, extra_env=None):
    registry = tmp_path / "port-registry.yaml"
    if registry_content is not None:
        registry.write_text(registry_content)
    # Pre-hold the start lock: if the real daemon is down, start.sh aborts at
    # the lock instead of actually spawning one from the unit suite. The
    # registry self-check runs before the lock, so its outcome is unaffected.
    lock_dir = os.path.join(os.path.dirname(_START_SH), ".fusion-speech.pid.lock")
    os.makedirs(lock_dir, exist_ok=True)
    env = {
        "FUSION_REGISTRY": str(registry),
        "FUSION_SOCKET_DIR": str(tmp_path / "sock"),
        "FUSION_SPEECH_LOG_DIR": str(tmp_path / "logs"),
        "FUSION_SPEECH_PORT": "11465",
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
    }
    env.update(extra_env or {})
    try:
        return subprocess.run(
            ["bash", _START_SH, "start"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        # Only remove the lock we pre-created — never one a real start holds.
        import contextlib

        with contextlib.suppress(OSError):
            os.rmdir(lock_dir)


def test_refuses_when_port_missing_from_registry(tmp_path):
    proc = _run_start(tmp_path, registry_content="services:\n  other-service:\n    port: 1\n")
    assert proc.returncode == 1
    assert "refusing to start" in (proc.stdout + proc.stderr)


def test_refuses_when_port_mismatches_registered_value(tmp_path):
    proc = _run_start(tmp_path, registry_content="services:\n  fusion-speech:\n    port: 12345\n")
    assert proc.returncode == 1
    assert "refusing to start" in (proc.stdout + proc.stderr)


def test_passes_when_entry_present(tmp_path):
    # A live daemon answers "already running"; a stopped one proceeds to spawn.
    # Either way the self-check must NOT refuse.
    proc = _run_start(tmp_path, registry_content=_GOOD_REGISTRY)
    assert "refusing to start" not in (proc.stdout + proc.stderr)


def test_bypass_env_skips_refusal(tmp_path):
    proc = _run_start(
        tmp_path,
        registry_content="services:\n",
        extra_env={"FUSION_SKIP_REGISTRY_CHECK": "1"},
    )
    assert "refusing to start" not in (proc.stdout + proc.stderr)
    assert "registry check bypassed" in (proc.stdout + proc.stderr)


def test_missing_registry_file_warns_and_continues(tmp_path):
    # FUSION_REGISTRY points at a path that is never written -> file missing.
    proc = _run_start(tmp_path, registry_content=None)
    assert "refusing to start" not in (proc.stdout + proc.stderr)
    assert "port-registry.yaml not found" in (proc.stdout + proc.stderr)
