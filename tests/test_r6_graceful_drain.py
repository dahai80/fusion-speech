from __future__ import annotations

import asyncio
import contextlib

import pytest

# R6: stop must drain bounded — no wedged component hangs shutdown. Broadcasts
# speech.shutdown to WS subscribers before tearing down; each leaf stop wrapped
# in wait_for(DRAIN_TIMEOUT). A slow/hung stop returns within the timeout
# instead of hanging; the WS subscriber receives the shutdown frame.


@pytest.mark.asyncio
async def test_stop_broadcasts_shutdown_and_drains_bounded(monkeypatch):
    from fusion_speech import server as server_mod
    from fusion_speech.config import load_config
    from fusion_speech.server import SpeechDaemon

    # shrink the drain timeout so the test runs fast; the real value is 10s.
    monkeypatch.setattr(server_mod, "DRAIN_TIMEOUT", 0.2)

    daemon = SpeechDaemon(load_config())
    q = daemon.subscribe()

    # inject a listener whose stop hangs forever — must be abandoned by the
    # wait_for backstop, not hang stop().
    class _HungStop:
        active = True

        async def stop(self):
            await asyncio.Event().wait()  # never resolves

    daemon._listener = _HungStop()
    daemon.listening = True

    loop = asyncio.get_running_loop()
    stop_task = loop.create_task(daemon.stop())
    await asyncio.wait_for(stop_task, timeout=2.0)
    assert stop_task.done()

    # shutdown frame must have been broadcast before teardown.
    seen_shutdown = False
    with contextlib.suppress(asyncio.QueueEmpty):
        while True:
            payload = q.get_nowait()
            if payload.get("event") == "speech.shutdown":
                seen_shutdown = True
                break
    assert seen_shutdown, "speech.shutdown not broadcast on stop"


@pytest.mark.asyncio
async def test_stop_clean_state_after_normal_components(monkeypatch):
    from fusion_speech import server as server_mod
    from fusion_speech.config import load_config
    from fusion_speech.server import SpeechDaemon

    monkeypatch.setattr(server_mod, "DRAIN_TIMEOUT", 0.5)
    daemon = SpeechDaemon(load_config())

    await daemon.stop()
    assert daemon._player is None
    assert daemon._wake is None
    assert daemon._listener is None
    assert daemon.listening is False
    assert daemon.wake_armed is False
    assert daemon.speaking is False
    assert daemon.running is False
