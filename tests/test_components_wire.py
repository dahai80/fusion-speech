from __future__ import annotations

import pytest

from fusion_speech.components.asr_service import AsrService
from fusion_speech.components.audio_bus import AudioBus
from fusion_speech.components.event_bus import EventBus
from fusion_speech.components.tts_service import TtsService
from fusion_speech.components.wake_service import WakeService
from fusion_speech.config import load_config
from fusion_speech.server import SpeechDaemon


def test_daemon_wires_five_services():
    daemon = SpeechDaemon(load_config())
    assert isinstance(daemon.event_bus, EventBus)
    assert isinstance(daemon.audio_bus, AudioBus)
    assert isinstance(daemon.asr, AsrService)
    assert isinstance(daemon.wake, WakeService)
    assert isinstance(daemon.tts, TtsService)
    # services share the same bus + facade notifier instances. Bound methods
    # compare equal by __func__/__self__, not identity, so check the underlying
    # function + instance rather than `is` on the bound method.
    assert daemon.asr._audio_bus is daemon.audio_bus
    assert daemon.wake._audio_bus is daemon.audio_bus
    assert daemon.asr._notifier.__func__ is SpeechDaemon.broadcast
    assert daemon.asr._notifier.__self__ is daemon
    assert daemon.tts._notifier.__func__ is SpeechDaemon.broadcast
    assert daemon.tts._notifier.__self__ is daemon
    assert daemon.wake._on_wake.__func__ is SpeechDaemon._on_wake
    assert daemon.wake._on_wake.__self__ is daemon


def test_daemon_properties_delegate_to_services():
    daemon = SpeechDaemon(load_config())
    assert daemon._listener is None
    assert daemon.listening is False
    assert daemon._wake is None
    assert daemon.wake_armed is False
    assert daemon.hotwords == []
    assert daemon.speaking is False
    assert daemon._player is None
    assert daemon.audio_lane == {"tts": False, "stt": False, "probed": False, "error": ""}


def test_daemon_subscribers_alias_event_bus_set():
    daemon = SpeechDaemon(load_config())
    q1 = daemon.subscribe()
    q2 = daemon.subscribe()
    assert q1 in daemon._subscribers and q2 in daemon._subscribers
    assert daemon._subscribers is daemon.event_bus._subscribers
    assert len(daemon._subscribers) == 2
    daemon.unsubscribe(q1)
    daemon.unsubscribe(q2)
    assert len(daemon._subscribers) == 0


@pytest.mark.asyncio
async def test_audio_bus_borrow_register_release():
    bus = AudioBus(load_config())

    class _FakeSrc:
        stopped = False

        def stop(self):
            self.stopped = True

    # borrow before any registration returns None (caller builds its own)
    assert bus.borrow("wake") is None
    src = _FakeSrc()
    bus.register(src, "wake")
    # second consumer borrows the same canonical source
    assert bus.borrow("asr") is src
    assert bus.source is src
    # release one borrower keeps the source alive (refcount)
    bus.release("wake")
    assert bus.source is src
    # last release closes it
    bus.release("asr")
    assert bus.source is None
    assert src.stopped is True


@pytest.mark.asyncio
async def test_audio_bus_release_unregistered_owner_is_noop():
    bus = AudioBus(load_config())
    bus.release("asr")  # nothing registered under asr — must not raise
    assert bus.source is None


def test_import_graph_clean():
    # H4: hoisting engine/mlx/fastapi/uvicorn to top must not introduce a cycle.
    import fusion_speech.components.asr_service  # noqa: F401
    import fusion_speech.components.audio_bus  # noqa: F401
    import fusion_speech.components.event_bus  # noqa: F401
    import fusion_speech.components.tts_service  # noqa: F401
    import fusion_speech.components.wake_service  # noqa: F401
    import fusion_speech.ipc.methods  # noqa: F401
    import fusion_speech.server  # noqa: F401
