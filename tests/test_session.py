from __future__ import annotations

import pytest

from fusion_speech.session import (
    SESSION_CLOSED,
    SESSION_INTERRUPTED,
    SESSION_LISTENING,
    SESSION_SPEAKING,
    VoiceSession,
)


def _collect(session_id="s1", **kwargs):
    events = []

    async def on_event(payload):
        events.append(payload)

    s = VoiceSession(session_id, on_event=on_event, **kwargs)
    return s, events


async def test_lifecycle_listening_speaking_interrupted_listening():
    s, events = _collect()
    await s.start()
    assert s.state == SESSION_LISTENING
    assert await s.begin_speak() is True
    assert s.state == SESSION_SPEAKING
    assert s.speaking is True
    barge = await s.on_vad_onset()
    assert barge is not None and barge["event"] == "speech.barge_in"
    assert s.state == SESSION_INTERRUPTED
    await s.end_speak(cancelled=True)
    assert s.state == SESSION_LISTENING
    await s.stop()
    assert s.state == SESSION_CLOSED
    states = [e["state"] for e in events if e["event"] == "speech.session"]
    assert states == [
        SESSION_LISTENING,
        SESSION_SPEAKING,
        SESSION_INTERRUPTED,
        SESSION_LISTENING,
        SESSION_CLOSED,
    ]


async def test_normal_speak_cycle_no_interrupt():
    s, _ = _collect()
    await s.start()
    await s.begin_speak()
    assert await s.on_vad_onset() is not None  # barge-in latched
    await s.end_speak()
    assert s.state == SESSION_LISTENING


async def test_barge_in_disabled_ignored():
    s, events = _collect(barge_in=False)
    await s.start()
    await s.begin_speak()
    assert await s.on_vad_onset() is None
    assert s.state == SESSION_SPEAKING
    assert not any(e["event"] == "speech.barge_in" for e in events)


async def test_barge_in_ignored_when_not_speaking():
    s, _ = _collect()
    await s.start()
    assert await s.on_vad_onset() is None  # listening, not speaking


async def test_begin_speak_rejected_when_not_listening():
    s, _ = _collect()
    assert await s.begin_speak() is False  # idle
    await s.start()
    await s.begin_speak()
    assert await s.begin_speak() is False  # already speaking


async def test_start_idempotent_and_closed_raises():
    s, _ = _collect()
    await s.start()
    await s.start()  # idempotent
    assert s.state == SESSION_LISTENING
    await s.stop()
    with pytest.raises(RuntimeError):
        await s.start()


async def test_reprompt_on_empty_transcript():
    s, events = _collect()
    await s.start()
    event = await s.on_asr_final({"text": "   ", "confidence": 0.9})
    assert event is not None and event["event"] == "speech.reprompt"
    assert event["reason"] == "empty_transcript"


async def test_reprompt_on_low_confidence():
    s, events = _collect(reprompt_min_confidence=0.7)
    await s.start()
    event = await s.on_asr_final({"text": "hello", "confidence": 0.4})
    assert event is not None and event["reason"] == "low_confidence"


async def test_no_reprompt_on_good_final():
    s, events = _collect(reprompt_min_confidence=0.7)
    await s.start()
    assert await s.on_asr_final({"text": "hello", "confidence": 0.95}) is None
    # missing confidence (local engine does not report one) is accepted
    assert await s.on_asr_final({"text": "hello"}) is None
    assert not any(e["event"] == "speech.reprompt" for e in events)


async def test_barge_in_count_increments():
    s, _ = _collect()
    await s.start()
    await s.begin_speak()
    await s.on_vad_onset()
    await s.end_speak(cancelled=True)
    await s.begin_speak()
    await s.on_vad_onset()
    assert s.barge_in_count == 2
