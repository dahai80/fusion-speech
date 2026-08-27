from __future__ import annotations

import pytest

# E4: SpeechConfig.__post_init__ fail-loud validation. Audio pipeline hard-
# contracts 16k mono + 1280-sample wake frames; bad env values must raise on
# construction, not silently corrupt detection at runtime.


def test_valid_default_config_constructs():
    from fusion_speech.config import SpeechConfig

    cfg = SpeechConfig()
    assert cfg.sample_rate == 16000
    assert cfg.channels == 1
    assert cfg.wake_chunk_samples == 1280
    assert cfg.ws_keepalive_s == 30.0
    assert cfg.ws_subscriber_queue == 64
    assert cfg.probe_interval_s == 60.0


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("sample_rate", 8000),
        ("channels", 2),
        ("wake_threshold", 0.0),
        ("wake_threshold", 1.5),
        ("wake_auto_listen_s", 0.0),
        ("wake_auto_listen_s", -1.0),
        ("wake_debounce_s", -0.5),
        ("vad_speech_threshold", 0.0),
        ("vad_speech_threshold", 1.0),
        ("tts_playback_buffer_s", 0.0),
        ("tts_playback_buffer_s", 15.0),
        ("chunk_ms", 0),
        ("chunk_ms", -100),
        ("port", -1),
        ("port", 70000),
        ("wake_chunk_samples", 2560),
        ("ws_keepalive_s", 0.0),
        ("ws_subscriber_queue", 0),
        ("probe_interval_s", 0.0),
    ],
)
def test_invalid_fields_raise(field, bad_value):
    from fusion_speech.config import SpeechConfig

    with pytest.raises(ValueError, match=field):
        SpeechConfig(**{field: bad_value})


def test_valid_boundary_values_accepted():
    from fusion_speech.config import SpeechConfig

    # port=0 = ephemeral bind (OS-assigned), valid in tests + headless.
    SpeechConfig(wake_threshold=1.0, vad_speech_threshold=0.99, tts_playback_buffer_s=10.0, port=0)
    SpeechConfig(wake_debounce_s=0.0, vad_speech_threshold=0.01, tts_playback_buffer_s=0.01, port=65535)
