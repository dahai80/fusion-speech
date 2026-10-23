from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

from fusion_core import config as _core_config

logger = _core_config.logger

DEFAULT_PORT = 11465
DEFAULT_SOCKET_NAME = "fusion-speech.sock"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_STT_MODEL = "openai/whisper-tiny"
DEFAULT_TTS_MODEL = "kokoro"
DEFAULT_TTS_VOICE = "af_heart"
DEFAULT_LANGUAGE = "en"
VAD_SPEECH_THRESHOLD = 0.3
CHUNK_MS = 300
DEFAULT_WAKE_WORD = "hey_jarvis"
DEFAULT_WAKE_THRESHOLD = 0.5
DEFAULT_WAKE_AUTO_LISTEN_S = 8.0
DEFAULT_WAKE_DEBOUNCE_S = 1.0
DEFAULT_WAKE_MODEL_DIR = "~/.fusion-speech/models/wakeword"
DEFAULT_WAKE_MODEL_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
DEFAULT_TTS_BUFFER_S = 2.0
DEFAULT_WS_KEEPALIVE_S = 30.0
DEFAULT_WS_SUBSCRIBER_QUEUE = 64
DEFAULT_WAKE_CHUNK_SAMPLES = 1280
DEFAULT_PROBE_INTERVAL_S = 60.0
DEFAULT_WS_INGRESS = True
DEFAULT_WS_TOKEN = ""
DEFAULT_WS_MAX_FRAME_BYTES = 65536
DEFAULT_BARGE_IN = True
DEFAULT_REPROMPT_CONFIDENCE = 0.0
REQUIRED_SAMPLE_RATE = 16000
REQUIRED_CHANNELS = 1
REQUIRED_WAKE_CHUNK = 1280


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        logger.warning("config: %s=%r not int, fallback default %d", key, raw, default)
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        logger.warning("config: %s=%r not float, fallback default %s", key, raw, default)
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class SpeechConfig:
    port: int = field(default_factory=lambda: _env_int("FUSION_SPEECH_PORT", DEFAULT_PORT))
    socket_path: str = field(default_factory=lambda: _resolve_socket_path())
    sample_rate: int = field(default_factory=lambda: _env_int("FUSION_SPEECH_SAMPLE_RATE", DEFAULT_SAMPLE_RATE))
    channels: int = 1
    stt_model: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_STT_MODEL", DEFAULT_STT_MODEL))
    tts_model: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_TTS_MODEL", DEFAULT_TTS_MODEL))
    tts_voice: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_TTS_VOICE", DEFAULT_TTS_VOICE))
    language: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_LANGUAGE", DEFAULT_LANGUAGE))
    mlx_base_url: str = field(default_factory=lambda: _core_config.default_mlx_base_url())
    mlx_api_key: str = field(default_factory=lambda: _core_config.resolve_api_key() or "")
    vad_speech_threshold: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_VAD_THRESHOLD", VAD_SPEECH_THRESHOLD)
    )
    chunk_ms: int = field(default_factory=lambda: _env_int("FUSION_SPEECH_CHUNK_MS", CHUNK_MS))
    wake_default_word: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_WAKE_WORD", DEFAULT_WAKE_WORD))
    wake_threshold: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_WAKE_THRESHOLD", DEFAULT_WAKE_THRESHOLD)
    )
    wake_auto_listen_s: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_WAKE_AUTO_LISTEN_S", DEFAULT_WAKE_AUTO_LISTEN_S)
    )
    wake_debounce_s: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_WAKE_DEBOUNCE_S", DEFAULT_WAKE_DEBOUNCE_S)
    )
    wake_model_dir: str = field(
        default_factory=lambda: os.path.expanduser(
            os.environ.get("FUSION_SPEECH_WAKE_MODEL_DIR", DEFAULT_WAKE_MODEL_DIR)
        )
    )
    wake_model_url: str = field(
        default_factory=lambda: os.environ.get("FUSION_SPEECH_WAKE_MODEL_URL", DEFAULT_WAKE_MODEL_URL)
    )
    tts_playback_buffer_s: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_TTS_BUFFER_S", DEFAULT_TTS_BUFFER_S)
    )
    ws_keepalive_s: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_WS_KEEPALIVE_S", DEFAULT_WS_KEEPALIVE_S)
    )
    ws_subscriber_queue: int = field(
        default_factory=lambda: _env_int("FUSION_SPEECH_WS_SUBSCRIBER_QUEUE", DEFAULT_WS_SUBSCRIBER_QUEUE)
    )
    wake_chunk_samples: int = field(
        default_factory=lambda: _env_int("FUSION_SPEECH_WAKE_CHUNK_SAMPLES", DEFAULT_WAKE_CHUNK_SAMPLES)
    )
    probe_interval_s: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_PROBE_INTERVAL_S", DEFAULT_PROBE_INTERVAL_S)
    )
    # M-A/M-B (issue #2): WS PCM ingress on /stream/audio. Token is optional —
    # the HTTP port is localhost-bound; multi-tenant auth stays upstream
    # (fusion-identity / control plane) per the issue's non-goals.
    ws_ingress_enabled: bool = field(
        default_factory=lambda: _env_bool("FUSION_SPEECH_WS_INGRESS", DEFAULT_WS_INGRESS)
    )
    ws_ingress_token: str = field(
        default_factory=lambda: os.environ.get("FUSION_SPEECH_WS_TOKEN", DEFAULT_WS_TOKEN).strip()
    )
    ws_max_frame_bytes: int = field(
        default_factory=lambda: _env_int("FUSION_SPEECH_WS_MAX_FRAME", DEFAULT_WS_MAX_FRAME_BYTES)
    )
    barge_in_enabled: bool = field(
        default_factory=lambda: _env_bool("FUSION_SPEECH_BARGE_IN", DEFAULT_BARGE_IN)
    )
    reprompt_min_confidence: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_REPROMPT_CONFIDENCE", DEFAULT_REPROMPT_CONFIDENCE)
    )
    tts_output_device: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_TTS_OUTPUT_DEVICE", ""))
    log_level: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_LOG_LEVEL", "INFO").upper())
    log_dir: str = field(
        default_factory=lambda: os.path.expanduser(os.environ.get("FUSION_SPEECH_LOG_DIR", "~/.fusion-speech/logs"))
    )

    def __post_init__(self) -> None:
        # E4: fail-loud validation. Audio pipeline (Silero VAD + openwakeword)
        # hard-contracts 16k mono + 1280-sample (80ms) wake frames. A non-16000
        # sample_rate or non-mono channels silently corrupts detection; raise so
        # startup fails instead of running broken. Other fields bound-checked to
        # catch bad env values early (negative buffers, out-of-range thresholds).
        if self.sample_rate != REQUIRED_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate must be {REQUIRED_SAMPLE_RATE} (VAD/wake/openwakeword contract), got {self.sample_rate}"
            )
        if self.channels != REQUIRED_CHANNELS:
            raise ValueError(f"channels must be {REQUIRED_CHANNELS} (mono pipeline), got {self.channels}")
        if not 0 < self.wake_threshold <= 1:
            raise ValueError(f"wake_threshold must be in (0, 1], got {self.wake_threshold}")
        if self.wake_auto_listen_s <= 0:
            raise ValueError(f"wake_auto_listen_s must be > 0, got {self.wake_auto_listen_s}")
        if self.wake_debounce_s < 0:
            raise ValueError(f"wake_debounce_s must be >= 0, got {self.wake_debounce_s}")
        if not 0 < self.vad_speech_threshold < 1:
            raise ValueError(f"vad_speech_threshold must be in (0, 1), got {self.vad_speech_threshold}")
        if not 0 < self.tts_playback_buffer_s <= 10:
            raise ValueError(f"tts_playback_buffer_s must be in (0, 10], got {self.tts_playback_buffer_s}")
        if self.chunk_ms <= 0:
            raise ValueError(f"chunk_ms must be > 0, got {self.chunk_ms}")
        if not 0 <= self.port <= 65535:
            raise ValueError(f"port must be in [0, 65535] (0 = ephemeral), got {self.port}")
        if self.wake_chunk_samples != REQUIRED_WAKE_CHUNK:
            raise ValueError(
                f"wake_chunk_samples must be {REQUIRED_WAKE_CHUNK} (openwakeword 80ms frame contract), got {self.wake_chunk_samples}"
            )
        if self.ws_keepalive_s <= 0:
            raise ValueError(f"ws_keepalive_s must be > 0, got {self.ws_keepalive_s}")
        if self.ws_subscriber_queue <= 0:
            raise ValueError(f"ws_subscriber_queue must be > 0, got {self.ws_subscriber_queue}")
        if self.probe_interval_s <= 0:
            raise ValueError(f"probe_interval_s must be > 0, got {self.probe_interval_s}")
        if self.ws_max_frame_bytes <= 0:
            raise ValueError(f"ws_max_frame_bytes must be > 0, got {self.ws_max_frame_bytes}")
        if not 0 <= self.reprompt_min_confidence <= 1:
            raise ValueError(
                f"reprompt_min_confidence must be in [0, 1] (0 = disabled), got {self.reprompt_min_confidence}"
            )


def _resolve_socket_path() -> str:
    socket_dir = os.environ.get("FUSION_SOCKET_DIR", "").strip()
    if not socket_dir:
        socket_dir = os.path.expanduser("~/.fusion-speech/run")
    socket_dir = os.path.expanduser(socket_dir)
    os.makedirs(socket_dir, mode=0o700, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(socket_dir, 0o700)
    logger.info("config: socket dir=%s", socket_dir)
    return os.path.join(socket_dir, DEFAULT_SOCKET_NAME)


def load_config() -> SpeechConfig:
    cfg = SpeechConfig()
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger.info(
        "config loaded port=%d socket=%s sample_rate=%d stt=%s tts=%s/%s mlx=%s wake=%s/%.2f",
        cfg.port,
        cfg.socket_path,
        cfg.sample_rate,
        cfg.stt_model,
        cfg.tts_model,
        cfg.tts_voice,
        cfg.mlx_base_url,
        cfg.wake_default_word,
        cfg.wake_threshold,
    )
    return cfg
