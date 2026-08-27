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
    tts_output_device: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_TTS_OUTPUT_DEVICE", ""))
    log_level: str = field(default_factory=lambda: os.environ.get("FUSION_SPEECH_LOG_LEVEL", "INFO").upper())
    log_dir: str = field(
        default_factory=lambda: os.path.expanduser(os.environ.get("FUSION_SPEECH_LOG_DIR", "~/.fusion-speech/logs"))
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
