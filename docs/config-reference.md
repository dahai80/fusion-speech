---
doc_id: fusion-speech/config-reference
product: fusion-speech
class: app
entry: service
doc_type: config-reference
title: { zh: "配置参考", en: "Configuration Reference" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - fusion_speech/config.py
  - README.md
  - start.sh
---

# Configuration Reference (配置参考)

All config via environment variables; no config file. Defaults live in
`fusion_speech/config.py`. The table in README "Configuration" documents every
variable; this page groups them and flags the hard contracts.

## Service and transport (服务与传输)

| Variable | Default | Purpose |
|---|---|---|
| `FUSION_SPEECH_PORT` | `11465` | HTTP port (localhost only; port-registry: 11465) |
| `FUSION_SOCKET_DIR` | `~/.fusion-speech/run` | UDS socket dir (mode 0o700) |
| `FUSION_SPEECH_WS_INGRESS` | `1` | enable `/stream/audio` WS PCM ingress |
| `FUSION_SPEECH_WS_TOKEN` | empty | bearer token for `/stream/audio` (`?token=` or `x-fusion-speech-token` header); empty = no auth (port is loopback-only) |
| `FUSION_SPEECH_WS_MAX_FRAME` | `65536` | max WS audio frame bytes; larger frames rejected |
| `FUSION_SPEECH_WS_KEEPALIVE_S` | `30.0` | `/stream/stt` idle keepalive interval |
| `FUSION_SPEECH_WS_SUBSCRIBER_QUEUE` | `64` | per-subscriber push-queue depth (drop-oldest) |

## Models and audio (模型与音频)

| Variable | Default | Purpose |
|---|---|---|
| `FUSION_SPEECH_STT_MODEL` | `openai/whisper-tiny` | STT model id at fusion-mlx; `whisper-large-v3` for zh |
| `FUSION_SPEECH_TTS_MODEL` | `kokoro` | TTS model id (fusion-mlx alias) |
| `FUSION_SPEECH_TTS_VOICE` | `af_heart` | TTS voice |
| `FUSION_SPEECH_LANGUAGE` | `en` | default STT language |
| `FUSION_SPEECH_SAMPLE_RATE` | `16000` | HARD CONTRACT: non-16000 fails startup (VAD/wake/openwakeword) |
| `FUSION_SPEECH_CHUNK_MS` | `300` | streaming chunk size |
| `FUSION_SPEECH_VAD_THRESHOLD` | `0.3` | VAD speech threshold |
| `FUSION_SPEECH_TTS_BUFFER_S` | `2.0` | TTS playback buffer cap (seconds) |
| `FUSION_SPEECH_TTS_OUTPUT_DEVICE` | system default | sounddevice output device |
| `FUSION_SPEECH_TTS_QUEUE_MAX` | `8` | per-session speak-frame queue depth |

## Wake word (唤醒)

| Variable | Default | Purpose |
|---|---|---|
| `FUSION_SPEECH_WAKE_WORD` | `hey_jarvis` | pretrained: alexa / hey_mycroft / hey_jarvis / hey_rhasspy / timer / weather |
| `FUSION_SPEECH_WAKE_THRESHOLD` | `0.5` | wake score threshold (0-1) |
| `FUSION_SPEECH_WAKE_AUTO_LISTEN_S` | `8.0` | auto-listen window after a wake |
| `FUSION_SPEECH_WAKE_DEBOUNCE_S` | `1.0` | cooldown between fires |
| `FUSION_SPEECH_WAKE_MODEL_DIR` | `~/.fusion-speech/models/wakeword` | ONNX model cache |
| `FUSION_SPEECH_WAKE_MODEL_URL` | GitHub releases v0.5.1 | download base URL (override for a mirror, e.g. offline install) |
| `FUSION_SPEECH_WAKE_CHUNK_SAMPLES` | `1280` | HARD CONTRACT: must be 1280 (openwakeword 80 ms) |

## Voice sessions (语音会话)

| Variable | Default | Purpose |
|---|---|---|
| `FUSION_SPEECH_BARGE_IN` | `1` | arm barge-in for full-duplex sessions |
| `FUSION_SPEECH_REPROMPT_CONFIDENCE` | `0.0` | ASR finals below this emit `speech.reprompt` (0 = off; empty transcripts always reprompt) |
| `FUSION_SPEECH_PROBE_INTERVAL_S` | `60.0` | background audio-lane re-probe interval |

## Cloud engines (Azure) (云引擎)

| Variable | Default | Purpose |
|---|---|---|
| `FUSION_SPEECH_AZURE_KEY` | empty | Azure Speech key (fusion-identity issued); both key+region required to register |
| `FUSION_SPEECH_AZURE_REGION` | empty | e.g. `eastus` |
| `FUSION_SPEECH_AZURE_LANGUAGE` | `en-US` | default STT language without LID |
| `FUSION_SPEECH_AZURE_STT_LANGUAGES` | empty | comma-separated LID candidates; enables dialect detection |
| `FUSION_SPEECH_AZURE_TTS_VOICE` | `en-US-AriaNeural` | default Azure TTS voice |

## Ops (运维)

| Variable | Default | Purpose |
|---|---|---|
| `FUSION_SPEECH_LOG_LEVEL` | `INFO` | log level |
| `FUSION_SPEECH_LOG_DIR` | `~/.fusion-speech/logs` | log dir (also `FUSION_SPEECH_LOG_DIR` for start.sh) |
| `FUSION_CLIENT_KEY` | (env only) | fusion-mlx bearer key via `fusion_core.config.client_key`; `FUSION_MLX_API_KEY` is an honored alias. Missing in dev = warning; missing with `FUSION_PROFILE=prod` = startup error |
| `FUSION_REGISTRY` | `~/fusion/architecture/port-registry.yaml` | port-registry path for the start.sh self-check |
| `FUSION_SKIP_REGISTRY_CHECK` | unset | `1` bypasses the registry self-check |
| `FUSION_VENV_DIR` | `<repo>/../.venv` | venv used by start.sh |

Hard contracts (硬约束): `FUSION_SPEECH_SAMPLE_RATE=16000` and
`FUSION_SPEECH_WAKE_CHUNK_SAMPLES=1280` fail startup when changed — they encode
cross-component DSP assumptions, not preferences.
