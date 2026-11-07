---
doc_id: fusion-speech/architecture
product: fusion-speech
class: app
entry: service
doc_type: architecture
title: { zh: "架构说明", en: "Architecture" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - fusion_speech
  - docs/architecture-design.md
  - README.md
---

# Architecture (架构说明)

Condensed map. Full module-by-module design: `docs/architecture-design.md`.

## Layer model (分层)

```
clients (fusion-studio, browsers, tests)
   │ UDS JSON-RPC 2.0              │ HTTP /health /metrics        │ WS /stream/stt, /stream/audio
   └───────────────┬───────────────┴──────────────┬──────────────┘
                   │        fusion_speech/server.py (asyncio FastAPI daemon)
                   │
   realtime layers: ring_buffer → vad → stt_stream → tts_playback → wake_word
                   │
   engines.py (ASREngine / TTSEngine registry: "local" + cloud)
                   │
   mlx_bridge.py ── HTTP ──► fusion-mlx :11434
                     /v1/audio/transcriptions (STT)
                     /v1/audio/speech (TTS PCM stream)
```

No model weights load in this process — fusion-speech is a realtime shell
around the fusion-mlx audio lane, probed at startup and re-probed every
`FUSION_SPEECH_PROBE_INTERVAL_S`. Design principle: delegate, don't duplicate;
local-first, HTTP only to 127.0.0.1.

本进程不加载任何模型权重：fusion-speech 是围绕 fusion-mlx 音频通道的实时壳层。

## Module map (模块划分)

| Module | Role |
|---|---|
| `server.py` | SpeechDaemon thin facade: lifecycle + orchestration |
| `components/` | leaf services split from the former god object: AudioBus (sole mic owner), AsrService, WakeService, TtsService, EventBus (fan-out + ASR-final registry) |
| `ipc/` | UDS transport, JSON-RPC dispatch, method handlers (peer-UID verified) |
| `ring_buffer.py` / `audio_source.py` | lock-free SPSC mic ring + sounddevice capture |
| `vad.py` | energy gate + Silero ONNX VAD, 16 kHz contract |
| `stt_stream.py` | chunked streaming ASR with R4 backpressure |
| `wake_word.py` | openwakeword ONNX, fixed 1280-sample frames |
| `tts_playback.py` | streaming playback, queue, cancel |
| `engines.py` / `cloud.py` / `cloud_azure.py` | pluggable ASR/TTS registry; Azure STT (SDK push-stream) / TTS (SSML REST) |
| `ws_ingress.py` | `/stream/audio` voice sessions |
| `session.py` | `VoiceSession` state machine (standalone-importable) |
| `mlx_bridge.py` | fusion-mlx HTTP client + audio-lane probe |
| `metrics.py` / `state.py` | zero-dependency Prometheus exposition; persistent daemon state |

Dependencies form a strict DAG: services hold no references to each other; the
facade is the only combiner.

服务之间互不引用，依赖构成严格有向无环图；SpeechDaemon 仅做生命周期与编排。

## Session state machine (会话状态机)

`listening → speaking → interrupted → listening`, with a `speech.reprompt`
branch on low-confidence/empty ASR finals. Barge-in: server VAD stays live
during full-duplex playback; VAD onset drops pending TTS audio, cancels
in-flight synthesis, and re-enters listening while the interrupting utterance
transcribes. TTS `speak` frames arriving during synthesis queue FIFO
(`FUSION_SPEECH_TTS_QUEUE_MAX`); overflow drops the oldest frame.

## Failure modes (故障模式)

| Failure | Behavior |
|---|---|
| fusion-mlx down | startup probe reports the lane down (non-fatal); `/health` + `speech.status` flag it; RPC transcribe/synthesize error; `speech.reprobe` or interval probe recovers |
| Sample-rate/frame mismatch | startup fails fast (16 kHz / 1280 hard contracts) |
| Cloud creds unset | cloud engines unregistered; local-only daemon (fail-soft) |
| Cloud error | stable slug + `retryable`; network/429 retries at 0.5/1/2 s; auth/bad-request never requeue |
| Wake model missing | downloaded from `FUSION_SPEECH_WAKE_MODEL_URL`; wake stays disarmed |
| TTS queue overflow | oldest frame dropped, `speech.queue_overflow` emitted |
| Non-loopback UDS peer | connection rejected (UID check, fail-closed) |
| Shutdown | SIGTERM → `speech.shutdown` broadcast → bounded 10s drain → socket unlink; SIGKILL after 15s grace |

## Cross-repo dependencies (跨仓依赖)

- fusion-mlx (:11434) — owns STT/TTS inference (`/v1/audio/*`). Upstream gaps
  go through issues first (e.g. dahai80/fusion-mlx#1339 for `ref_audio` on
  `/v1/audio/speech`).
- fusion-core — `mlx_client.FusionMLXClient`, FastAPI app factory, config
  resolution, logging.
- fusion-studio — Swift desktop client over UDS (M5); upstream issue
  dahai80/fusion-studio#337.
- port-registry — claims 11465; start.sh self-checks it.
