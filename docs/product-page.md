---
doc_id: fusion-speech/product-page
product: fusion-speech
class: app
entry: service
doc_type: product-page
title: { zh: "Fusion-Speech 产品页", en: "Fusion-Speech Product Page" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - pyproject.toml
  - start.sh
  - fusion_speech
---

# Fusion-Speech Product Page (产品页)

## What it is (产品定位)

fusion-speech is a local-first, low-latency speech daemon for the Fusion
ecosystem: streaming STT, TTS playback, hotword wake, and full-duplex voice
sessions. It runs on Apple Silicon and delegates all model inference to
fusion-mlx over HTTP, adding only the realtime layers fusion-mlx lacks
(mic capture, VAD, wake, streaming ASR/TTS).

fusion-speech 是 Fusion 生态的本地低延迟语音守护进程：流式识别、语音合成播放、
热词唤醒与全双工语音会话。模型推理全部经 HTTP 委托给 fusion-mlx，本进程只补齐
fusion-mlx 缺失的实时层（麦克风采集、VAD、唤醒、流式 ASR/TTS）。

## Grid position (契约网格)

| Key | Value | Evidence (证据) |
|---|---|---|
| class | app | Domain service in the fusion monorepo |
| entry | service | asyncio FastAPI daemon on `127.0.0.1:11465` plus UDS JSON-RPC (`~/.fusion-speech/run/fusion-speech.sock`); no UI of its own — consumed by fusion-studio and other clients |

## What consumers no longer have to do (消费方不再需要做什么)

- No more loading STT/TTS weights in every business app: one daemon serves all
  of them over UDS/HTTP while fusion-mlx owns the models. Constraint: the
  daemon is localhost-only by design; remote clients need their own daemon or
  the WS ingress path.
- No more writing mic/VAD/wake plumbing: ring-buffer capture, VAD segmentation,
  and openwakeword ONNX wake are built in. Constraint: sample rate is
  hard-locked at 16 kHz mono (VAD/wake contract) and wake frames at 1280
  samples — other rates fail startup rather than silently degrade.
- No more fork-per-cloud-engine: pluggable `ASREngine`/`TTSEngine` registry
  (first cloud engine: Azure AI Speech). Constraint: credentials live in the
  daemon env (fusion-identity issued), not in business apps.

- 业务应用不再各自加载 STT/TTS 权重：一个守护进程经 UDS/HTTP 服务所有消费方。
  约束：默认仅监听本机回环。
- 不再自写麦克风/VAD/唤醒管线：内置采集、VAD 分段与 openwakeword 唤醒。
  约束：采样率锁死 16 kHz 单声道，唤醒帧 1280 样本，配置不符直接启动失败。
- 云引擎不再各自 fork：可插拔引擎注册表（首个为 Azure AI Speech）。
  约束：凭据保存在守护进程环境变量中，不下发到业务应用。

## Key capabilities (核心能力)

- RPC: 13 JSON-RPC 2.0 methods over UDS (peer-UID verified, fail-closed) —
  transcribe, synthesize, speak, wake, hotwords, listening control, reprobe.
- Realtime push: WebSocket `/stream/stt` emits `speech.asr` / `speech.wake` /
  `speech.tts` / keepalive frames with distinct event types.
- Voice sessions: `/stream/audio` WS PCM ingress — full-duplex mode keeps VAD
  live during TTS playback, fires `speech.barge_in` on interruption, queues
  TTS sentences FIFO with `speech.queue_overflow` signaling.
- Metrics: Prometheus text at `GET /metrics` (hand-written, zero-dependency)
  covering RPC latency, ASR segments, wake fires, TTS queue, cloud errors.
- Cloud engines: Azure STT (word timestamps + LID dialect detection) and Azure
  TTS (SSML REST, 16 kHz mono, no resampling), with stable error slugs and
  retry classification.

## Interfaces (入口)

| Interface | How | Source |
|---|---|---|
| UDS JSON-RPC | `~/.fusion-speech/run/fusion-speech.sock`, newline-delimited JSON-RPC 2.0 | `fusion_speech/ipc/`, `server.py` |
| HTTP | `./start.sh start` → `http://127.0.0.1:11465` (`/health`, `/metrics`) | `start.sh`, `fusion_speech/server.py` |
| WebSocket | `/stream/stt` (events), `/stream/audio` (voice sessions) | `fusion_speech/ws_ingress.py` |

## Non-goals (边界)

- Not an inference engine: no model weights in this process; fusion-mlx
  (:11434) must be reachable for STT/TTS.
- Not a network service: binds localhost only; the UDS peer-UID check is the
  trust boundary.
- 非推理引擎（权重在 fusion-mlx）；非网络服务（仅回环绑定，UDS 对端 UID 校验为信任边界）。

## Status (状态)

v0.1.6. Milestones M1-M7 landed; see README "Status" for the phase record and
`docs/ARCHITECTURE.md` for the internal design.
