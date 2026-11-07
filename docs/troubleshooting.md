---
doc_id: fusion-speech/troubleshooting
product: fusion-speech
class: app
entry: service
doc_type: troubleshooting
title: { zh: "故障排查", en: "Troubleshooting" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - start.sh
  - README.md
---

# Troubleshooting (故障排查)

## Daemon won't start (守护进程启动失败)

1. Boot log: `~/.fusion-speech/logs/boot.log` (start.sh), then runtime logs in
   the same dir — `./start.sh log -f`.
2. `sample_rate must be 16000` / `wake chunk must be 1280`: these are hard
   contracts — revert `FUSION_SPEECH_SAMPLE_RATE` /
   `FUSION_SPEECH_WAKE_CHUNK_SAMPLES` to defaults.
3. Port in use: `lsof -i :11465`; start.sh also refuses to start if the port
   is not registered in `architecture/port-registry.yaml` — in a sandbox set
   `FUSION_SKIP_REGISTRY_CHECK=1` or point `FUSION_REGISTRY` at a custom
   registry file.
4. Stale PID file `.fusion-speech.pid` after a crash: remove it and start.
5. `fusion_core` import error: `pip install -e fusion-core` from the monorepo
   root (shared venv must be active).

## STT/TTS calls fail (识别/合成失败)

1. `curl http://127.0.0.1:11465/health` — if the audio lane is not ready,
   fusion-mlx is down or has no audio models: start it with
   `~/fusion/fusion-mlx/start.sh start`, then `speech.reprobe` or restart.
2. Empty transcription: the default `whisper-tiny` is English-weak on long
   audio; for Chinese set `FUSION_SPEECH_STT_MODEL=openai/whisper-large-v3`
   and `FUSION_SPEECH_LANGUAGE=zh` on the daemon.
3. Auth error from fusion-mlx: set `FUSION_CLIENT_KEY` (or legacy
   `FUSION_MLX_API_KEY`); with `FUSION_PROFILE=prod` a missing key is a
   startup error, not a warning.
4. TTS produces no sound: check `FUSION_SPEECH_TTS_OUTPUT_DEVICE` (system
   default if unset) and macOS microphone/audio permissions for the terminal.

## Wake word not firing (唤醒不触发)

- The `[audio]` extra must be installed: `pip install -e ".[audio]"`.
- ONNX model missing: downloaded on first use from
  `FUSION_SPEECH_WAKE_MODEL_URL`; on an offline machine pre-populate
  `FUSION_SPEECH_WAKE_MODEL_DIR` (e.g. via the hf-mirror equivalent of the
  release asset).
- Threshold too high: lower `FUSION_SPEECH_WAKE_THRESHOLD` (default 0.5).
- Spurious wakes on silence: the real-load test asserts silence does not fire;
  if it happens at runtime, raise the threshold and check `wake_armed` gauge.

## WebSocket issues (WebSocket 问题)

- `/stream/audio` rejected with 401-ish failure: a token is set
  (`FUSION_SPEECH_WS_TOKEN`) — pass `?token=` or the
  `x-fusion-speech-token` header.
- Frames dropped immediately: frame exceeds `FUSION_SPEECH_WS_MAX_FRAME`
  (65536 bytes) or PCM is not 16 kHz mono — resample client-side.
- No keepalive frames: they arrive only after `FUSION_SPEECH_WS_KEEPALIVE_S`
  idle and as `{"event": "speech.keepalive"}` — do not treat them as audio
  events.
- `speech.queue_overflow`: the client speaks faster than playback; depth is
  `FUSION_SPEECH_TTS_QUEUE_MAX`. Slow the sentence feed or raise the depth.

## Cloud engine (Azure) issues (云引擎问题)

- `speech.engines` does not list azure: both `FUSION_SPEECH_AZURE_KEY` and
  `_REGION` must be set before startup (fail-soft — unset means local-only).
- Error slugs: `azure_auth` (check key/identity), `azure_rate_limited`
  (retries automatically), `azure_unreachable`/`azure_timeout` (network),
  `azure_sdk_missing` (`pip install -e ".[azure]"`).
- Counters `cloud_errors_total{engine,kind,reason}` on `/metrics` give the
  per-reason breakdown.

## Metrics and status (指标与状态)

- `GET /metrics` — full Prometheus exposition; `speech.status` (UDS) returns
  the summary (`uptime`, `rpc_total`, `errors_total`, `mlx_audio_lane`).
- `asr_segments_dropped_total` climbing: consumer is slow — raise
  `FUSION_SPEECH_WS_SUBSCRIBER_QUEUE` or drain `/stream/stt` faster.

## Tests (测试)

- Offline suite: `pytest tests/ -q` (no model load). The 7 `real_load` skips
  are opt-in: `FUSION_SPEECH_REAL_LOAD=1` with fusion-mlx up.
