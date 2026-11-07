---
doc_id: fusion-speech/readme
product: fusion-speech
class: app
entry: service
doc_type: readme
title: { zh: "fusion-speech", en: "fusion-speech" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
---

# fusion-speech

> [English](README.md) | [中文](README_CN.md)

> Local-first low-latency speech daemon for the Fusion ecosystem.

Local-first speech service: speech-to-text, text-to-speech, hotword wake, realtime
streaming. Runs on Apple Silicon. Delegates model inference to fusion-mlx via HTTP,
adds only the realtime layers fusion-mlx lacks (mic capture, wake, streaming ASR,
streaming TTS playback).

## Overview

- **Daemon**: asyncio FastAPI service on `127.0.0.1:11465` (HTTP `/health`) plus
  Unix Domain Socket for JSON-RPC 2.0 IPC (`~/.fusion-speech/run/fusion-speech.sock`).
- **Delegation**: STT/TTS model inference runs in fusion-mlx (`localhost:11434`).
  fusion-speech posts audio to `/v1/audio/transcriptions` and reads PCM streams
  from `/v1/audio/speech`. No model weights loaded in this process.
- **Realtime layers (M2-M4)**: ring-buffer mic capture, VAD, chunked streaming ASR,
  streaming TTS playback, hotword wake.
- **Voice sessions (M7)**: WebSocket PCM ingress (`/stream/audio`) as an
  alternative to the local mic — full-duplex mode keeps VAD live during TTS
  playback, fires `speech.barge_in` on user interruption, and streams TTS PCM
  back over the same socket. Pluggable `ASREngine`/`TTSEngine` adapter layer
  (cloud engines register by name) and a reusable `VoiceSession` state machine.
- **IPC**: newline-delimited JSON-RPC 2.0 over UDS, with peer-UID verification
  (fail-closed) so only same-user processes can call the daemon.

## Architecture

```
┌──────────────┐   JSON-RPC/UDS    ┌───────────────┐   HTTP   ┌────────────┐
│ fusion-studio│ ────────────────▶ │ fusion-speech │ ───────▶ │ fusion-mlx │
│  (Swift UI)  │ ◀──────────────── │   (daemon)    │ ◀─────── │ (models)   │
└──────────────┘  notifications    └───────────────┘  PCM/    └────────────┘
                                     mic│VAD│wake     audio
                                        │
                                  realtime layers (M2-M4)
```

Inference boundary: fusion-mlx owns the weights and the OpenAI-compatible
`/v1/audio/*` endpoints. fusion-speech owns everything between the microphone and
those endpoints. See `docs/ARCHITECTURE.md` for the full layer breakdown and the
fusion-* integration map.

## Quick Start

```bash
# from the fusion monorepo root
source .venv/bin/activate
pip install -e fusion-core
pip install -e ".[test]"        # from fusion-speech/

# start the inference engine fusion-speech delegates to
~/fusion/fusion-mlx/start.sh start

# start the speech daemon
./start.sh start
./start.sh status
./start.sh doctor

# logs
./start.sh log -f

# stop
./start.sh stop
```

Smoke test the RPC lane:

```bash
python3 -c "
import asyncio, json, os
async def main():
    r, w = await asyncio.open_unix_connection(os.path.expanduser('~/.fusion-speech/run/fusion-speech.sock'))
    w.write((json.dumps({'jsonrpc':'2.0','id':1,'method':'speech.ping'}) + '\n').encode())
    await w.drain()
    print(json.loads(await r.readline()))
    w.close()
asyncio.run(main())
"
```

## Configuration

All config via environment variables (no config file). Defaults live in
`fusion_speech/config.py`.

| Env var | Default | Purpose |
|---|---|---|
| `FUSION_SPEECH_PORT` | `11465` | HTTP port (localhost only) |
| `FUSION_SOCKET_DIR` | `~/.fusion-speech/run` | UDS socket directory (0o700) |
| `FUSION_SPEECH_STT_MODEL` | `openai/whisper-tiny` | STT model id (use `openai/whisper-large-v3` for multilingual incl. zh) |
| `FUSION_SPEECH_TTS_MODEL` | `kokoro` | TTS model id (fusion-mlx alias) |
| `FUSION_SPEECH_TTS_VOICE` | `af_heart` | TTS voice |
| `FUSION_SPEECH_LANGUAGE` | `en` | default STT language (set `zh` with `openai/whisper-large-v3`) |
| `FUSION_SPEECH_SAMPLE_RATE` | `16000` | capture sample rate — hard-locked 16000 (VAD/wake/openwakeword contract); non-16000 fails startup |
| `FUSION_SPEECH_VAD_THRESHOLD` | `0.3` | VAD speech threshold |
| `FUSION_SPEECH_CHUNK_MS` | `300` | streaming chunk size |
| `FUSION_SPEECH_WAKE_WORD` | `hey_jarvis` | default wake word (pretrained: alexa/hey_mycroft/hey_jarvis/hey_rhasspy/timer/weather) |
| `FUSION_SPEECH_WAKE_THRESHOLD` | `0.5` | wake score threshold (0-1) |
| `FUSION_SPEECH_WAKE_AUTO_LISTEN_S` | `8.0` | auto-listen window after a wake |
| `FUSION_SPEECH_WAKE_DEBOUNCE_S` | `1.0` | cooldown between wake fires |
| `FUSION_SPEECH_WAKE_MODEL_DIR` | `~/.fusion-speech/models/wakeword` | wake ONNX model cache dir |
| `FUSION_SPEECH_WAKE_MODEL_URL` | GitHub releases v0.5.1 | wake model download base URL (override for a mirror) |
| `FUSION_SPEECH_TTS_BUFFER_S` | `2.0` | TTS playback buffer cap (seconds) |
| `FUSION_SPEECH_TTS_OUTPUT_DEVICE` | (system default) | sounddevice output device |
| `FUSION_SPEECH_WS_KEEPALIVE_S` | `30.0` | `/stream/stt` WebSocket idle keepalive interval (sec) |
| `FUSION_SPEECH_WS_SUBSCRIBER_QUEUE` | `64` | per-subscriber push-queue depth (drop-oldest on full) |
| `FUSION_SPEECH_WAKE_CHUNK_SAMPLES` | `1280` | fixed wake frame size — must be 1280 (openwakeword 80ms contract); non-1280 fails startup |
| `FUSION_SPEECH_PROBE_INTERVAL_S` | `60.0` | background audio-lane re-probe interval (sec) |
| `FUSION_SPEECH_WS_INGRESS` | `1` | enable the `/stream/audio` WS PCM ingress endpoint |
| `FUSION_SPEECH_WS_TOKEN` | (empty) | optional bearer token for `/stream/audio` (`?token=` or `x-fusion-speech-token` header); empty = no auth (port is localhost-only) |
| `FUSION_SPEECH_WS_MAX_FRAME` | `65536` | max WS audio frame size in bytes (larger frames rejected) |
| `FUSION_SPEECH_BARGE_IN` | `1` | default barge-in armed for full-duplex sessions |
| `FUSION_SPEECH_REPROMPT_CONFIDENCE` | `0.0` | ASR finals with confidence below this emit `speech.reprompt` (0 = disabled; empty transcripts always reprompt) |
| `FUSION_SPEECH_TTS_QUEUE_MAX` | `8` | per-session speak-frame queue depth; overflow drops the oldest frame and emits `speech.queue_overflow` |
| `FUSION_SPEECH_AZURE_KEY` | (empty) | Azure Speech subscription key (fusion-identity issued; unset = cloud engines stay unregistered) |
| `FUSION_SPEECH_AZURE_REGION` | (empty) | Azure region (e.g. `eastus`) |
| `FUSION_SPEECH_AZURE_LANGUAGE` | `en-US` | default Azure STT language when no LID candidates |
| `FUSION_SPEECH_AZURE_STT_LANGUAGES` | (empty) | comma-separated LID candidate list (e.g. `ar-EG,ar-SA,en-US`); enables dialect detection |
| `FUSION_SPEECH_AZURE_TTS_VOICE` | `en-US-AriaNeural` | default Azure TTS voice |
| `FUSION_SPEECH_LOG_LEVEL` | `INFO` | log level |
| `FUSION_SPEECH_LOG_DIR` | `~/.fusion-speech/logs` | log directory |
| `FUSION_CLIENT_KEY` | (env only) | fusion-mlx bearer key via `fusion_core.config.client_key` (issue #6); legacy `FUSION_MLX_API_KEY` still honored as alias. Dev missing = empty + warning, `FUSION_PROFILE=prod` missing = startup error |
| `FUSION_REGISTRY` | `~/fusion/architecture/port-registry.yaml` | port-registry location for the start.sh self-check (issue #7) |
| `FUSION_SKIP_REGISTRY_CHECK` | (unset) | `1` bypasses the registry self-check (unregistered port would otherwise refuse to start) |

## RPC API

JSON-RPC 2.0 over UDS, newline-delimited. Peer UID verified (fail-closed).

| Method | Params | Returns |
|---|---|---|
| `speech.ping` | `{}` | `{pong: true, timestamp}` |
| `speech.status` | `{}` | running, socket_path, port, uptime, mlx_audio_lane, stt_model, tts_model, listening, hotwords |
| `speech.transcribe` | `{audio: base64, language?, model?, filename?, engine?}` | `{text, language, duration, segments?, engine?, words?, dialect?, dialect_detected?}` |
| `speech.synthesize` | `{text, voice?, model?, engine?}` | `{audio: base64 wav, format: "wav", size}` |
| `speech.list_models` | `{}` | `{total, audio_models}` |
| `speech.engines` | `{}` | `{asr: [...], tts: [...]}` — registered engine names (local + cloud) |
| `speech.start_listening` | `{}` | `{listening: true}` — arms mic capture + streaming ASR |
| `speech.stop_listening` | `{}` | `{listening: false, segments: <int>}` |
| `speech.set_hotwords` | `{hotwords: [...]}` | `{hotwords: [...]}` — register wake words (pretrained names only; unknown dropped) |
| `speech.wake` | `{arm: true\|false}` | `{armed: bool, words: [...]}` — arm/disarm wake detection |
| `speech.speak` | `{text, voice?, model?}` | `{played, chars, sample_rate}` — streams TTS to the speaker (pauses mic) |
| `speech.stop_tts` | `{}` | `{stopped}` — cancel active playback |
| `speech.reprobe` | `{}` | `{audio_lane}` — re-probe the fusion-mlx audio lane on demand (refreshes `list_models` cache) |

M2 realtime push: connect WebSocket `/stream/stt` to receive `speech.asr`
notification frames (`listening` → `speaking` → `transcribing` → `final`, or
`error`). M3 adds `speech.wake` notifications on the same socket (then
auto-starts ASR for the listen window). M4 adds `speech.tts` progress events
on the same socket. Keepalive is its own event type — `{"event":
"speech.keepalive"}` every `FUSION_SPEECH_WS_KEEPALIVE_S` idle, distinct from
`speech.asr` so clients can tell keepalive apart from real audio events.

### Wake (M3)

`speech.wake {arm: true}` arms always-on wake-word detection (openwakeword, ONNX).
On a wake the daemon emits a `speech.wake` notification to `/stream/stt`
subscribers, pauses wake, auto-starts streaming ASR for
`FUSION_SPEECH_WAKE_AUTO_LISTEN_S` seconds (reusing the same mic source), then
stops ASR and resumes wake — the classic assistant flow. Wake is paused during
ASR so your own speech does not self-trigger. Fail-soft: if openwakeword or the
ONNX models are unavailable, `speech.wake` returns `{armed: false}` and the
daemon keeps running.

Wake models (the wake word `.onnx` + `melspectrogram.onnx` + `embedding_model.onnx`)
download on first arm to `FUSION_SPEECH_WAKE_MODEL_DIR` from
`FUSION_SPEECH_WAKE_MODEL_URL` (default: openWakeWord GitHub releases v0.5.1).
Pretrained words: `alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy`, `timer`,
`weather`.

Handler errors map to `-32000` as `"<ExceptionType> [(status N)]: <message>"`
(e.g. `AudioBridgeError (status 404): synthesize failed: Model ... not found`).

Model resolution: when no `model` param is given, the daemon uses the configured
default (`FUSION_SPEECH_TTS_MODEL` / `FUSION_SPEECH_STT_MODEL`); if that id is not
in the probed fusion-mlx model set, it falls back to the first probed audio model
of that modality (logged at WARNING). This makes `speech.synthesize {}` work
out-of-box even when fusion-mlx's loaded model differs from the configured default.

Error codes follow JSON-RPC 2.0: `-32700` parse error, `-32600` invalid request,
`-32601` method not found, `-32603` internal error, `-32000` handler error.

## Voice Sessions (WS PCM Ingress)

`/stream/audio` is an alternative audio transport to the local mic: a browser
(or any client) streams 16 kHz mono PCM over a WebSocket and gets ASR, TTS, and
session events back on the same socket. The ASR path is the exact mic pipeline
(ring buffer → VAD → segment → transcribe), so WS-fed audio behaves identically.

Protocol (one connection = one voice session, one ASR stream at a time):

| Client frame | Meaning |
|---|---|
| `{"type":"start", "mode":"full_duplex"\|"half_duplex", "encoding":"int16"\|"float32", "tts_sink":"ws"\|"local", "asr_engine":"local", "tts_engine":"local", "barge_in":bool, "reprompt_min_confidence":float, "session":"id"}` | start a session (answers `speech.session.info`) |
| binary frame | 16 kHz mono PCM (`int16` LE by default, `float32` LE via `encoding`) |
| `{"type":"speak","text","voice?","model?","ref_audio?","ref_text?"}` | synthesize + play (streams binary PCM back when `tts_sink=ws`) |
| `{"type":"tts_cancel"}` | cancel playback immediately (drop queued audio) |
| `{"type":"stop"}` | end the session (or just disconnect) |

Server frames: JSON events (same `speech.*` contract as `/stream/stt`, plus
`speech.session` state transitions, `speech.barge_in`, `speech.reprompt`,
`speech.queue_overflow`, `speech.session.error`) and binary int16 PCM while TTS
is streaming to the WS.

TTS queueing: a streaming agent feeds `speak` frames sentence by sentence,
milliseconds apart — a `speak` arriving while synthesis is active is queued
(FIFO, depth `FUSION_SPEECH_TTS_QUEUE_MAX`), not rejected, and plays in order
the moment the previous sentence finishes (no client-side serialization gate
needed). Overflow drops the oldest queued sentence and emits
`speech.queue_overflow` (`{queue_depth}`) so the client can decide to skip.
`tts_cancel` and barge-in clear the queue along with the in-flight synthesis —
the interrupted answer's remaining sentences are stale and never play.

Full-duplex + barge-in (the point of the mode): the client plays the TTS audio
and runs AEC, so server-side VAD stays live during playback. When the user
speaks over the TTS, VAD onset fires `speech.barge_in`, pending audio is
dropped, playback is cancelled (server-side cancel latency is well under the
100 ms budget), and the session returns to `listening` while the interrupting
utterance is transcribed. Half-duplex keeps the mic-era behavior (ASR paused
during playback).

State machine (importable standalone from `fusion_speech.session.VoiceSession`):
`listening → speaking → interrupted → listening`, with a low-confidence /
empty-transcript `speech.reprompt` branch.

Pluggable engines (`fusion_speech.engines`): `ASREngine`
(`start_stream/feed/finish/cancel`) and `TTSEngine` (`synthesize` WAV stream +
`cancel`). `"local"` is the fusion-mlx delegation (including `ref_audio`/
`ref_text` cloning params — forwarded only when set; fusion-mlx does not accept
them on `/v1/audio/speech` yet, upstream issue dahai80/fusion-mlx#1339). Cloud
engines (e.g. iFlytek WS) register via `register_asr_engine` /
`register_tts_engine` and are selected per session by name — no fork needed.

## Cloud Engines (issue #4)

Pluggable cloud STT/TTS on the same engine registry, first implementation:
**Azure AI Speech** (`pip install -e ".[azure]"`). Credentials never live in
business apps — the daemon holds them (env `FUSION_SPEECH_AZURE_KEY` /
`_REGION`, issued via fusion-identity) and business clients select the engine
by name (`engine: "azure"` on `speech.transcribe`/`speech.synthesize`, or
`asr_engine`/`tts_engine` on a voice session `start`). Engines register at
startup only when both env vars are set (fail-soft: unset = local-only daemon);
a `credentials` provider hook is available for embedding-side key rotation.

- **Azure STT** — SDK push-stream + one-shot recognize (word alignment and
  language ID are SDK-only features; the short-audio REST API returns neither).
  Returns word-level timestamps (`words`: word/offset_ms/duration_ms/confidence,
  100 ns ticks converted) and dialect detection (`dialect`, e.g. `ar-EG`) when
  LID candidates are configured via `FUSION_SPEECH_AZURE_STT_LANGUAGES` — the
  ai-speech-exam accent flag that raises human-review priority. SDK calls run
  in a worker thread; retry rebuilds the recognizer from the retained PCM.
- **Azure TTS** — REST SSML POST, `riff-16khz-16bit-mono-pcm` output (the
  daemon 16 kHz mono contract, no resampling), streamed chunk by chunk. Retry
  covers the connection open only — mid-stream replay would duplicate playback.
- **Failure reasons for the audio job pipeline** — errors carry a stable slug
  (`azure_auth`, `azure_unreachable`, `azure_rate_limited`, `azure_service`,
  `azure_timeout`, `azure_sdk_missing`, `azure_invalid_response`,
  `azure_not_configured`, `invalid_audio`, `azure_bad_request`) embedded in the
  JSON-RPC error message plus a retryable flag; auth/bad-request never requeue,
  network/429 retry with 0.5/1/2 s backoff. Counters:
  `cloud_requests_total{engine,kind}` / `cloud_errors_total{engine,kind,reason}`.
- `speech.engines` lists the registered engines. Real-load gate:
  `tests/test_cloud_real_load.py` (opt-in, skips without live Azure creds).

## Metrics

`GET /metrics` on the HTTP port (`127.0.0.1:11465`) returns Prometheus text
exposition (zero-dependency, hand-written — no `prometheus_client` added). Every
RPC handler is timed and counted; the listener, wake engine, and TTS player
report state gauges; the ASR/TTS bridges report throughput.

| Metric | Type | Labels | What it measures |
|---|---|---|---|
| `rpc_requests_total` | Counter | `method` | RPC calls per method |
| `rpc_errors_total` | Counter | `method` | RPC handler errors per method |
| `rpc_duration_seconds` | Histogram | `method` | RPC handler latency (seconds) |
| `asr_segments_total` | Counter | — | ASR segments closed (VAD segment-close) |
| `asr_transcribe_seconds` | Histogram | — | per-segment transcribe latency |
| `asr_segments_dropped_total` | Counter | — | segments dropped by R4 backpressure |
| `wake_fires_total` | Counter | — | wake-word fires |
| `tts_speak_total` | Counter | — | `speech.speak` invocations |
| `subscribers` | Gauge | — | current WS/event subscribers |
| `listening` | Gauge | — | 1 when ASR listener active, else 0 |
| `wake_armed` | Gauge | — | 1 when wake engine armed, else 0 |
| `speaking` | Gauge | — | 1 when TTS player active, else 0 |
| `sessions` | Gauge | — | current WS voice sessions |
| `barge_in_total` | Counter | — | barge-in interruptions (VAD onset during playback) |
| `tts_queue_overflow_total` | Counter | — | speak frames dropped from a full per-session TTS queue |
| `cloud_requests_total` | Counter | `engine`, `kind` | cloud engine calls (kind = stt / tts) |
| `cloud_errors_total` | Counter | `engine`, `kind`, `reason` | cloud engine errors by reason slug |

`speech.status` also returns a metrics summary (`uptime`, `rpc_total`,
`errors_total`). Scrape with:

```bash
curl http://127.0.0.1:11465/metrics
```

## Integration

- **fusion-mlx** (hard dependency): owns `/v1/audio/transcriptions` (STT) and
  `/v1/audio/speech` (TTS). fusion-speech probes the audio lane at startup and
  reports readiness in `/health` and `speech.status`.
- **fusion-core** (in-tree shared base): `mlx_client.FusionMLXClient` for HTTP,
  `http.app.create_app` for the FastAPI factory, `config` for base URL / API key
  resolution, `logging` for log setup.
- **fusion-studio** (Swift desktop, M5): calls the daemon over UDS JSON-RPC;
  registers a SpeechBridge + mic permission. Upstream change — issue
  dahai80/fusion-studio#337 filed; Swift PR to follow per the upstream-first
  workflow.
- **port-registry**: claims `11465` in `architecture/port-registry.yaml` (M5).

Only this project's directory is modified. Upstream fusion-mlx / fusion-studio /
port-registry changes go through issue first, then PR.

For scenario-based integration recipes (Python/Swift UDS clients for transcribe,
synthesize, wake, streaming ASR/TTS), see
[`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md). For developer setup,
testing, style, and the upstream issue→PR workflow, see
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Testing

Unit + adversarial gate suite run offline (no model load, no live daemon):

```bash
cd /Users/dahai/fusion && source .venv/bin/activate
cd fusion-speech
FUSION_MLX_API_KEY=<key> pytest tests/ -q              # 229 passed, 7 skipped
ruff check fusion_speech tests                         # clean
```

The 7 skipped tests carry the `real_load` marker — they start the real
fusion-speech daemon (real port 11465 + real socket) backed by a real
fusion-mlx on 11434, so they are opt-in. Set `FUSION_SPEECH_REAL_LOAD=1`
(and have fusion-mlx up) to run them:

```bash
~/fusion/fusion-mlx/start.sh start
FUSION_SPEECH_REAL_LOAD=1 FUSION_MLX_API_KEY=<your-key> \
  pytest tests/test_m6_real_load_integration.py -m real_load -v
```

The real-load suite is a 6-step full-chain gate:

1. `speech.ping`/`speech.status` — liveness + audio-lane probed.
2. `speech.transcribe` — real STT over fusion-mlx against the bundled
   English fixture (whole-file WAV upload). The upstream fusion-mlx STT 500
   (issue #668, `mlx-audio` Whisper `non_speech_tokens` `IndexError`) is
   fixed upstream — this step now asserts real transcription text.
3. `speech.synthesize` — real TTS, asserts non-empty WAV (`RIFF` + PCM body).
4. `speech.speak` — streaming playback, polls `speech.status` until settled.
5. `speech.wake` — arm/disarm over real openwakeword ONNX; headless silence
   must not fire a spurious wake.
6. `/stream/audio` voice session — full-duplex WS session on the real daemon:
   silence must not open a segment, WS `speak` streams real Kokoro PCM back
   as binary frames, mid-stream `tts_cancel` (barge-in path) stops audio,
   stop releases the listener slot.

Test fixtures live in `tests/fixtures/` (kept as final artifacts). The UDS
client helper is `tests/_uds_client.py`; the `real_daemon` session fixture
(spawn via `start.sh`, real socket/port) is in `tests/conftest.py`.

Developer setup, test commands, lint, and code style are documented in
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Status

Milestone progress:

- **M0** — engineering skeleton (this release): daemon, UDS JSON-RPC, fusion-mlx
  audio-lane probe, `/health`, `start.sh` lifecycle, unit tests. **done**
- **M1** — fusion-mlx delegation bridge (transcribe/synthesize via HTTP). **done**
  - TTS verified real-load: `speech.synthesize` over UDS JSON-RPC → fusion-mlx
    Qwen3-TTS, 388 KB WAV in 5.3 s; out-of-box with probe-driven model fallback.
  - STT verified real-load: `speech.transcribe` over UDS JSON-RPC → fusion-mlx
    `openai/whisper-tiny`, `assistant_bank_en.wav` → "Welcome to First National
    Bank! How may I assist you today?" in 1.5 s.
  - Required an upstream fusion-mlx discovery fix (issue #649): `_is_hf_cache_mlx_compatible`
    rejected HF-format audio repos (`openai/whisper-tiny` ships HF `model.safetensors`
    + full tokenizer, no MLX metadata, repo not under `mlx-community/`). `mlx_audio`
    loads HF-format audio weights natively, so the fix adds an audio-model branch.
    Note: `mlx-community/whisper-*-mlx*` repos ship NO tokenizer files → unusable
    for `mlx_audio.stt` (empty vocab → `IndexError`); use `openai/whisper-*`.
    Tracked: issue #649 + PR on fusion-mlx, per the upstream-first workflow.
- **M2** — realtime mic capture + streaming ASR. **done**
  - Ring buffer: lock-free SPSC numpy ring, drop-oldest on overrun (capture never
    blocks). AudioSource wraps `sounddevice` 16 kHz mono (lazy import, headless-safe).
  - VAD: two-tier energy RMS gate + Silero ONNX (`silero_vad_16k_op15.onnx`). The
    64-sample rolling context prepend is required — without it probs collapse to
    ~0.001 on all audio. Fail-soft to energy-only if the model is missing.
  - StreamingListener: drain mirrors samples, VAD segments on speech onset /
    silence-after-speech, segment PCM → 16-bit WAV → `transcribe_stream` →
    `speech.asr` `final` notification. Sub-0.1s segments skipped.
  - Realtime push: WebSocket `/stream/stt` with per-subscriber queue fan-out
    (drop-oldest on full); `speech.start_listening`/`stop_listening` RPC.
  - Verified: 55 tests green incl. real-load Silero on a speech WAV fixture and
    an end-to-end start → capture → transcribe → final-to-subscriber integration test.
- **M3** — hotword wake. **done**
  - `wake_word.py`: openwakeword ONNX keyword spotting (`hey_jarvis` default).
    Reuses the M2 mic ring; drain reads 1280-sample (80ms) frames →
    `openwakeword.Model.predict`. Score ≥ threshold fires `on_wake` once, then a
    debounce cooldown blocks repeats. `pause()`/`resume()` suspend detection
    during active ASR (no SPSC read race — paused drain stops reading the ring).
    Fail-soft: missing openwakeword/models → `enabled=False`, daemon unharmed.
  - Model cache: wake word + `melspectrogram.onnx` + `embedding_model.onnx`
    download on first arm to `~/.fusion-speech/models/wakeword/` from the
    openWakeWord GitHub releases (URL configurable for a mirror).
  - RPC: `speech.wake {arm}` arms/disarms; `speech.set_hotwords {hotwords}`
    swaps words at runtime (pretrained names only). Coexistence: on wake →
    `speech.wake` notification → pause wake → auto `start_listening` for the
    listen window → `stop_listening` → resume wake.
  - Verified: 72 tests green incl. a real-load test (downloads hey_jarvis ONNX,
    feeds 2s silence, asserts no false wake).
- **M4** — streaming TTS playback. **done**
  - `tts_playback.py`: `StreamingPlayer` consumes `mlx_bridge.synthesize_stream`,
    resamples 24k→16k, feeds a bounded queue a writer task drains into a
    `sounddevice.OutputStream` (started on first chunk = low TTFA). Cancel drops
    pending + closes the stream. Fail-soft on device error.
  - Mic guard: while speaking, wake paused; if ASR was listening, stopped then
    restarted — no self-trigger, no echo. Reuses M2/M3 primitives.
  - RPC: `speech.speak`, `speech.stop_tts`; `speech.tts` events
    (starting/speaking/done/cancelled/error) on `/stream/stt`.
  - Verified: ~80+ tests green incl. real-load (Kokoro via fusion-mlx, fake output, headless).

### Hardening (P0-P3 audit pass)

Adversarial audit (`audit/fusion-speech-audit-0827.md`, 13 findings across
FATAL/LOGIC/ARCH/PERF/MAINT) fully fixed. No public contract changed; this
pass tightens security, correctness, and resource handling:

- **F1 (FATAL)** — wake never fired: `wake_word._resolve_score_keys` now reads
  the real openwakeword `class_mapping`/`models` keys (`hey_jarvis_v0.1`) and
  matches by prefix, instead of `splitext(word)[0]` which stripped `.1`.
- **F2 (FATAL)** — IPC auth fail-open: `_verify_peer_uid` rejects when the
  socket is missing (fail-closed, was `return True`).
- **F3 (FATAL)** — unbounded-line DoS: `handle_client` caps a line at 1 MiB
  and closes the connection on overflow (bytearray buffer, no O(n²) copy).
- **M1** — HTTP retry: `mlx_bridge` retries transient failures (5xx /
  `TransportError` / `TimeoutException`) with 0.5/1.0/2.0s backoff; 4xx not
  retried; stream retry is connection-only (no double playback).
- **M2** — input size limits: `speech.transcribe` rejects audio >25 MB,
  `speech.speak`/`synthesize` reject text >5000 chars (`-32000`).
- **M3** — socket unlink TOCTOU: `server.start` suppresses `FileNotFoundError`
  on the stale-socket unlink.
- **L1/L2/L3** — wake auto-listen window is interruptible on ASR `final`; wake
  resume guarded against `None`; subscriber queue bounded at 64 (drop-oldest).
- **A1** — in-flight transcribe tasks are tracked, sem-limited (2), and
  cancelled/cleared on `stop()` (no orphans).
- **A2/A3/P2-1** — realtime capture callback writes a view (no per-frame
  alloc); shared ring not closed by a borrowing listener; ring overflow
  path in `write()` no longer logs I/O on the PortAudio thread (counters
  only; overruns surfaced in `stop()` and `stats()`). The ring notifier
  is unset in production — consumers poll the ring (`_drain_loop` every
  20ms), so the notifier is never called from the realtime callback.
- **P1** — TTS resample uses `scipy.signal.resample_poly` (anti-aliased);
  fail-soft to `np.interp` if scipy is absent. Queue cap clamped to [1, 10s].
- **P2** — VAD `feed` uses a pre-allocated buffer (no quadratic realloc).
- **P3** — `collect_pcm` trims by whole chunk (no second full-copy).

Verified: 93 tests green incl. adversarial gate suite (`tests/test_audit_fixes.py`)
and real-load gates (openwakeword ONNX silence, fusion-mlx STT/TTS).

### Architecture audit (5 waves, 20 findings)

Architecture audit (`audit/fusion-speech-audit-report-0827.md`, verdict: not
commercial-ready) fully remediated. 20 findings: 5 architecture (H), 7 runtime
risk (R), 8 engineering defect (E). No public RPC contract changed; the one
addition (`speech.reprobe`) is non-breaking. See `docs/ARCHITECTURE.md` §3a for
the component split and §12 (M4-hardening) for the wave-by-wave breakdown.

- **Wave 2 — god-object split (H1/H3/H4)**: `SpeechDaemon` split into a thin
  facade + 5 leaf services (`AsrService`/`WakeService`/`TtsService`/`AudioBus`/
  `EventBus`) under `fusion_speech/components/`. Mic ownership centralized in
  `AudioBus` (acquire/release, no shared-source racy peek). Lazy imports of
  engine modules promoted to top-level (import graph verified acyclic).
- **Wave 3 — concurrency correctness (H5/R3/R4/E7)**: per-service
  `asyncio.Lock` on state-mutating ops; facade `_handshake_lock` serializes the
  wake→ASR handshake (kills the double-wake check-then-act tear). ASR-final
  handshake frames delivered out-of-band via a per-handshake token registry
  (slow TTS drop-oldest can no longer drop the final frame). `_on_close` task
  creation bounded (Semaphore(4), drop + log on overflow). 6 concurrency tests
  added.
- **Wave 4 — observability + shutdown + persistence (E2/R6/E8/E6)**:
  zero-dependency `/metrics` (see above). Bounded shutdown drain: broadcast
  `speech.shutdown`, wait-for WS close with a 10s cap, cancel stragglers.
  `start.sh` stop grace raised 3s → 15s to absorb the drain. Persistent
  `~/.fusion-speech/run/state.json` (`start_time`/`restart_count`); on a
  crash-restart the daemon broadcasts `speech.restarted`.
- **Wave 5 — config/probe/error/wake-frame (E4/E1/E3/R5/H2/R7/E5)**:
  `SpeechConfig.__post_init__` fail-loud validation (sample_rate==16000,
  channels==1, port range, thresholds, fixed 1280 wake frame) — non-16000
  startup now fails instead of silently corrupting VAD/wake. `speech.reprobe`
  RPC + background re-probe loop (kills stale `audio_lane`). Malformed HTTP-200
  bodies (non-JSON STT, zero-rate WAV TTS) raise `AudioBridgeError` carrying
  the status instead of a bare `JSONDecodeError`/garbage playback. Wake/ASR
  resume failure after `speech.speak` broadcasts a `speech.tts` error event
  with `degraded: true` and latches `tts.degraded` (cleared on clean resume).
  Keepalive is a distinct `speech.keepalive` event type. Wake drain reads a
  fixed 1280-sample frame (was variable `available % 1280` → openwakeword
  probability corruption). Keepalive/queue depth now config-driven.

Verified: 147 tests green (`tests/test_wave5.py`, `tests/test_config_validation.py`,
`tests/test_r7_fixed_chunk_read.py`, `tests/test_concurrency.py`,
`tests/test_metrics.py`, `tests/test_state.py`, `tests/test_r6_graceful_drain.py`
+ the prior gate suites), ruff clean.

- **M5** — fusion-studio integration
  - Port `11465` registered in `architecture/port-registry.yaml` (check-ports CI
    green for fusion-speech).
  - Upstream issue filed: dahai80/fusion-studio#337 — requests a `SpeechBridge`
    Swift class (UDS JSON-RPC client) + mic permission + `/stream/stt` event
    subscription. Swift implementation is out of this project's scope (upstream
    issue-first → PR per workflow).
- **M6** — tests, integration, docs. **done**
  - Real-load full-chain integration gate (`tests/test_m6_real_load_integration.py`):
    5 steps over UDS JSON-RPC — ping/status, transcribe, synthesize, speak,
    wake arm/disarm — opt-in via the `real_load` marker (see [Testing](#testing)).
  - UDS client helper `tests/_uds_client.py` (newline-delimited JSON-RPC 2.0,
    4 MiB readline limit for large base64 payloads). `real_daemon` session
    fixture in `tests/conftest.py` spawns the real daemon on port 11465 +
    real socket, stripped of the autouse per-test isolation vars.
  - Upstream blocker filed: fusion-mlx issue #668 — `mlx-audio` Whisper
    `non_speech_tokens` `IndexError` makes STT return 500 for every model;
    step 2 skips (not xfails) so the gate is green upstream-side and flips to
    a real assertion on the fix. TTS path unaffected.
  - Verified: 148 tests green, 5 skipped (real_load-gated), ruff clean.

- **M7** — full-duplex voice sessions (issue #2). **done**
  - M-A WS PCM ingress: `/stream/audio` — 16 kHz mono PCM (int16 LE default,
    float32 optional) into the exact mic pipeline (ring → VAD → segment →
    transcribe); one ASR stream at a time (mic or WS); optional token auth;
    per-frame size bound.
  - M-B full-duplex + barge-in: VAD stays live during TTS playback;
    `speech.barge_in` on VAD onset; sink flush + player cancel land well
    under the 100 ms server-side budget; half-duplex keeps mic-era behavior.
    TTS streams back as binary PCM (`tts_sink=ws`) or plays locally.
  - M-C pluggable engines: `ASREngine` (`start_stream/feed/finish/cancel`) +
    `TTSEngine` (WAV stream + cancel) with a name registry; `local` = the
    fusion-mlx bridge (incl. `ref_audio`/`ref_text` cloning forward —
    upstream gap: fusion-mlx issue #1339); a registered cloud stub passes the
    same player conformance suite as the local engine.
  - M-D session state machine: `VoiceSession` (`listening → speaking →
    interrupted → listening`), barge-in latch + count, empty/low-confidence
    `speech.reprompt` branch; importable standalone or daemon-wired.
  - Also: audio-utility models (DeepFilterNet/denoise) excluded from the STT
    probe classification (they poisoned the model fallback with a 400), and
    the real-load transcribe step now runs a real assertion (upstream #668
    fixed).
  - Verified: 183 tests green + 6 real-load steps green (real STT/TTS/wake
    + WS session), ruff clean.
  - **M-E** TTS queueing (issue #3). **done** — concurrent `speak` frames
    queue FIFO per session (depth `FUSION_SPEECH_TTS_QUEUE_MAX`, default 8)
    instead of aborting the session; overflow drops the oldest frame and
    emits `speech.queue_overflow`; `tts_cancel` and barge-in clear the queue
    with the in-flight synthesis; sentence N+1 starts the moment N finishes
    (no client-side serialization, no inter-sentence gap beyond synthesis
    latency). Metric: `tts_queue_overflow_total`.
- **0.1.1** — bilingual docs (Chinese `README_CN.md`), scenario-based
  [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md) and developer
  [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md); patch over 0.1.0rc1. No runtime
  change. Known upstream limit: fusion-mlx issue #668 (STT 500) — see Testing.
- **0.1.4** — pluggable cloud engines + Azure AI Speech (issue #4). See
  [Cloud Engines](#cloud-engines-issue-4). `engine` param on
  `speech.transcribe`/`speech.synthesize`, `asr_engine`/`tts_engine` on voice
  sessions, new `speech.engines` RPC, word-level STT timestamps + dialect
  detection, reason-slug failure reporting, `cloud_requests_total` /
  `cloud_errors_total` metrics. Fail-soft: without Azure env vars the daemon
  is unchanged, local-only.
- **0.1.5** — process title + config single-point resolution (issues #5, #6).
  Daemon now shows as `fusion-speech server :11465` in `ps`/Activity Monitor
  (setproctitle, new core dep, fail-soft fallback). API key / port / data dir
  resolution migrated to `fusion_core.config` (`client_key`, `service_port`,
  `service_data_dir`): `FUSION_CLIENT_KEY` canonical, `FUSION_MLX_API_KEY`
  honored as deprecated alias, dev missing key = warning, `FUSION_PROFILE=prod`
  missing key = startup error. No behavior change otherwise.
- **0.1.6** — start.sh registry self-check (issue #7, config Phase 3.3). Start
  refuses when port 11465 is missing/mismatched in
  `architecture/port-registry.yaml`; `FUSION_SKIP_REGISTRY_CHECK=1` bypasses;
  missing registry file = warn + continue (standalone deploy).
- **0.1.7** — fusion docs-system v1.1 adoption (PR #8): six-page docs set,
  `CHANGELOG.md`, 12-key frontmatter + CI validation (`ci/validate_docs.py`).
  Wake R2 download test now skips when the optional `[audio]` extra is absent.

License: Apache-2.0.
