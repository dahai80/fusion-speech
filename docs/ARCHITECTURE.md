# fusion-speech — Technical Architecture

## 1. Design Principle

**Delegate, don't duplicate.** fusion-speech never loads model weights. All STT/TTS
inference runs in fusion-mlx over HTTP. fusion-speech owns only what fusion-mlx
structurally cannot: the microphone, the speaker, wake-word detection, and the
streaming glue between them and the HTTP endpoints.

This keeps one source of truth for models (fusion-mlx's `~/.fusion-mlx/models`
cache, `mlx-community/*` weights) and lets fusion-speech stay a thin realtime
process. The cost is HTTP latency on the loopback interface (sub-millisecond on
Apple Silicon), acceptable because audio is chunked and streamed, not sent whole.

**Local-first.** No network egress. HTTP only to `127.0.0.1`. UDS only on the
local filesystem. Peer-UID verified so only same-user processes may call the daemon.

## 2. Process & Lifecycle

One long-running asyncio daemon: `python3 -m fusion_speech.server`.

- **HTTP server** (uvicorn) on `127.0.0.1:11465` — `/health` only (M0). Used by
  `start.sh doctor` and external liveness checks. Will host streaming WebSocket
  endpoints in M2/M4 for non-UDS clients.
- **UDS server** (`asyncio.start_unix_server`) at `~/.fusion-speech/run/fusion-speech.sock`
  (mode 0o600, dir 0o700) — newline-delimited JSON-RPC 2.0. Primary control plane.
- **Startup probe** — `probe_mlx_audio_lane` hits fusion-mlx `/v1/models`, checks
  `mlx_audio` importability, reports `{tts, stt, probed, error}` into `audio_lane`.
  Failures are non-fatal: daemon runs, but `/health` and `speech.status` flag the
  lane as down.
- **Shutdown** — SIGINT/SIGTERM set a stop event; UDS server closes, HTTP task
  cancels, socket file unlinked. `start.sh stop` sends SIGTERM then SIGKILL after
  a 3s grace window.

Lifecycle is owned by `start.sh` (PID file + mkdir atomic lock + process-identity
check `*fusion_speech.server*`), mirroring the canonical `fusion-projects/start.sh`.

## 3. Layer Breakdown

```
┌─────────────────────────────────────────────────────────┐
│ fusion-speech daemon (asyncio, one process)             │
│                                                         │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ UDS JSON-RPC│   │ HTTP /health │   │  WS streams  │  │
│  │  (control)  │   │   (liveness) │   │ (M2/M4 data) │  │
│  └──────┬──────┘   └──────┬───────┘   └──────┬───────┘  │
│         │                 │                  │          │
│         └────────┬────────┴──────────────────┘          │
│                  ▼                                      │
│         ┌────────────────┐    ┌──────────────┐          │
│         │ SpeechDaemon   │───▶│ mlx_bridge   │── HTTP ─▶│ fusion-mlx
│         │ (state machine)│    │ (STT/TTS)    │          │ /v1/audio/*
│         └────────────────┘    └──────────────┘          │
│                 ▲                  ▲                     │
│         ┌───────┴───────┐  ┌───────┴────────┐           │
│         │ realtime      │  │  tts_playback  │           │
│         │ mic→VAD→ASR   │  │  (M4)          │           │
│         │ (M2)          │  └────────────────┘           │
│         └───────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

Modules (M1 state marked):

| Module | Layer | Milestone | Status |
|---|---|---|---|
| `config.py` | config | M0 | done |
| `logging.py` | logging (re-export fusion_core) | M0 | done |
| `ipc/jsonrpc.py` | UDS transport + JSON-RPC dispatch | M0/M1 | done (+ typed error formatting) |
| `ipc/methods.py` | RPC method handlers | M0/M1 | done (ping/status/transcribe/synthesize/list_models) |
| `mlx_bridge.py` | fusion-mlx HTTP delegation | M1 | done (probe/transcribe_stream/synthesize_stream/synthesize_full) |
| `server.py` | daemon lifecycle + FastAPI | M0 | done |
| `ring_buffer.py` | lock-free SPSC audio ring | M2 | todo |
| `audio_source.py` | sounddevice capture | M2 | todo |
| `vad.py` | voice activity detection | M2 | todo |
| `stt_stream.py` | chunked streaming ASR | M2 | todo |
| `wake_word.py` | hotword wake | M3 | todo |
| `tts_playback.py` | streaming TTS playback | M4 | todo |

## 4. Inference Delegation (fusion-mlx boundary)

fusion-mlx exposes an OpenAI-compatible API. fusion-speech consumes exactly two
audio endpoints and nothing else:

**STT** — `POST /v1/audio/transcriptions`
- Multipart: `file` = PCM/WAV bytes, `model` = `mlx-community/whisper-large-v3-mlx`,
  `language` = `zh` (default).
- Response: `{text, language, duration, segments?}`.
- M1 bridge: `transcribe_stream(config, audio_bytes, *, filename, language, model)` —
  multipart POST, returns the parsed dict, raises `AudioBridgeError(status=...)` on
  non-200. M2 makes this chunked (VAD-segmented audio, post when a segment closes).
- **Upstream blocker (M1):** mlx-community whisper checkpoints ship
  `model_type:"whisper"` but no `architectures` field, so fusion-mlx
  `detect_model_type` skips them → 404 for `whisper`/`whisper-large-v3-mlx`. The
  bridge is correct against the contract; `speech.transcribe` returns a clear
  `AudioBridgeError (status 404)`. Tracked: file issue → PR on fusion-mlx (fall
  back to `model_type:"whisper"` → audio_stt when `architectures` absent).

**TTS** — `POST /v1/audio/speech`
- JSON: `{model, input, voice, response_format, stream, streaming_interval?}`.
  `response_format=wav`; non-stream = whole WAV, stream = 44-byte WAV header
  (data-size `0xFFFFFFFF`) + chunked int16 PCM.
- M1 bridge: `synthesize_stream(config, text, *, voice, model, stream, streaming_interval)`
  — async generator yielding WAV header then PCM chunks; parses sample_rate from the
  header. `synthesize_full` collects all chunks into one `bytes`. M4 wires the
  generator straight to the speaker.
- **Verified real-load (M1):** `speech.synthesize` over UDS JSON-RPC → fusion-mlx
  Qwen3-TTS, 388 KB 16-bit mono 24 kHz WAV in 5.3 s.

**Why not load models here?** fusion-mlx already manages the MLX weight cache,
warm models, and a shared HTTP surface for every other fusion-* app. Duplicating
that would split the cache and double GPU memory. The bridge adds only the two
audio POSTs on top of `FusionMLXClient.client` (an `httpx.AsyncClient` with Bearer
auth), since `FusionMLXClient` itself has no audio methods.

**Auth.** `FusionMLXClient` sends `Authorization: Bearer <key>`. The key resolves
from `fusion_core.config.resolve_api_key()` (env `FUSION_MLX_API_KEY`, else
`~/.fusion-mlx/settings.json` `auth.api_key`). Mismatched env vs settings causes
401; the startup probe surfaces this as `audio_lane.error`.

## 5. IPC Contract (UDS JSON-RPC)

Newline-delimited JSON-RPC 2.0 over UDS. One JSON object per line, UTF-8.

**Frame.** `handle_client` reads up to 65536 bytes, splits on `\n`, parses each
line. A line may be a request (has `id`) or a notification (no `id`, no reply).
Replies are single-line JSON terminated by `\n`.

**Peer verification.** `_verify_peer_uid` calls `getsockopt`:
- macOS: `SOL_LOCAL=0`, `LOCAL_PEERCRED=0x001`.
- Linux: `SOL_SOCKET=1`, `SO_PEERCRED=17`.
On mismatch (different UID) or `getsockopt` failure, the connection is dropped
(fail-closed). Only same-user processes may call the daemon.

**Dispatch.** `JsonRpcDispatcher.dispatch` validates `jsonrpc == "2.0"` (else
`-32600`), rejects empty/missing method (`-32601`), routes to handler. String
params are normalized to dict. Handler exceptions map to `-32000` via
`_format_handler_error` as `"<ExceptionType> [(status N)]: <message>"` (so an
`httpx.ReadTimeout` with empty `str()` still surfaces a clear
`ReadTimeout`, and `AudioBridgeError` carries its HTTP status); parse errors to
`-32700`; unexpected internal errors to `-32603`.

**Methods (M0/M1, implemented).**

| Method | Params | Result |
|---|---|---|
| `speech.ping` | `{}` | `{pong: true, timestamp: <float>}` |
| `speech.status` | `{}` | `{running, socket_path, port, uptime, mlx_audio_lane, stt_model, tts_model, listening, hotwords}` |
| `speech.transcribe` | `{audio: base64, language?, model?, filename?}` | `{text, language, duration, segments?}` (STT — upstream blocker) |
| `speech.synthesize` | `{text, voice?, model?}` | `{audio: base64 wav, format: "wav", size}` |
| `speech.list_models` | `{}` | `{total, audio_models}` |

Model resolution: when `model` is omitted, the daemon uses the configured default
(`FUSION_SPEECH_STT_MODEL` / `FUSION_SPEECH_TTS_MODEL`); if that id is not in the
probed fusion-mlx set, it falls back to the first probed audio model of that
modality (WARNING-logged). This makes `speech.synthesize {}` work out-of-box even
when fusion-mlx's loaded model differs from the configured default.

**Notifications (M2+).** The daemon pushes async events to connected clients:
`speech.asr` (partial/final transcript), `speech.wake` (wake fired), `speech.tts_chunk`
(playback progress). Notifications carry no `id`; clients must not reply.

**Methods (M2-M4, planned).**

| Method | Purpose |
|---|---|
| `speech.set_hotwords` | register hotword phrases |
| `speech.wake` | arm/disarm wake detection |
| `speech.start_listening` / `speech.stop_listening` | control mic capture (M2) |

## 6. Realtime Audio Pipeline (M2-M4)

Goal: capture → VAD → chunked STT → transcript notification, all streaming.

```
sounddevice (16kHz mono) ─▶ ring_buffer ─▶ VAD ─▶ stt_stream ─▶ mlx_bridge.transcribe ─▶ speech.asr
       (capture)            (SPSC lock-free) (energy+silero)  (segment close)           (notification)
```

- **ring_buffer.py** — lock-free single-producer single-consumer ring (numpy
  backing). Producer = sounddevice callback (realtime thread, must not block);
  consumer = asyncio task draining into VAD. SPSC avoids locks on the hot path.
- **audio_source.py** — `sounddevice.Stream` wrapper, 16 kHz mono, chunk callback
  into the ring. Lifecycle: `start()`/`stop()`; the daemon toggles via
  `speech.start_listening`/`speech.stop_listening`.
- **vad.py** — two-tier: fast energy gate (RMS threshold) filters silence, then
  `silero-vad` (ONNX) for robust speech/non-speech. Emits segment-open and
  segment-close events. Silence-after-speech closes a segment and triggers STT.
- **stt_stream.py** — on segment close, hands the segment's PCM to
  `mlx_bridge.transcribe_stream`. Partial results (if the STT model supports
  streaming chunks) are pushed as `speech.asr` notifications with `partial=true`;
  final results with `final=true`.

**Backpressure.** The ring buffer drops oldest on overflow (realtime capture
must never block). The VAD consumer signals overrun via a counter surfaced in
`speech.status`.

## 7. Hotword Wake (M3)

Goal: low-CPU always-on listening for wake phrases, then hand off to streaming ASR.

- **wake_word.py** — ONNX keyword-spotting model via `onnxruntime` + `openwakeword`.
  Runs on the same mic ring buffer but on a downsampled/lighter path to save power.
- **Two-level filter** — short energy pre-filter (skip obvious silence), then the
  KWS model. Reduces ONNX invocations.
- **API** — `speech.set_hotwords` registers phrases; `speech.wake` arms/disarms.
  On a wake, the daemon emits a `speech.wake` notification and (if configured)
  auto-starts the streaming ASR pipeline for a fixed listen window.

**Open decision (pre-M3):** engine choice (openwakeword vs a custom KWS), model
cache path, and the realtime-channel priority between wake and active ASR. These
are confirmed against fusion-mlx's audio lane before M3 lands.

## 8. Streaming TTS Playback (M4)

Goal: speak text with minimal time-to-first-audio, playing PCM chunks as they
arrive rather than waiting for the full synthesis.

```
speech.synthesize_stream ─▶ mlx_bridge.synthesize_stream ─▶ tts_playback ─▶ sounddevice (output)
      (RPC)                  (POST /v1/audio/speech stream)  (边收边播)        (speaker)
```

- **mlx_bridge.synthesize_stream** — async generator yielding PCM chunks from the
  chunked HTTP response as they arrive.
- **tts_playback.py** — owns a `sounddevice.OutputStream`. Feeds chunks to a
  playback buffer; starts the output stream on the first chunk (time-to-first-audio).
  Resamples if the TTS sample rate differs from the capture rate.
- **API** — `speech.synthesize_stream` starts a stream, returns a stream id;
  `speech.tts_chunk` pulls progress; the daemon can also push `speech.tts_chunk`
  notifications. WS `/stream/tts` offers the same flow for non-UDS clients.

**Cancellation.** `speech.tts_cancel` (planned) stops playback and drops pending
chunks. Playback buffer bounded; if the consumer stalls, synthesis backpressures
through httpx's streaming read.

## 9. Configuration & Runtime State

No config file — environment only. `SpeechConfig` (dataclass) reads env at
construction in `config.py`; `load_config()` materializes it, makes the log dir,
and logs the resolved values.

Key fields: `port`, `socket_path`, `sample_rate`, `stt_model`, `tts_model`,
`tts_voice`, `language`, `mlx_base_url`, `mlx_api_key`, `vad_speech_threshold`,
`chunk_ms`, `log_level`, `log_dir`. See README for the env-var table.

**Runtime state** lives on `SpeechDaemon`:
- `running`, `start_time` — lifecycle + uptime.
- `audio_lane` — `{tts, stt, probed, error}` from the startup probe.
- `listening` — mic capture armed (M2).
- `hotwords` — registered wake phrases (M3).
- `_server`, `_app`, `_http_task` — owned asyncio/uvicorn handles.

Socket dir default `~/.fusion-speech/run` (0o700); socket file 0o600. Log dir
default `~/.fusion-speech/logs`. Both overridable via env.

## 10. fusion-* Integration Map

| Project | Relationship | Boundary |
|---|---|---|
| **fusion-mlx** | hard dependency (inference) | fusion-speech POSTs to `/v1/audio/transcriptions` + `/v1/audio/speech`; reads nothing else. No code change in fusion-mlx. If an audio endpoint is missing/misbehaving, file an issue there first. |
| **fusion-core** | shared in-tree base | consumes `mlx_client.FusionMLXClient`, `http.app.create_app`, `config`, `logging`. No modification — `pip install -e fusion-core`. |
| **fusion-studio** | upstream consumer (M5) | Swift SpeechBridge calls the daemon over UDS JSON-RPC; Info.plist mic permission. Upstream change → issue first, then PR. |
| **fusion-cli** | optional launcher | the `fusion` binary may wrap `start.sh`; no code change required unless a subcommand is added. |
| **port-registry** | registry (M5) | claims `11465` in `architecture/port-registry.yaml` + `check-ports.py`. Upstream doc change → issue/PR. |
| other fusion-* apps | none | no direct dependency; can use the daemon over UDS if same-user. |

**Rule enforced:** only `fusion-speech/` is modified in this repo's working tree.
Every other project is touched only through the issue → PR flow, code landing here
after merge.

## 11. Security

- **Local-only transport.** HTTP binds `127.0.0.1` only; UDS is a filesystem
  socket under a 0o700 dir. No listening port on any network interface.
- **Peer-UID verification.** Every UDS connection is verified via `getsockopt`
  (`LOCAL_PEERCRED` on macOS, `SO_PEERCRED` on Linux). Different UID → dropped.
  `getsockopt` failure → dropped (fail-closed, never fail-open).
- **Bearer auth to fusion-mlx.** API key from `fusion_core.config.resolve_api_key()`,
  never logged. Env override `FUSION_MLX_API_KEY`.
- **No secrets in logs.** Config logs the resolved base URL and model ids, never
  the API key.
- **No network egress.** All inference stays on loopback to fusion-mlx.

## 12. Milestone Plan

- **M0** — engineering skeleton: daemon, UDS JSON-RPC (ping/status), fusion-mlx
  audio-lane probe, `/health`, `start.sh`, unit tests, docs. **done**
- **M1** — fusion-mlx delegation bridge: implement `transcribe_stream` /
  `synthesize_stream` over HTTP; checkpoint = real-load whisper-large-v3-mlx +
  Kokoro, transcribe/synthesize playable. (uses key from settings.json; start/stop
  fusion-mlx via `~/claude-home/fusion-mlx/start.sh`)
- **M2** — realtime mic capture + streaming ASR: ring_buffer, audio_source, vad,
  stt_stream, `speech.asr`, WS `/stream/stt`.
- **M3** — hotword wake: wake_word, `set_hotwords`/`wake`, two-level filter.
- **M4** — streaming TTS playback: tts_playback, `synthesize_stream`/`tts_chunk`,
  WS `/stream/tts`.
- **M5** — fusion-studio integration: SpeechBridge.swift, mic permission,
  port-registry `11465`. Upstream issue → PR.
- **M6** — tests, integration (real fusion-mlx start/stop), final docs.

## 12. Milestone Plan
