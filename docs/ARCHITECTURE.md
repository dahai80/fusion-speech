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
| `server.py` | daemon lifecycle + FastAPI | M0/M2 | done (+ /stream/stt WS + subscriber fan-out) |
| `ring_buffer.py` | lock-free SPSC audio ring | M2 | done |
| `audio_source.py` | sounddevice capture | M2 | done |
| `vad.py` | voice activity detection | M2 | done (energy gate + Silero ONNX) |
| `stt_stream.py` | chunked streaming ASR | M2 | done |
| `wake_word.py` | hotword wake | M3 | done (openwakeword ONNX) |
| `tts_playback.py` | streaming TTS playback | M4 | todo |

## 4. Inference Delegation (fusion-mlx boundary)

fusion-mlx exposes an OpenAI-compatible API. fusion-speech consumes exactly two
audio endpoints and nothing else:

**STT** — `POST /v1/audio/transcriptions`
- Multipart: `file` = PCM/WAV bytes, `model` = `openai/whisper-tiny` (English;
  `openai/whisper-large-v3` for multilingual incl. zh), `language` = `en`/`zh`.
- Response: `{text, language, duration, segments?}`.
- M1 bridge: `transcribe_stream(config, audio_bytes, *, filename, language, model)` —
  multipart POST, returns the parsed dict, raises `AudioBridgeError(status=...)` on
  non-200. M2 makes this chunked (VAD-segmented audio, post when a segment closes).
- **Upstream fix required (M1, issue #649):** fusion-mlx `_is_hf_cache_mlx_compatible`
  rejected HF-format audio repos — `openai/whisper-tiny` ships HF `model.safetensors`
  (metadata `format:pt`, not `mlx`) + a full tokenizer, and its repo name carries no
  `mlx` marker, so it was skipped → 404. `mlx_audio` loads HF-format audio weights
  natively, so the fix adds an audio-model branch (`detect_model_type` ∈ audio_* with
  any weights → accept). Verified real-load after fix: `speech.transcribe` over UDS →
  `openai/whisper-tiny` → "Welcome to First National Bank! How may I assist you today?"
  in 1.5 s. Note: `mlx-community/whisper-*-mlx*` repos ship NO tokenizer files →
  unusable (empty vocab → `IndexError` at `non_speech_tokens`); use `openai/whisper-*`.

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

**Methods (M0/M1/M2, implemented).**

| Method | Params | Result |
|---|---|---|
| `speech.ping` | `{}` | `{pong: true, timestamp: <float>}` |
| `speech.status` | `{}` | `{running, socket_path, port, uptime, mlx_audio_lane, stt_model, tts_model, listening, hotwords}` |
| `speech.transcribe` | `{audio: base64, language?, model?, filename?}` | `{text, language, duration, segments?}` |
| `speech.synthesize` | `{text, voice?, model?}` | `{audio: base64 wav, format: "wav", size}` |
| `speech.list_models` | `{}` | `{total, audio_models}` |
| `speech.start_listening` | `{}` | `{listening: true}` — arms mic capture + streaming ASR |
| `speech.stop_listening` | `{}` | `{listening: false, segments: <int>}` |
| `speech.set_hotwords` | `{hotwords: [...]}` | `{hotwords: [...]}` — register wake words (pretrained only) |
| `speech.wake` | `{arm: bool}` | `{armed: bool, words: [...]}` — arm/disarm wake |

Model resolution: when `model` is omitted, the daemon uses the configured default
(`FUSION_SPEECH_STT_MODEL` / `FUSION_SPEECH_TTS_MODEL`); if that id is not in the
probed fusion-mlx set, it falls back to the first probed audio model of that
modality (WARNING-logged). This makes `speech.synthesize {}` work out-of-box even
when fusion-mlx's loaded model differs from the configured default.

**Notifications (M2/M3, implemented).** The daemon pushes `speech.asr` and
`speech.wake` events to connected WebSocket subscribers on `/stream/stt` (not
over UDS — WS is the realtime push channel; UDS stays request/response).
Notification frames carry no `id`; clients must not reply. `speech.asr` states:
`listening` (capture armed), `speaking` (VAD onset), `transcribing` (segment
posted to fusion-mlx), `final` (transcript + `text`/`language`/`duration`, or
`skipped:true` for sub-0.1s segments), `error`. `speech.wake` carries `{word,
ts}` and is immediately followed by the auto-started `speech.asr` frames. A
`keepalive` frame is sent after 30s idle so proxies do not close the socket.

**Methods (M4, planned).**

| Method | Purpose |
|---|---|
| `speech.synthesize_stream` | start a streaming TTS session (M4) |
| `speech.tts_chunk` | pull/notify TTS playback progress (M4) |

## 6. Realtime Audio Pipeline (M2-M4)

Goal: capture → VAD → chunked STT → transcript notification, all streaming.

```
sounddevice (16kHz mono) ─▶ ring_buffer ─▶ VAD ─▶ stt_stream ─▶ mlx_bridge.transcribe ─▶ speech.asr
       (capture)            (SPSC lock-free) (energy+silero)  (segment close)           (notification)
```

- **ring_buffer.py** — lock-free single-producer single-consumer ring (numpy
  backing, absolute unwrapped `_total_in`/`_total_out` counters so full/empty
  stay unambiguous). Producer = sounddevice callback (realtime thread, must not
  block); consumer = asyncio drain task feeding VAD. SPSC avoids locks on the hot
  path. `write()` drops oldest on overflow (`_overruns`, `_total_dropped`).
- **audio_source.py** — `sounddevice.InputStream` wrapper, 16 kHz mono, callback
  downmixes to mono and writes the ring. Lazy `import sounddevice` (headless-safe);
  tests inject `stream_factory`. Lifecycle: `start()`/`stop()`.
- **vad.py** — two-tier. Tier 1: energy RMS gate (skip ONNX when RMS below
  `0.01`). Tier 2: `silero_vad_16k_op15.onnx` via onnxruntime (CPUExecutionProvider).
  The ONNX model needs a rolling 64-sample context prepended to each 512-sample
  window (input length 576); the wrapper saves the last 64 as next context —
  without this, probabilities collapse to ~0.001 on all audio. State machine:
  onset (`prob >= threshold`), max-speech cut, silence close
  (`prob < neg_threshold` for `min_silence_samples`). Emits `on_open` (speech
  onset) and `on_close` (segment-ready) callbacks. Fail-soft to energy-only if
  the model file is missing.
- **stt_stream.py** — the consumer glue. Drain task reads the ring AND mirrors
  every sample into a `_capture` list (the ring is already drained, so a closed
  segment's PCM is sliced from `_capture`, not re-read). `on_open` marks the slice
  start with a 0.2s lead-in; `on_close` schedules `_transcribe_segment` (async).
  Skip gate: segments with `segment.samples < sample_rate//10` (sub-0.1s of pure
  speech) emit a `final` with `skipped:true` without calling transcribe. Real
  segments are PCM→16-bit WAV → `mlx_bridge.transcribe_stream` → `final`
  notification with text/language/duration. `on_open` fires a `speaking`
  notification via `run_coroutine_threadsafe` (the VAD runs on the drain task).

**Push channel.** `speech.asr` frames go to WebSocket `/stream/stt` subscribers
(FastAPI), not over UDS. `SpeechDaemon.subscribe()` returns a per-client
`asyncio.Queue`; `broadcast()` fans out with drop-oldest on `QueueFull` (a slow
client must never stall the capture pipeline). The WS handler reads its queue
with a 30s timeout, sending `keepalive` on idle.

**Backpressure.** The ring buffer drops oldest on overflow (realtime capture
must never block); overrun counters are tracked on the ring. The subscriber
fan-out is also drop-oldest so a stalled WS client does not backpressure the
notifier.

## 7. Hotword Wake (M3)

Goal: low-CPU always-on listening for wake phrases, then hand off to streaming ASR.

```
sounddevice (16kHz mono) ─▶ ring_buffer ─▶ wake_word (80ms frames) ─▶ openwakeword.predict ─▶ on_wake
       (shared mic)         (SPSC, shared)  (drain task)              (ONNX, score 0-1)        (handshake)
```

- **wake_word.py** — two classes, mirroring the M2 `vad.py` shape.
  - `WakeModelDownloader.ensure_models(words, cache_dir, base_url)` — fetches the
    wake word's `.onnx` + the shared `melspectrogram.onnx` + `embedding_model.onnx`
    into `cache_dir` via httpx, idempotent. `base_url` configurable (default
    openWakeWord GitHub releases v0.5.1) so a mirror can override it. openwakeword's
    own `download_models()` has no URL-override hook, so we roll our own.
  - `WakeWordEngine(config, *, on_wake, audio_source=None)` — the streaming
    detector. Reuses `audio_source.ring` if passed (shared mic with an active
    listener), else builds its own RingBuffer + AudioSource. Drain task reads
    1280-sample (80ms @ 16k) frames — openwakeword's native chunk size — and calls
    `openwakeword.Model.predict`. `Model` is built with `inference_framework="onnx"`
    and `melspec_model_path`/`embedding_model_path` pointed at the cache dir, so all
    three ONNX files load from our cache (no reliance on package resources, which
    ship empty). A score ≥ `wake_threshold` (default 0.5) fires `on_wake(word)`
    ONCE, then a `wake_debounce_s` cooldown blocks repeats (one utterance = one
    wake). `on_wake` may be a coroutine (the daemon's `_on_wake` is); the engine
    schedules it via `asyncio.ensure_future` and tracks the task for clean stop.
- **Fail-soft** — if openwakeword is unimportable, the model download fails, or
  `Model` construction raises, `enabled=False` and the engine is a no-op.
  `arm_wake()` returns `{armed: false}`; the daemon keeps running. Never fatal.
- **Coexistence (chosen model)** — wake paused during ASR + auto-listen on wake.
  On `_on_wake`: broadcast `speech.wake` → `wake.pause()` → `start_listening()`
  (reusing wake's mic source if wake owns it) → sleep `wake_auto_listen_s`
  (default 8s) → `stop_listening()` → `wake.resume()`. The pause is critical:
  while paused the drain does NOT read the ring, so the ASR listener is the sole
  SPSC consumer (no read race). If ASR is already active when wake fires, the
  handshake is skipped (already listening).
- **set_hotwords** — `WakeWordEngine.set_hotwords(words)` swaps the active wake
  words at runtime. Only pretrained names (`alexa`, `hey_mycroft`, `hey_jarvis`,
  `hey_rhasspy`, `timer`, `weather`) are usable; unknown names are logged + dropped.
  Rebuilds the openwakeword `Model` with the new word set.
- **API** — `speech.wake {arm}` arms/disarms; `speech.set_hotwords {hotwords}`
  registers words. `speech.wake` notifications flow on the same `/stream/stt` WS
  as `speech.asr` (no new endpoint); clients see `speech.wake` then the auto-started
  `speech.asr` frames on one socket.

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
- `wake_armed` — wake detection armed (M3).
- `_wake` — the active `WakeWordEngine`, lazily built on `speech.wake {arm:true}`,
  torn down on disarm/daemon stop.
- `_listener` — the active `StreamingListener`, lazily built on
  `speech.start_listening`, torn down on stop/daemon stop.
- `_subscribers` — set of per-WS-client `asyncio.Queue` for `/stream/stt`
  fan-out; `subscribe`/`unsubscribe`/`broadcast` manage it.
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
  `synthesize_stream` over HTTP; checkpoint = real-load STT (`openai/whisper-tiny`) +
  TTS (Qwen3-TTS), transcribe/synthesize playable. **done** (STT needed upstream
  fusion-mlx discovery fix, issue #649). (uses key from settings.json; start/stop
  fusion-mlx via `~/claude-home/fusion-mlx/start.sh`)
- **M2** — realtime mic capture + streaming ASR: ring_buffer, audio_source, vad,
  stt_stream, `speech.asr`, WS `/stream/stt`. **done** — ring buffer (SPSC,
  drop-oldest), energy-gate + Silero ONNX VAD (64-sample context fix), streaming
  listener (capture-mirror → segment → WAV → transcribe → final notification),
  `/stream/stt` WS subscriber fan-out, `speech.start_listening`/`stop_listening`
  RPC. Verified: 55 tests green (incl. real-load Silero on speech fixture +
  end-to-end start→capture→transcribe→final-to-subscriber integration).
- **M3** — hotword wake: `wake_word.py` (openwakeword ONNX), `set_hotwords`/`wake`
  RPC, auto-listen handshake. **done** — WakeWordEngine (shared mic ring, 80ms
  frame drain, predict→on_wake with debounce cooldown, pause/resume for ASR
  coexistence, fail-soft), WakeModelDownloader (httpx, configurable mirror URL),
  `speech.wake`/`set_hotwords` RPC + `speech.wake` notification on `/stream/stt`.
  Coexistence: wake paused during ASR, auto-start ASR on wake for
  `wake_auto_listen_s` then resume. Verified: 72 tests green incl. real-load
  (downloads hey_jarvis ONNX, 2s silence → no false wake).
- **M4** — streaming TTS playback: tts_playback, `synthesize_stream`/`tts_chunk`,
  WS `/stream/tts`.
- **M5** — fusion-studio integration: SpeechBridge.swift, mic permission,
  port-registry `11465`. Upstream issue → PR.
- **M6** — tests, integration (real fusion-mlx start/stop), final docs.
