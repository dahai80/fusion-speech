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

- **HTTP server** (uvicorn) on `127.0.0.1:11465` — `/health` (M0) plus the
  `/stream/stt` WebSocket (M2, reused by M4) for non-UDS realtime clients. Used
  by `start.sh doctor` and external liveness checks.
- **UDS server** (`asyncio.start_unix_server`) at `~/.fusion-speech/run/fusion-speech.sock`
  (mode 0o600, dir 0o700) — newline-delimited JSON-RPC 2.0. Primary control plane.
- **Startup probe** — `probe_mlx_audio_lane` hits fusion-mlx `/v1/models`, checks
  `mlx_audio` importability, reports `{tts, stt, probed, error}` into `audio_lane`.
  Failures are non-fatal: daemon runs, but `/health` and `speech.status` flag the
  lane as down.
- **Shutdown** — SIGINT/SIGTERM set a stop event; daemon broadcasts
  `speech.shutdown` to WS subscribers, drains each leaf (player/wake/listener/
  UDS/http) under a bounded 10s timeout (R6), then unlinks the socket.
  `start.sh stop` sends SIGTERM then SIGKILL after a 15s grace window (E8),
  which accommodates the 10s drain plus fusion-mlx latency headroom.

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
| `config.py` | config | M0 | done (+ `__post_init__` fail-loud validation, E4) |
| `logging.py` | logging (re-export fusion_core) | M0 | done |
| `ipc/jsonrpc.py` | UDS transport + JSON-RPC dispatch | M0/M1 | done (+ typed error formatting, dispatch timeout) |
| `ipc/methods.py` | RPC method handlers | M0/M1/M5 | done (12 methods incl. `speech.reprobe`) |
| `mlx_bridge.py` | fusion-mlx HTTP delegation | M1 | done (+ malformed-200 explicit error, E3) |
| `metrics.py` | zero-dependency Prometheus metrics | M4-hardening | done (Counter/Gauge/Histogram, `/metrics`) |
| `state.py` | persistent daemon state | M4-hardening | done (restart_count, `speech.restarted`) |
| `server.py` | facade: lifecycle + orchestration | M0/M2/M4-hardening | done (SpeechDaemon thin facade, see §3a) |
| `components/audio_bus.py` | mic ownership (H3) | Wave 2 | done |
| `components/asr_service.py` | streaming ASR leaf | Wave 2 | done |
| `components/wake_service.py` | hotword wake leaf | Wave 2 | done |
| `components/tts_service.py` | TTS playback leaf | Wave 2 | done (+ R5 degraded latch) |
| `components/event_bus.py` | subscriber fan-out + ASR-final registry | Wave 2 | done (R3 per-handshake token) |
| `ring_buffer.py` | lock-free SPSC audio ring | M2 | done |
| `audio_source.py` | sounddevice capture | M2 | done |
| `vad.py` | voice activity detection | M2 | done (energy gate + Silero ONNX) |
| `stt_stream.py` | chunked streaming ASR | M2 | done (+ R4 bounded task creation) |
| `wake_word.py` | hotword wake | M3 | done (openwakeword ONNX, R7 fixed 1280-frame) |
| `tts_playback.py` | streaming TTS playback | M4 | done |

## 3a. Component Architecture (Wave 2 — god-object split)

`SpeechDaemon` was a 386-line god-object holding every resource + all
orchestration. The hardening audit (H1/H3/H4) split it into **five leaf
services + one thin facade**, mirroring the `ipc/` sub-package convention.
The facade owns lifecycle + cross-cutting orchestration only; each service
owns one resource and its leaf operations. Dependencies form a strict DAG
(no cycles — verified; H4 lifted lazy imports to module top).

```
                       ┌──────────────────────────────┐
                       │ SpeechDaemon (facade)        │
                       │  lifecycle + speak/_on_wake  │
                       │  + property delegation       │
                       └──┬───────┬───────┬───────┬───┘
                          │       │       │       │
                 ┌────────┘  ┌────┘  ┌────┘  ┌───┘
                 ▼           ▼       ▼       ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
          │AudioBus  │ │AsrService│ │WakeSvc   │ │TtsService│
          │ (mic own)│ │(listener)│ │(engine)  │ │(player)  │
          └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │            │
               └────────────┴───┬────────┴────────────┘
                                ▼
                         ┌─────────────┐
                         │  EventBus   │  (subscribe/publish,
                         │  + asr-final │   per-handshake handler
                         │   registry  │   registry — R3)
                         └─────────────┘
```

- **AudioBus** (H3) — sole mic owner. `acquire()`/`release()` borrow the
  shared `AudioSource`; ASR and wake never peek each other's source.
- **AsrService/WakeService/TtsService** — leaves holding `listener`/`engine`/
  `player` + their leaf ops. Each has a per-service `asyncio.Lock` (H5) so
  concurrent RPCs cannot tear check-then-set state. Services hold NO
  references to each other — the facade is the only combiner, which keeps
  test patches on `daemon.speak`/`daemon.start_listening` landing on the
  real facade path (93 tests stayed green with zero test edits).
- **EventBus** (R3) — subscriber fan-out + an out-of-band ASR-final handler
  registry. The auto-listen handshake registers a token before its
  `wait_for`; `publish` invokes the registered handler synchronously, so a
  slow TTS subscriber's drop-oldest cannot eat the handshake `final` frame.
  Per-handshake tokens (not a single slot) prevent a second wake from
  overwriting/clobbering the first's handler.
- **Facade** (`server.py`) — owns `config`, `audio_lane`, lifecycle
  (`start`/`stop`/`run_forever`/`build_app`/`probe_audio_lane`), and
  orchestration (`speak` wraps wake pause/resume + ASR stop/resume around
  `TtsService._play`; `_on_wake` runs the auto-listen handshake). rw
  properties (`daemon._listener`, `.listening`, `._wake`, `.wake_armed`,
  `.hotwords`, `.speaking`, `._player`) delegate to the services so existing
  tests reading/assigning facade fields keep working.

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
  E3: a 200 with a non-JSON/malformed body (a transient mid-load hiccup) is retried
  and, on final failure, raised as `AudioBridgeError("transcribe: malformed response
  (status N): ...", status=N)` carrying the status — never a bare `JSONDecodeError`.
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
  generator straight to the speaker. E3: a 200 whose first chunk is not a valid
  RIFF/WAVE (sample_rate parses to 0) aborts the stream with
  `AudioBridgeError("synthesize: malformed WAV (sample_rate=0)", status=200)` rather
  than silently piping zero-rate PCM that plays as garbage or hangs the player.
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

## 4a. Cloud Engine Layer (issue #4)

`cloud.py` + `cloud_azure.py` extend the M7 engine registry with cloud STT/TTS.
Same `ASREngine`/`TTSEngine` ABCs, so cloud engines slot into every existing
consumer path (RPC `engine` param, voice-session `asr_engine`/`tts_engine`,
mic pipeline via the injectable transcribe adapter).

- **Credentials** — fetched per call from an async provider (`StaticCredentials`
  wraps `FUSION_SPEECH_AZURE_KEY`/`_REGION` from config), so fusion-identity
  rotation needs no restart. Business apps never see keys; they select an
  engine by name. Registration is fail-soft: unset env vars = engines absent,
  daemon local-only.
- **Azure STT** — SDK push-stream + one-shot recognize in a worker thread
  (`asyncio.to_thread`); SDK-only because word-level timestamps and language ID
  are absent from the short-audio REST API. Detailed JSON (`NBest[0].Words`,
  100 ns ticks → ms) and the LID locale (dialect, e.g. `ar-EG`) ride along in
  the result dict and in `speech.asr` `final` frames. Retry rebuilds the
  recognizer from the retained PCM buffer (push streams are not seekable).
- **Azure TTS** — REST SSML POST (`X-Microsoft-OutputFormat:
  riff-16khz-16bit-mono-pcm` = the 16 kHz mono contract, no resampling),
  streamed chunk-wise; retry covers connection open only.
- **Failure model** — `CloudEngineError(reason=<slug>, retryable=...)`; the slug
  and flag are embedded in the message because JSON-RPC errors are
  message-only. Auth/config/bad-request are non-retryable (dead-letter);
  network/timeout/5xx/429 retry with 0.5/1/2 s backoff inside
  `cloud_retry`, wrapped by `cloud_call` which counts
  `cloud_requests_total{engine,kind}` / `cloud_errors_total{engine,kind,reason}`.
- **Optional dep** — `pip install -e ".[azure]"`; a missing SDK import raises
  `CloudEngineError(reason="azure_sdk_missing", retryable=False)` instead of
  crashing the daemon. Real-load gate: `tests/test_cloud_real_load.py`
  (module-level skip without live creds — flips to real assertions once
  fusion-identity provisions them).

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
| `speech.status` | `{}` | `{running, socket_path, port, uptime, mlx_audio_lane, stt_model, tts_model, listening, hotwords, speaking, restart_count, last_wake_word, metrics}` |
| `speech.transcribe` | `{audio: base64, language?, model?, filename?}` | `{text, language, duration, segments?}` |
| `speech.synthesize` | `{text, voice?, model?}` | `{audio: base64 wav, format: "wav", size}` |
| `speech.list_models` | `{}` | `{total, audio_models}` |
| `speech.start_listening` | `{}` | `{listening: true}` — arms mic capture + streaming ASR |
| `speech.stop_listening` | `{}` | `{listening: false, segments: <int>}` |
| `speech.set_hotwords` | `{hotwords: [...]}` | `{hotwords: [...]}` — register wake words (pretrained only) |
| `speech.wake` | `{arm: bool}` | `{armed: bool, words: [...]}` — arm/disarm wake |
| `speech.speak` | `{text, voice?, model?}` | `{played, chars, sample_rate}` — stream TTS to speaker (pauses mic) |
| `speech.stop_tts` | `{}` | `{stopped}` — cancel active playback |
| `speech.reprobe` | `{}` | `{audio_lane}` — re-probe fusion-mlx audio lane (E1, additive non-breaking) |

Model resolution: when `model` is omitted, the daemon uses the configured default
(`FUSION_SPEECH_STT_MODEL` / `FUSION_SPEECH_TTS_MODEL`); if that id is not in the
probed fusion-mlx set, it falls back to the first probed audio model of that
modality (WARNING-logged). This makes `speech.synthesize {}` work out-of-box even
when fusion-mlx's loaded model differs from the configured default.

**Notifications (M2/M3/M4, implemented).** The daemon pushes `speech.asr`,
`speech.wake`, and `speech.tts` events to connected WebSocket subscribers on
`/stream/stt` (not over UDS — WS is the realtime push channel; UDS stays
request/response). Notification frames carry no `id`; clients must not reply.
- `speech.asr` states: `listening` (capture armed), `speaking` (VAD onset),
  `transcribing` (segment posted to fusion-mlx), `final` (transcript +
  `text`/`language`/`duration`, or `skipped:true` for sub-0.1s segments),
  `error`.
- `speech.wake` carries `{word, ts}` and is immediately followed by the
  auto-started `speech.asr` frames.
- `speech.tts` states: `starting`/`speaking`/`done`/`cancelled`/`error`
  (M4). `error` carries `degraded: true` + the resume-failure detail when a
  `speak` could not cleanly re-arm wake/ASR after playback (R5).
- `speech.shutdown` is broadcast on graceful stop so WS clients can drain
  in-flight frames before the socket closes (R6).
- `speech.restarted` carries `{restart_count, ts}` and is broadcast when the
  daemon boots and `restart_count > 1` (crash recovery or deliberate restart,
  E6).
- `speech.keepalive` (H2) is a distinct event type sent after `ws_keepalive_s`
  idle so proxies do not close the socket. It is NOT a disguised `speech.asr`
  frame — clients filtering `speech.asr` for real ASR state do not sift
  keepalive noise.

**Methods (M4, implemented).**

| Method | Purpose |
|---|---|
| `speech.speak` | `{text, voice?, model?}` → stream TTS to speaker, mic paused |
| `speech.stop_tts` | `{}` → cancel playback |

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
(FastAPI), not over UDS. `EventBus.subscribe()` returns a per-client
`asyncio.Queue` (maxsize from `FUSION_SPEECH_WS_SUBSCRIBER_QUEUE`, default 64);
`publish()` fans out with drop-oldest on `QueueFull` (a slow client must never
stall the capture pipeline). The WS handler reads its queue with a
`ws_keepalive_s` timeout (default 30s), sending `speech.keepalive` on idle (H2:
a distinct event type, not a disguised `speech.asr` frame).

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
    listener), else builds its own RingBuffer + AudioSource. Drain task reads a
    **fixed** 1280-sample (80ms @ 16k) frame — openwakeword's native chunk size —
    and calls `openwakeword.Model.predict`. R7: the read is fixed at
    `wake_chunk_samples` (default 1280), not `available - (available % 1280)` —
    a variable-length read (2560/3840…) corrupts the model's 80ms-frame score
    contract. Leftover samples stay in the ring for the next pass (no loss). The
    predict runs off the event loop via `run_in_executor` (R1) so a blocking ONNX
    call does not starve ASR-final/WS writes; the single drain task serializes
    frames so the non-thread-safe ONNX session sees no concurrent access.
    `Model` is built with `inference_framework="onnx"` and
    `melspec_model_path`/`embedding_model_path` pointed at the cache dir, so all
    three ONNX files load from our cache (no reliance on package resources, which
    ship empty). A score ≥ `wake_threshold` (default 0.5) fires `on_wake(word)`
    ONCE, then a `wake_debounce_s` cooldown blocks repeats (one utterance = one
    wake). `on_wake` may be a coroutine (the daemon's `_on_wake` is); the engine
    schedules it via `asyncio.ensure_future` and tracks the task for clean stop.
- **Fail-soft** — if openwakeword is unimportable, the model download fails, or
  `Model` construction raises, `enabled=False` and the engine is a no-op.
  `arm_wake()` returns `{armed: false}`; the daemon keeps running. Never fatal.
- **Coexistence (chosen model)** — wake paused during ASR + auto-listen on wake.
  On `_on_wake` (serialized by `_handshake_lock`, H5): broadcast `speech.wake` →
  `wake.pause()` → `start_listening()` (reusing wake's mic source if wake owns
  it) → wait on a per-handshake `asyncio.Event` bounded by `wake_auto_listen_s`
  (default 8s) → `stop_listening()` → `wake.resume()`. The window ends early on
  an ASR `final` frame (delivered out-of-band via EventBus's handler registry,
  R3, so a slow TTS subscriber cannot drop it) plus a `wake_debounce_s` silence
  guard — no fixed dead-wait. The pause is critical: while paused the drain does
  NOT read the ring, so the ASR listener is the sole SPSC consumer (no read
  race). If ASR is already active when wake fires, the handshake is skipped
  (already listening).
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
speech.speak ─▶ mlx_bridge.synthesize_stream ─▶ StreamingPlayer ─▶ sounddevice.OutputStream (speaker)
   (RPC)        (POST /v1/audio/speech stream)  (fetch→resample→queue→writer)  (started on first chunk)
```

The daemon owns the `StreamingPlayer`, which in turn owns the
`sounddevice.OutputStream`. `tts_playback.StreamingPlayer`
runs the pipeline: fetch PCM chunks from `mlx_bridge.synthesize_stream` →
resample 24k→16k → push into a bounded queue → a writer task drains the queue
into the `OutputStream`. The stream is started on the first chunk, so
time-to-first-audio is bound by the first PCM chunk, not the full synthesis.

- **mlx_bridge.synthesize_stream** — async generator yielding the WAV header then
  PCM chunks from the chunked HTTP response as they arrive.
- **tts_playback.py** — `StreamingPlayer` owns the playback lifecycle. Bounded
  queue (`FUSION_SPEECH_TTS_BUFFER_S` seconds) backpressures synthesis through
  httpx's streaming read if the consumer stalls. Cancel drops pending chunks and
  closes the stream. Fail-soft on device error: if `OutputStream` construction or
  write fails, playback reports the error and the daemon stays up.
- **Mic guard.** While speaking, wake is paused; if ASR was listening, it is
  stopped then restarted after playback ends — no self-trigger, no echo. Reuses
  the M2/M3 primitives (`wake.pause()`, `stop_listening()`/`start_listening()`).
  R5: if the resume (`wake.resume`/`start_listening`) fails, the facade no
  longer swallows it silently — it broadcasts a `speech.tts error` event with
  `degraded: true` and latches `TtsService.degraded` so clients know the
  pipeline needs re-arm; a clean resume clears the flag.
- **Events.** `speech.tts` notifications (`starting`/`speaking`/`done`/`cancelled`/`error`)
  flow on the existing `/stream/stt` WebSocket — not a separate endpoint, and
  there is no stream id (one active playback at a time).

The old planned-API sketch (`synthesize_stream`/`tts_chunk` RPC + a separate
WS `/stream/tts` with stream ids) is collapsed into the above.

## 9. Configuration & Runtime State

No config file — environment only. `SpeechConfig` (dataclass) reads env at
construction in `config.py`; `load_config()` materializes it, makes the log dir,
and logs the resolved values.

**Validation (E4).** `SpeechConfig.__post_init__` fail-loud-raises `ValueError`
on invalid values rather than silently running a broken pipeline:
- `sample_rate` must be 16000 — hard-pinned by the VAD/wake/openwakeword
  contract (each model is trained on 16 kHz mono). Setting
  `FUSION_SPEECH_SAMPLE_RATE` to anything else makes startup fail loudly
  instead of producing garbage probabilities.
- `channels` must be 1 (mono pipeline).
- `0 < wake_threshold <= 1`, `wake_auto_listen_s > 0`, `wake_debounce_s >= 0`,
  `0 < vad_speech_threshold < 1`, `0 < tts_playback_buffer_s <= 10`,
  `chunk_ms > 0`, `0 <= port <= 65535` (0 = ephemeral OS bind, used by tests),
  `wake_chunk_samples` must be 1280 (openwakeword frame contract),
  `ws_keepalive_s > 0`, `ws_subscriber_queue > 0`, `probe_interval_s > 0`.

Key fields: `port`, `socket_path`, `sample_rate`, `stt_model`, `tts_model`,
`tts_voice`, `language`, `mlx_base_url`, `mlx_api_key`, `vad_speech_threshold`,
`chunk_ms`, `wake_chunk_samples`, `ws_keepalive_s`, `ws_subscriber_queue`,
`probe_interval_s`, `log_level`, `log_dir`. See README for the env-var table.

**Persistent state (E6).** `DaemonState` (`state.py`) persists
`~/.fusion-speech/run/state.json`: `start_time`, `restart_count`,
`last_wake_word`, `uptime_at_stop`. `start()` bumps `restart_count`
(`enter_start`) and, if count > 1, broadcasts `speech.restarted` so clients
learn the daemon was rebooted (crash recovery). `stop()` records
`uptime_at_stop` (`enter_stop`). A crash (no `stop()`) leaves the file in
place; the next boot's `enter_start` still increments, so `restart_count`
reflects reboots regardless of clean vs crash.

**Runtime state** lives on the leaf services + facade (Wave 2):
- Facade: `running`, `start_time`, `audio_lane`, `socket_path`, `port`,
  `_handshake_lock`, `_wake_listen_done`/`_last_final_ts` (handshake), `state`,
  `_restart_count`, `_probe_task`.
- `AsrService`: `listener`, `listening`.
- `WakeService`: `engine`, `armed`, `hotwords`.
- `TtsService`: `player`, `speaking`, `degraded` (R5).
- `EventBus`: `_subscribers` set, ASR-final handler registry.
- `AudioBus`: the shared `AudioSource` (sole mic owner, H3).
- `_server`, `_app`, `_http_task` — owned asyncio/uvicorn handles (facade).

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
  fusion-mlx via `~/fusion/fusion-mlx/start.sh`)
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
- **M4** — streaming TTS playback: `tts_playback.py`, `speech.speak`/`speech.stop_tts`
  RPC, `speech.tts` events on `/stream/stt`. **done** — StreamingPlayer
  (fetch→resample 24k→16k→bounded queue→writer→`sounddevice.OutputStream`,
  TTFA on first chunk, cancel, fail-soft), mic guard (pause wake / stop-restart
  ASR), `speech.tts` events (starting/speaking/done/cancelled/error) on
  `/stream/stt`. Verified: 83 tests green incl. real-load (Kokoro via fusion-mlx,
  fake output, headless).
- **M4-hardening** — architecture audit remediation (5 waves, 20 findings).
  **done** — fixes the structural blockers to production commercial release
  identified in `audit/fusion-speech-audit-report-0827.md`. Wave 2 split the
  `SpeechDaemon` god-object into 5 leaf services + a thin facade (§3a); Wave 3
  added per-service locks + a per-handshake ASR-final registry (H5/R3/R4); Wave 4
  added zero-dependency `/metrics`, bounded shutdown drain, 15s start.sh grace,
  and persistent restart state (E2/R6/E8/E6); Wave 5 added config validation,
  self-healing re-probe, malformed-200 errors, resume-failure surfacing,
  separated keepalive event type, and a fixed 1280 wake frame (E4/E1/E3/R5/H2/R7/E5).
  See README Hardening for the finding-by-finding summary. Verified: 147 tests
  green, ruff clean.
- **M5** — fusion-studio integration: SpeechBridge.swift, mic permission,
  port-registry `11465`. Upstream issue → PR.
- **M6** — tests, integration (real fusion-mlx start/stop), final docs.
