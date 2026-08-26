# fusion-speech

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
~/claude-home/fusion-mlx/start.sh start

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
| `FUSION_SPEECH_TTS_MODEL` | `mlx-community/Kokoro-82M-bf16` | TTS model id |
| `FUSION_SPEECH_TTS_VOICE` | `af_heart` | TTS voice |
| `FUSION_SPEECH_LANGUAGE` | `en` | default STT language (set `zh` with `openai/whisper-large-v3`) |
| `FUSION_SPEECH_SAMPLE_RATE` | `16000` | capture sample rate |
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
| `FUSION_SPEECH_LOG_LEVEL` | `INFO` | log level |
| `FUSION_SPEECH_LOG_DIR` | `~/.fusion-speech/logs` | log directory |
| `FUSION_MLX_API_KEY` | (from settings.json) | fusion-mlx bearer key |

## RPC API

JSON-RPC 2.0 over UDS, newline-delimited. Peer UID verified (fail-closed).

| Method | Params | Returns |
|---|---|---|
| `speech.ping` | `{}` | `{pong: true, timestamp}` |
| `speech.status` | `{}` | running, socket_path, port, uptime, mlx_audio_lane, stt_model, tts_model, listening, hotwords |
| `speech.transcribe` | `{audio: base64, language?, model?, filename?}` | `{text, language, duration, segments?}` |
| `speech.synthesize` | `{text, voice?, model?}` | `{audio: base64 wav, format: "wav", size}` |
| `speech.list_models` | `{}` | `{total, audio_models}` |
| `speech.start_listening` | `{}` | `{listening: true}` — arms mic capture + streaming ASR |
| `speech.stop_listening` | `{}` | `{listening: false, segments: <int>}` |
| `speech.set_hotwords` | `{hotwords: [...]}` | `{hotwords: [...]}` — register wake words (pretrained names only; unknown dropped) |
| `speech.wake` | `{arm: true\|false}` | `{armed: bool, words: [...]}` — arm/disarm wake detection |
| `speech.speak` | `{text, voice?, model?}` | `{played, chars, sample_rate}` — streams TTS to the speaker (pauses mic) |
| `speech.stop_tts` | `{}` | `{stopped}` — cancel active playback |

M2 realtime push: connect WebSocket `/stream/stt` to receive `speech.asr`
notification frames (`listening` → `speaking` → `transcribing` → `final`, or
`error`; `keepalive` every 30s idle). M3 adds `speech.wake` notifications on the
same socket (then auto-starts ASR for the listen window). M4 adds speech.tts
progress events on the same socket.

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

## Integration

- **fusion-mlx** (hard dependency): owns `/v1/audio/transcriptions` (STT) and
  `/v1/audio/speech` (TTS). fusion-speech probes the audio lane at startup and
  reports readiness in `/health` and `speech.status`.
- **fusion-core** (in-tree shared base): `mlx_client.FusionMLXClient` for HTTP,
  `http.app.create_app` for the FastAPI factory, `config` for base URL / API key
  resolution, `logging` for log setup.
- **fusion-studio** (Swift desktop, M5): calls the daemon over UDS JSON-RPC;
  registers a SpeechBridge + mic permission. Upstream change, filed as issue then PR.
- **port-registry**: claims `11465` in `architecture/port-registry.yaml` (M5).

Only this project's directory is modified. Upstream fusion-mlx / fusion-studio /
port-registry changes go through issue first, then PR.

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
  - RPC: `speech.speak`, `speech.stop_tts`; `speech.tts` events (starting/speaking/done/cancelled/error) on `/stream/stt`.
  - Verified: ~80+ tests green incl. real-load (Kokoro via fusion-mlx, fake output, headless).
- **M5** — fusion-studio integration (upstream issue/PR)
- **M6** — tests, integration, docs

License: Apache-2.0.
