---
doc_id: fusion-speech/guide
product: fusion-speech
class: app
entry: service
doc_type: guide
title: { zh: "M4 设计说明", en: "M4 Design Spec" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - docs/superpowers/specs/2026-08-26-m4-streaming-tts-playback-design.md
---

# M4 — Streaming TTS Playback (design)

Status: approved (2026-08-26). Owner: fusion-speech.

## Context

fusion-speech M0–M3 are done: daemon + UDS JSON-RPC + fusion-mlx delegation
bridge (M1), realtime mic capture + VAD + streaming ASR (M2), hotword wake (M3).
M4 adds the missing realtime layer: **streaming TTS playback** — speak text with
low time-to-first-audio (TTFA), playing PCM chunks as they arrive from fusion-mlx
rather than waiting for the full synthesis (the current M1 `speech.synthesize`
collects the whole WAV first).

The inference-side streaming fetch already exists:
`mlx_bridge.synthesize_stream` (`mlx_bridge.py:128`) opens a chunked HTTP response
from fusion-mlx `POST /v1/audio/speech {stream:true, response_format:"wav"}`,
parses the WAV header's sample rate on the first chunk (Kokoro ~24000 Hz), and
yields raw int16 PCM bytes thereafter. **M4 is the playback layer that consumes
that stream** — plus RPC, mic coexistence, and progress notifications.

### Decisions (confirmed)

- **Where audio renders:** the **daemon owns a `sounddevice.OutputStream`** and
  plays TTS to the default output as PCM chunks arrive. WS only carries progress
  events. Symmetric to how the daemon already owns mic capture (M2). Best fit for
  the README's "streaming TTS playback" realtime layer; lowest latency.
- **Mic coexistence while speaking:** **pause the mic while speaking.** While the
  daemon plays TTS, pause wake (mirror the M3 `_on_wake` pause). If a `speech.speak`
  arrives during an active ASR listen window, stop listening first, play, then
  restart ASR. No self-trigger wake on the assistant's own voice, no echoed-TTS
  transcription. Reuses existing M2/M3 primitives — no new concurrency.

These collapse the ARCHITECTURE-sketched `synthesize_stream` + `tts_chunk` +
`/stream/tts` trio into `speech.speak` + `speech.stop_tts` + `speech.tts` events
on the existing `/stream/stt` WS (single-speaker daemon needs no stream-id
multiplexing or separate endpoint).

## Approach

One new module `tts_playback.py`, mirroring the M2 `stt_stream.py` shape (one
active session, a fetch→write pipeline, fail-soft). Wire into the daemon the same
way `start_listening`/`stop_listening` did. Reuse `mlx_bridge.synthesize_stream`.

### New module: `fusion_speech/tts_playback.py`

`StreamingPlayer(config, *, synthesize=mlx_bridge.synthesize_stream, notifier=None,
stream_factory=None)`:

- `speak(text, voice="", model="") -> dict` — the single entry point.
  1. Open the `synthesize` async generator (`synthesize(config, text, voice=…,
     model=…, stream=True)`).
  2. First chunk: it carries the WAV header + first PCM bytes. Parse the TTS
     sample rate via `mlx_bridge._parse_wav_sample_rate(header)`. Strip the 44-byte
     header; the rest is int16 PCM.
  3. **Resample** if `tts_rate != config.sample_rate` (TTS is ~24k, daemon rate
     16k): numpy linear interpolation, logged once. Skip if rates match.
  4. int16 bytes → float32 PCM (×1/32768). Push each frame into a bounded
     `asyncio.Queue` (cap `tts_playback_buffer_s` seconds, default 2.0).
  5. A writer task drains the queue and writes frames to a
     `sounddevice.OutputStream(samplerate=tts_rate, channels=1, dtype="float32")`,
     **started on the first playable chunk** (TTFA). `sounddevice` imported lazily
     (headless-safe); `stream_factory` injectable for tests (mirrors
     `audio_source.py`).
  6. On generator exhaustion: signal the writer to flush + close the stream,
     return `{played: true, chars: len(text), sample_rate: tts_rate}`.
- **Bounded buffer + backpressure.** The queue cap prevents unbounded memory for
  long utterances. When the writer stalls (slow device), the fetch task blocks on
  `queue.put`, which backpressures httpx's streaming read — natural flow control.
  No silent drop (playback is not realtime capture; a stall should slow the
  producer, not lose audio).
- **Cancellation.** `cancel() -> dict` sets a cancel flag, closes the output
  stream, and drains the queue. The fetch task sees the flag and stops iterating
  the generator (closes the httpx stream). Returns `{stopped: true}`.
- **Properties:** `active` (a speak is in flight), `speaking` (writer task
  writing to the device).
- **Fail-soft.** `sounddevice` unimportable, output device open fails, or mlx
  returns a 4xx/5xx → `speak` returns `{played: false, error: str}`, the player
  is a no-op, daemon unharmed. Never fatal. Mirrors M3 wake fail-soft.

### Daemon wiring: `fusion_speech/server.py`

- `__init__`: `self._player = None`, `self.speaking = False`.
- `speak(text, voice="", model="") -> dict`:
  1. Build (or reuse) `StreamingPlayer` with `notifier=self.broadcast`.
  2. **Mic guard:** if `self._wake` armed → `wake.pause()`; if `self._listener`
     active → `was_listening = True; await self.stop_listening()`.
  3. `result = await player.speak(...)`, setting `self.speaking` around it.
  4. Finally: if `was_listening` → `await self.start_listening()`; if wake armed
     → `wake.resume()`. Reuses existing M2 `start/stop_listening` + M3
     `pause/resume`. Guarded: if player is None mid-flight, no-op.
  5. Return result.
- `stop_tts() -> dict` — `await self._player.cancel()` if active.
- `stop()` — stop player before listener/wake (add to existing cleanup block,
  ordered: player → wake → listener → servers).

### RPC: `fusion_speech/ipc/methods.py`

- `handle_speak(params, daemon)` — validate `text` non-empty; `voice`/`model`
  optional (resolve via existing `_resolve_tts_model`); `await daemon.speak(...)`;
  return `{played, chars, sample_rate}`.
- `handle_stop_tts(params, daemon)` — `await daemon.stop_tts()`; return
  `{stopped}`.
- `handle_status` — add `"speaking": daemon.speaking`.
- Register `speech.speak`, `speech.stop_tts` in `build_methods`.

### Notifications

Reuse the existing `/stream/stt` WS + `SpeechDaemon.broadcast` (no new endpoint,
no stream id). `speech.tts` frames:
`{event:"speech.tts", state, chars?, sample_rate?, error?}` where `state` ∈
`starting` (speak RPC accepted, fetch begun), `speaking` (first chunk played),
`done` (synthesis exhausted, stream closed), `cancelled` (stop_tts), `error`
(fail-soft). Same drop-oldest fan-out as `speech.asr`/`speech.wake`.

### Config: `fusion_speech/config.py`

Add two fields (env pattern via existing `_env_float` + `os.environ.get`):
- `tts_playback_buffer_s: float` → `2.0` (env
  `FUSION_SPEECH_TTS_BUFFER_S`) — bounded-queue cap in seconds.
- `tts_output_device: int | str | None` → `None` (env
  `FUSION_SPEECH_TTS_OUTPUT_DEVICE`) — output device, None = system default.

### Dependencies

No new dependency. `sounddevice` and `numpy` are already used by M2
(`audio_source.py`). `httpx` via `mlx_bridge`. No `pyproject.toml` change.

## Files

| File | Change |
|---|---|
| `fusion_speech/tts_playback.py` | **NEW** — `StreamingPlayer` (fetch→resample→bounded queue→OutputStream, cancel, fail-soft) |
| `fusion_speech/server.py` | `_player`, `speaking`, `speak` (mic guard), `stop_tts`, stop-cleanup |
| `fusion_speech/ipc/methods.py` | `handle_speak`, `handle_stop_tts`; register; status `speaking` |
| `fusion_speech/config.py` | 2 TTS playback config fields |
| `tests/test_tts_playback.py` | **NEW** — fake synthesize async-gen + fake OutputStream: TTFA, resample, bounded backpressure, cancel, fail-soft |
| `tests/test_tts_rpc.py` | **NEW** — `speech.speak`/`stop_tts` round-trip; mic guard (pause wake, stop/restart ASR); `speech.tts` events on subscriber queue |
| `README.md` | RPC table + `speech.speak`/`stop_tts`, env vars, M4 status |
| `docs/ARCHITECTURE.md` | module table M4→done; §5 methods (M4 implemented); §8 rewrite (collapsed API, mic guard, fail-soft); §9 runtime state (`_player`/`speaking`); §12 milestone M4→done |

## Verification

1. **Unit (no model, no hardware):**
   `pytest tests/test_tts_playback.py tests/test_tts_rpc.py -q`. Fake `synthesize`
   async-gen yields a synthetic WAV header + PCM chunks; fake OutputStream
   captures written frames. Assert: output stream started on first chunk (TTFA),
   resample 24k→16k applied on rate mismatch (and skipped when matched), bounded
   queue blocks the fetch task when full (backpressure), `cancel()` drops pending
   + closes the stream, fail-soft returns `{played:false}` on device error.
   RPC tests assert `speech.speak`/`stop_tts` round-trip, mic guard (wake paused
   during speak, ASR stopped-then-restarted), and `speech.tts` events on the
   subscriber queue.
2. **Real-load checkpoint** (project rule: really load model): start fusion-mlx
   (`~/claude-home/fusion-mlx/start.sh start`), `StreamingPlayer` fetches a short
   phrase via the real `synthesize_stream` (real Kokoro TTS), injects a fake
   OutputStream (keeps headless, no real speaker), asserts chunks arrive + TTFA +
   `done` event. Skips if mlx unreachable. Clean up test scratch; keep logs.
3. **No regressions:** `pytest tests/ -q` all green (M0–M3 + M4). Target ~80+ tests.
4. **Ruff:** `ruff check fusion_speech tests` + `ruff format` clean.
5. **Docs:** README + ARCHITECTURE M4 sections updated to reflect the collapsed
   API (not the old `synthesize_stream`/`tts_chunk`/`/stream/tts` sketch).

## Out of scope (deferred)

- `/stream/tts` separate WebSocket (not needed — `/stream/stt` carries `speech.tts`).
- Stream-id multiplexing (single-speaker daemon).
- Client-side PCM rendering (decided against — daemon plays to speaker).
- AEC/echo cancellation (the mic guard sidesteps this; true AEC is a later hardening).
- fusion-studio Swift integration (M5).
