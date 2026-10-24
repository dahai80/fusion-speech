# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

fusion-speech: local-first low-latency speech daemon (streaming STT/TTS/wake) for the Fusion ecosystem. **No model weights in this process** — all inference delegates to fusion-mlx over HTTP (`localhost:11434`, `/v1/audio/transcriptions` + `/v1/audio/speech`). fusion-speech owns only the realtime layers: mic capture, VAD, streaming ASR, hotword wake, streaming TTS playback.

Two lanes:

- **HTTP** on `127.0.0.1:11465` — `/health`, `/metrics` (Prometheus), `/stream/stt` WebSocket (event push)
- **UDS JSON-RPC 2.0** at `~/.fusion-speech/run/fusion-speech.sock` (newline-delimited, peer-UID verified fail-closed) — the 12 `speech.*` RPC methods

`fusion-core` (in-tree shared base) is a hard prerequisite: `fusion_core.mlx_client.FusionMLXClient`, `fusion_core.http.app.create_app`, `fusion_core.config`, `fusion_core.logging`.

## Commands

```bash
cd /Users/dahai/fusion && source .venv/bin/activate   # REQUIRED — shared monorepo venv
cd fusion-speech

# install (first time)
pip install -e fusion-core
pip install -e ".[test]"

# test (offline, no model load, no live daemon)
FUSION_MLX_API_KEY=fg-admin-key pytest tests/ -q       # verify key first — it rotates
pytest tests/test_foo.py::test_bar -v                 # single test

# lint (PR must be clean)
ruff check fusion_speech tests
ruff format .

# real-load gate (starts REAL daemon on 11465 + REAL fusion-mlx; opt-in marker)
~/fusion/fusion-mlx/start.sh start
FUSION_SPEECH_REAL_LOAD=1 FUSION_MLX_API_KEY=fg-admin-key \
  pytest tests/test_m6_real_load_integration.py -m real_load -v

# daemon lifecycle
./start.sh start | status | doctor | log -f | stop   # stop grace is 15s (drain)
```

Notes:

- `asyncio_mode = "auto"` — async tests need no `@pytest.mark.asyncio`.
- fusion-mlx API keys rotate — the env var value goes stale with no code change. When tests 401 against fusion-mlx, verify the live key with `curl -H "Authorization: Bearer <key>" http://localhost:11434/v1/models` and export it explicitly (working key as of 2026-10-24: `fg-admin-key`; `dahai168` now 401s).
- Model downloads use mirror `HF_MIRROR=https://hf-mirror.com`; cache `~/.fusion-mlx/models`.
- Inference-path changes (STT/TTS bridge, streaming, playback) must be verified real-load — never mock model output and claim done.

## Architecture

```
fusion-studio ──JSON-RPC/UDS──▶ fusion-speech ──HTTP──▶ fusion-mlx
              ◀── notifications      mic│VAD│wake         (weights)
```

`SpeechDaemon` (`server.py`) is a thin facade over 5 leaf services:

- `fusion_speech/server.py` — facade: FastAPI app build, UDS server, wake→ASR handshake (serialized by `_handshake_lock`), audio-lane probe loop
- `fusion_speech/components/` — `AsrService` (streaming listener), `WakeService` (openwakeword), `TtsService` (streaming playback), `AudioBus` (sole mic owner — acquire/release), `EventBus` (WS fan-out, bounded per-subscriber queues, drop-oldest)
- `fusion_speech/ipc/` — `jsonrpc.py` (dispatcher, peer-UID verify, 1 MiB line cap), `methods.py` (all 12 RPC handlers + `build_methods()` registry)
- `fusion_speech/mlx_bridge.py` — HTTP delegation with retry (5xx/transport only, backoff 0.5/1/2s; 4xx never retried)
- `fusion_speech/stt_stream.py` — `StreamingListener`: ring → VAD → segment → transcribe → `speech.asr` frames
- `fusion_speech/wake_word.py` — openwakeword ONNX; score keys matched by prefix (NOT `splitext` — strips `.1` suffix, F1 bug)
- `fusion_speech/vad.py` — energy RMS + Silero ONNX; the 64-sample rolling context prepend is REQUIRED (without it probs collapse)
- `fusion_speech/audio_source.py` + `ring_buffer.py` — sounddevice 16 kHz mono capture into lock-free SPSC numpy ring (drop-oldest on overrun; realtime callback must not alloc or log — PortAudio thread)
- `fusion_speech/config.py` — all env vars + `__post_init__` fail-loud validation
- `fusion_speech/metrics.py` — zero-dependency Prometheus text exposition (no `prometheus_client`)
- `fusion_speech/state.py` — persistent `~/.fusion-speech/run/state.json` (restart count, `speech.restarted` broadcast on crash-restart)

Adding an RPC method: `async def handle_xxx(params, daemon) -> dict` in `methods.py`, register in `build_methods()`, bounds-check inputs there (`MAX_AUDIO_B64` 25 MB, `MAX_TEXT` 5000 chars), raise `ValueError` → maps to `-32000`. Add a test.

Full layer breakdown: `docs/ARCHITECTURE.md`. Client integration recipes: `docs/INTEGRATION_GUIDE.md`. Dev workflow: `docs/CONTRIBUTING.md`.

## Hard contracts — do not break

- **Sample rate is 16000, mono** — VAD/wake/openwakeword contract. Non-16000 fails startup by design (config `__post_init__`), never "support" other rates silently.
- **Wake frame is exactly 1280 samples** (openwakeword 80 ms contract). Non-1280 fails startup. Variable-frame reads corrupt probabilities (R7).
- **Wake score keys matched by prefix** against openwakeword `class_mapping`/`models` keys (e.g. `hey_jarvis_v0.1`).
- **Peer-UID verification is fail-closed** — reject when peer creds missing.
- **Wake paused during ASR** (no self-trigger); mic guard during TTS speak (stop+restart ASR, pause wake). Mic ownership only via `AudioBus`.
- **Fail-soft vs fail-loud**: missing openwakeword/ONNX/VAD model degrades the feature (`enabled=False`, daemon keeps running); bad config fails startup loudly. Match this split for new features.

## Upstream workflow (hard rule)

Only `fusion-speech/` is in scope. Never modify fusion-mlx, fusion-studio, fusion-core, or other sub-projects. Upstream-rooted problem: file issue (English) → PR on upstream → wire consume-side here only after merge. Never locally patch an upstream file. Blocked test: `skip` (not `xfail`) citing the issue, so it flips to a real assertion when the fix lands.

Open upstream items:

- **fusion-mlx #668** — Whisper `non_speech_tokens` IndexError makes `/v1/audio/transcriptions` return 500 for every model. Blocks the `speech.transcribe` real-load step (currently skipped).
- **fusion-studio #337** — requests Swift `SpeechBridge` (UDS client) + mic permission. Swift side is upstream scope.

Port `11465` is registered in `architecture/port-registry.yaml`.

## Style

4-space (multiples of 4) indent, no docstrings, logging on every non-trivial path (`logging.getLogger(__name__)`, never print), comments explain why not what. Conventional Commits, English, scope usually `speech`. Commits land on `main` directly; tags are plain versions (no `v` prefix). README.md English only, README_CN.md Chinese.

Tests: keep only final artifacts (`tests/fixtures/`) and logs; delete scratch/process data after runs. Helpers: `tests/_uds_client.py` (JSON-RPC client, 4 MiB readline), `tests/conftest.py` (`real_daemon` session fixture).
