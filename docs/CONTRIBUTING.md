---
doc_id: fusion-speech/guide
product: fusion-speech
class: app
entry: service
doc_type: guide
title: { zh: "贡献指南", en: "Contributing Guide" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - docs/CONTRIBUTING.md
---

# Contributing to fusion-speech

> Developer setup, test, style, and the upstream issue→PR workflow.

fusion-speech is a local-first asyncio speech daemon. It delegates model
inference to fusion-mlx and owns the realtime layers (mic capture, VAD, wake,
streaming ASR/TTS). This guide is for contributors extending or fixing the
daemon itself. For integrating the daemon into an app as a client, see
[`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) instead.

## Contents

1. [Environment setup](#1-environment-setup)
2. [Lifecycle + logs](#2-lifecycle--logs)
3. [Testing](#3-testing)
4. [Lint](#4-lint)
5. [Code style](#5-code-style)
6. [Architecture entry points](#6-architecture-entry-points)
7. [Upstream workflow (hard rule)](#7-upstream-workflow-hard-rule)
8. [Commit convention](#8-commit-convention)
9. [Model-backed testing](#9-model-backed-testing)
10. [Release process](#10-release-process)

<!-- skeleton — sections filled below -->

---

## 1. Environment setup

The repo shares one `.venv/` at the fusion monorepo root. Python 3.12.

```bash
cd ~/fusion
source .venv/bin/activate
pip install -e fusion-core         # in-tree shared base (LLM client, FastAPI factory, config, logging)
pip install -e ".[test]"           # from fusion-speech/ — installs test extras (pytest, ruff, ...)
```

`fusion-core` is a hard prerequisite: fusion-speech imports
`fusion_core.mlx_client.FusionMLXClient`, `fusion_core.http.app.create_app`,
`fusion_core.config`, and `fusion_core.logging`. Without it installed editable,
imports fail at startup.

All work happens inside `fusion-speech/`. Do not modify other sub-projects
(see [Upstream workflow](#7-upstream-workflow-hard-rule)).

---

## 2. Lifecycle + logs

The daemon needs fusion-mlx (the inference engine) up first — it owns the
model weights and the `/v1/audio/*` endpoints on `localhost:11434`.

```bash
# inference engine
~/fusion/fusion-mlx/start.sh start
~/fusion/fusion-mlx/start.sh status
~/fusion/fusion-mlx/start.sh stop

# the speech daemon (127.0.0.1:11465 + UDS)
./start.sh start
./start.sh status
./start.sh doctor          # health: port + socket + fusion-mlx audio-lane probe
./start.sh log -f          # tail logs
./start.sh stop
```

The daemon reads the fusion-mlx bearer key from `FUSION_MLX_API_KEY` (or
fusion-mlx's `settings.json`). If `./start.sh doctor` reports an unreachable
audio lane, set `FUSION_MLX_API_KEY=<your-key>` for local dev. Note: a stale
`<your-key>` may leak from `~/.zshrc` and 401 against fusion-mlx — export
the working key explicitly in that case.

Logs go to `~/.fusion-speech/logs/` (`FUSION_SPEECH_LOG_DIR`). Runtime state
(start time, restart count) persists at `~/.fusion-speech/run/state.json`.

---

## 3. Testing

Unit + adversarial gate suite run fully offline — no model load, no live
daemon, no network:

```bash
cd ~/fusion && source .venv/bin/activate
cd fusion-speech
FUSION_MLX_API_KEY=<your-key> pytest tests/ -q     # 148 passed, 5 skipped
```

The 5 skipped tests carry the `real_load` marker — they start the real daemon
(real port 11465 + real socket) backed by a real fusion-mlx, so they are
opt-in and slow. Single test:

```bash
pytest tests/test_foo.py::test_bar -v
```

Real-load (start fusion-mlx first, then opt in):

```bash
~/fusion/fusion-mlx/start.sh start
FUSION_SPEECH_REAL_LOAD=1 FUSION_MLX_API_KEY=<your-key> \
  pytest tests/test_m6_real_load_integration.py -m real_load -v
```

Test conventions (from `pyproject.toml`):

- `asyncio_mode = "auto"` — async tests need no `@pytest.mark.asyncio`.
- `testpaths = ["tests"]` — run from the project root.
- Markers: `real_load` (registered) gates the live integration gate.
- **Clean up process data:** tests keep only final artifacts (`tests/fixtures/`)
  and logs; scratch/repro files created mid-run are removed. No `/tmp` litter.

Helpers: `tests/_uds_client.py` (newline-delimited JSON-RPC 2.0 client, 4 MiB
readline), `tests/conftest.py` (`real_daemon` session fixture that spawns the
real daemon and strips per-test isolation vars).

Known upstream skip: `speech.transcribe` real-load step skips on fusion-mlx
STT 500 (issue #668) — turns into a real assertion on the fix.

---

## 4. Lint

Ruff config lives in `pyproject.toml` (`[tool.ruff]` + `[tool.ruff.lint]`):
target `py312`, line length 120, selects `E F W I UP B SIM`, ignores `E501`,
excludes `*.md`.

```bash
ruff check fusion_speech tests       # lint
ruff format .                        # format
```

A PR must be `ruff check` clean. Run both before committing.

---

## 5. Code style

- **Indentation:** 4 spaces, multiples of 4 (no 5/9/11-space alignment hacks).
- **No docstrings.** Comments explain *why*; the code shows *what*. A function
  name + a short inline comment beat a docstring.
- **Logging by default.** Every non-trivial path logs (info on lifecycle,
  warning on fallback, exception on caught errors). Use the module logger:
  `logger = logging.getLogger(__name__)`. Do not print.
- **Match existing patterns.** Read adjacent files before writing new ones. Two
  patterns are worse than one — prefer the established one even if yours is
  better.
- **Surgical changes.** Touch only what the task requires. Do not "improve"
  adjacent code, reformat untouched lines, or refactor what is not broken.
- **Fail visibly.** If something goes wrong, raise/log loudly — no silent
  fallback that hides a real failure. The config layer fails startup loud
  (`__post_init__` validation) rather than corrupting VAD/wake silently.

---

## 6. Architecture entry points

Start with [`docs/architecture.md`](architecture.md) — the full layer breakdown,
the fusion-\* integration map (§10), security model (§11), and milestone plan
(§12). §3a documents the component split.

Code layout:

- `fusion_speech/server.py` — `SpeechDaemon`: thin facade. Builds the FastAPI
  app (`/health`, `/metrics`, `/stream/stt` WebSocket), owns the UDS server,
  the wake→ASR handshake, and the audio-lane probe loop.
- `fusion_speech/components/` — 5 leaf services behind the facade:
  `AsrService` (streaming listener), `WakeService` (openwakeword), `TtsService`
  (streaming playback), `AudioBus` (mic ownership), `EventBus` (WS fan-out).
- `fusion_speech/ipc/` — `jsonrpc.py` (dispatcher + peer-UID verify + line
  cap), `methods.py` (the 12 RPC handlers + `build_methods` registry).
- `fusion_speech/mlx_bridge.py` — HTTP delegation to fusion-mlx
  (`transcribe_stream`, `synthesize_full`, `synthesize_stream`, retry logic).
- `fusion_speech/stt_stream.py` — `StreamingListener`: ring → VAD → segment →
  transcribe → `speech.asr` frames.
- `fusion_speech/config.py` — `SpeechConfig`: all env vars, `__post_init__`
  fail-loud validation (sample_rate==16000, channels==1, wake_chunk==1280).
- `fusion_speech/metrics.py` — zero-dependency Prometheus Counter/Gauge/Histogram
  + text exposition.

Adding an RPC method: write an `async def handle_xxx(params, daemon) -> dict`
in `methods.py`, register it in `build_methods()`. Bounds-check inputs there
(see `MAX_AUDIO_B64` / `MAX_TEXT`); raise `ValueError` → maps to `-32000`.
Add a test under `tests/`.

---

## 7. Upstream workflow (hard rule)

**Only `fusion-speech/` is in scope.** Do not modify fusion-mlx, fusion-studio,
fusion-core, or any other sub-project in this repo. When a problem roots in an
upstream project, follow the issue-first flow:

1. **File an issue** on the upstream repo (English for GitHub ops), with a
   minimal repro and the exact file:line.
2. **Open a PR** on the upstream repo with the fix.
3. **Land code here only after the upstream PR merges** — wire the consuming
   side in fusion-speech to the now-fixed upstream contract.

Do not patch an upstream file locally to unblock a downstream test. If a test
is blocked upstream, mark it `skip` (not `xfail`) with a clear message citing
the upstream issue, so it flips to a real assertion the moment the fix lands.

Currently open upstream items (consume-side handled in fusion-speech):

- **fusion-mlx #668** — `mlx-audio` Whisper `non_speech_tokens` `IndexError`
  makes `/v1/audio/transcriptions` return 500 for every model. Blocks the
  `speech.transcribe` real-load step (skipped). TTS unaffected.
  https://github.com/dahai80/fusion-mlx/issues/668
- **fusion-studio #337** — requests a `SpeechBridge` Swift class (UDS JSON-RPC
  client) + mic permission + `/stream/stt` event subscription, so fusion-studio
  can consume fusion-speech. Swift implementation is upstream scope.
  https://github.com/dahai80/fusion-studio/issues/337

---

## 8. Commit convention

Conventional Commits, English (GitHub operations default English):

```
<type>(<scope>): <subject>

<body — why, not what>
```

Types: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`, `perf`. Scope is
usually `speech`. Subject ≤ ~72 chars, imperative mood. Body explains the
motivation and any non-obvious tradeoff. Reference upstream issues/PRs by
number (`#668`, `fusion-studio#337`).

Examples: `feat(speech): add speech.reprobe RPC`, `fix(wake): match score keys
by prefix not splitext (F1)`, `docs(speech): bilingual README_CN + integration
guide`.

Commit on `main` directly for this project is the established flow (the RC and
prior milestones landed as main commits + tags). Keep commits atomic — one
logical change per commit.

---

## 9. Model-backed testing

When a change touches the inference delegation path (STT/TTS bridge, streaming
transcribe, TTS playback), the real-load suite must actually load models — no
mocking the model output.

- Start/stop fusion-mlx with `~/fusion/fusion-mlx/start.sh start|stop`.
- If a model is missing, download via the mirror **`https://hf-mirror.com`**
  (set `HF_MIRROR=https://hf-mirror.com`), not the upstream HF endpoint.
- Model cache: `~/.fusion-mlx/models`.
- Run the real-load gate: `FUSION_SPEECH_REAL_LOAD=1 FUSION_MLX_API_KEY=<your-key>
  pytest tests/test_m6_real_load_integration.py -m real_load -v`.

A unit test that passes against a stub for the wrong reason is worse than no
test — it creates false confidence. If you cannot run the real model, leave the
test `real_load`-gated and document the skip; do not replace the real
assertion with a mocked one.

---

## 10. Release process

Patch/minor releases follow the same flow as the RC:

1. Bump `version` in `pyproject.toml` (e.g. `0.1.0rc1` → `0.1.1`).
2. Update `README.md` / `README_CN.md` Status section with a one-line release
   note if behavior changed; pure-docs patches note the doc additions.
3. Commit: `docs(speech): ... release 0.1.1` (or `chore(speech): release 0.1.1`).
4. Tag: `git tag 0.1.1` (plain version, no `v` prefix — matches `0.1.0rc1`).
5. Push: `git push origin main` then `git push origin 0.1.1`.
6. GitHub release (English): `gh release create 0.1.1 --repo dahai80/fusion-speech
   --title "0.1.1 — <summary>" --notes "..."`. Add `--prerelease` only for an
   `rc` tag; a plain version is a stable release.

Release notes list what changed, the version transition (e.g. `0.1.0rc1 →
0.1.1`), and any known upstream limits (e.g. issue #668 STT 500). Verify after:
`gh release view 0.1.1 --repo dahai80/fusion-speech`, `git tag | grep 0.1.1`.

Rollback: `git revert <release-commit>`, `git tag -d 0.1.1`,
`git push origin :refs/tags/0.1.1`, `gh release delete 0.1.1`.
