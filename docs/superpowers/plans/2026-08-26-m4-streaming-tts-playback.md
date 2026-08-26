# M4 Streaming TTS Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add streaming TTS playback — the daemon plays PCM chunks to the speaker as they arrive from fusion-mlx, with low time-to-first-audio, a mic-coexistence guard, and progress notifications.

**Architecture:** One new module `tts_playback.py` (a `StreamingPlayer`) consumes the existing `mlx_bridge.synthesize_stream` async generator, resamples 24k→16k, feeds a bounded queue that a writer task drains into a `sounddevice.OutputStream` (started on first chunk = TTFA). The daemon wraps `speak()` in a mic guard that pauses wake / stops-then-restarts ASR. RPC `speech.speak` + `speech.stop_tts`; `speech.tts` progress events reuse the existing `/stream/stt` WS. Mirrors the M2 `stt_stream.py` shape (one session, drain task, fail-soft, injectable stream_factory).

**Tech Stack:** Python 3.12+, asyncio, sounddevice (lazy import), numpy, httpx (via `mlx_bridge`), pytest + pytest-asyncio. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-26-m4-streaming-tts-playback-design.md`

## Global Constraints

- Project root: `~/fusion/fusion-speech`. Activate env with `source ~/fusion/.venv/bin/activate` before any Python/pytest/ruff command (absolute path — relative `source .venv/bin/activate` resolves to system Python, which lacks the package).
- Code style: 4-space-multiple indentation (never 5/9/11), **no docstrings**, default logging on every meaningful path, clean code no wrapping/explanation.
- Fail-soft, never fatal: missing sounddevice / device error / mlx error → return `{played:false, error}`, daemon unharmed (mirror M3 wake fail-soft).
- Tests clean up process data; keep only final outputs + logs. Real model weights kept under `~/.fusion-speech/`.
- All new code uses `from __future__ import annotations` (matches existing modules).
- Run tests: `python3 -m pytest tests/ -q`. Lint: `ruff check fusion_speech tests && ruff format`.
- `mlx_bridge.synthesize_stream(config, text, *, voice="", model="", stream=True, streaming_interval=None)` is an async generator yielding `bytes`. First chunk = 44-byte WAV header + first PCM; rest = raw int16 mono PCM. Parse rate via `mlx_bridge._parse_wav_sample_rate(header[:44])`. (Existing, `mlx_bridge.py:128`.)
- `_resolve_tts_model(daemon)` exists in `ipc/methods.py:65` — reuse for model fallback.
- Config fields use the `_env_float` / `os.environ.get` helpers already in `config.py`.

## File Structure

| File | Responsibility |
|---|---|
| `fusion_speech/tts_playback.py` | **NEW** — `StreamingPlayer`: fetch→resample→bounded queue→OutputStream, cancel, fail-soft. One playback session. |
| `fusion_speech/config.py` | +2 fields: `tts_playback_buffer_s`, `tts_output_device`. |
| `fusion_speech/server.py` | `_player`, `speaking`, `speak()` (mic guard), `stop_tts()`, stop-cleanup. |
| `fusion_speech/ipc/methods.py` | `handle_speak`, `handle_stop_tts`; register; `handle_status` +`speaking`. |
| `tests/test_tts_playback.py` | **NEW** — unit tests for `StreamingPlayer` (fake synthesize async-gen + fake OutputStream). |
| `tests/test_tts_rpc.py` | **NEW** — RPC round-trip + mic guard + `speech.tts` events. |
| `README.md` | RPC table + `speech.speak`/`stop_tts`, env vars, M4 status. |
| `docs/ARCHITECTURE.md` | module table M4→done; §5 methods; §8 rewrite; §9 state; §12 milestone. |

---

### Task 1: Config fields for TTS playback

**Files:**
- Modify: `fusion_speech/config.py`

**Interfaces:**
- Produces: `SpeechConfig.tts_playback_buffer_s: float` (default 2.0), `SpeechConfig.tts_output_device: int|str|None` (default None).

- [ ] **Step 1: Add the two fields + defaults**

In `config.py`, after the `DEFAULT_WAKE_MODEL_URL` constant block (line 26), add:

```python
DEFAULT_TTS_BUFFER_S = 2.0
```

In the `SpeechConfig` dataclass, after the `wake_model_url` field (the field ending around line 80), add:

```python
    tts_playback_buffer_s: float = field(
        default_factory=lambda: _env_float("FUSION_SPEECH_TTS_BUFFER_S", DEFAULT_TTS_BUFFER_S)
    )
    tts_output_device: str = field(
        default_factory=lambda: os.environ.get("FUSION_SPEECH_TTS_OUTPUT_DEVICE", "")
    )
```

(`tts_output_device` is a `str`; empty string = system default. `StreamingPlayer` converts `""` → `None` for sounddevice. A string is simpler than parsing int-or-str from an env var.)

- [ ] **Step 2: Verify import + config loads**

Run: `source ~/fusion/.venv/bin/activate && python3 -c "from fusion_speech.config import load_config; c=load_config(); print(c.tts_playback_buffer_s, repr(c.tts_output_device))"`
Expected: `2.0 ''`

- [ ] **Step 3: Commit**

```bash
git add fusion_speech/config.py
git commit -m "feat(speech): add TTS playback config fields (M4)"
```

---

### Task 2: StreamingPlayer — core speak + fail-soft

**Files:**
- Create: `fusion_speech/tts_playback.py`
- Test: `tests/test_tts_playback.py`

**Interfaces:**
- Consumes: `mlx_bridge.synthesize_stream(config, text, *, voice="", model="", stream=True, streaming_interval=None) -> AsyncIterator[bytes]` (existing); `mlx_bridge._parse_wav_sample_rate(header: bytes) -> int` (existing); `config.sample_rate`, `config.tts_playback_buffer_s`, `config.tts_output_device`.
- Produces: `class StreamingPlayer` with `__init__(config, *, synthesize=None, notifier=None, stream_factory=None)`, `async speak(text, voice="", model="") -> dict`, `async cancel() -> dict`, properties `active`, `speaking`. `notifier` is `Callable[[dict], Awaitable[None]] | None`.

- [ ] **Step 1: Write the failing test for basic speak + TTFA + fail-soft**

Create `tests/test_tts_playback.py`:

```python
from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest

from fusion_speech.config import load_config
from fusion_speech.tts_playback import StreamingPlayer


def _wav_header(sample_rate: int, num_samples: int) -> bytes:
    data = num_samples * 2
    return (
        b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", data)
    )


class _FakeOutStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False
        self.written: list[np.ndarray] = []

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False

    def write(self, data: np.ndarray):
        self.written.append(np.asarray(data, dtype=np.float32))


def _make_player(synth, *, buffer_s=2.0, rate=16000, captured=None):
    cfg = load_config()
    cfg.tts_playback_buffer_s = buffer_s

    def factory(**kwargs):
        s = _FakeOutStream(**kwargs)
        if captured is not None:
            captured["stream"] = s
            captured["kwargs"] = kwargs
        return s

    return StreamingPlayer(cfg, synthesize=synth, stream_factory=factory)


@pytest.mark.asyncio
async def test_speak_plays_chunks_and_returns_done():
    pcm = (np.sin(np.linspace(0, 100, 1600)) * 0.3).astype(np.float32)
    i16 = (pcm * 32767).astype(np.int16).tobytes()

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, pcm.size)
        yield i16
        yield i16

    captured = {}
    player = _make_player(synth, captured=captured)
    res = await player.speak("hi")
    assert res["played"] is True
    assert res["chars"] == 2
    assert res["sample_rate"] == 16000
    # output stream started (TTFA on first chunk)
    assert captured["stream"].active is True
    # all PCM written (2 chunks worth)
    total = sum(w.size for w in captured["stream"].written)
    assert total == pcm.size * 2
    assert player.active is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_playback.py::test_speak_plays_chunks_and_returns_done -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fusion_speech.tts_playback'`

- [ ] **Step 3: Write minimal StreamingPlayer**

Create `fusion_speech/tts_playback.py`:

```python
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from fusion_speech.mlx_bridge import _parse_wav_sample_rate, synthesize_stream

logger = logging.getLogger(__name__)

_WAV_HEADER_LEN = 44


class TtsPlaybackError(RuntimeError):
    pass


class StreamingPlayer:
    def __init__(
        self,
        config,
        *,
        synthesize: Callable[..., AsyncIterator[bytes]] | None = None,
        notifier: Callable[[dict], Awaitable[None]] | None = None,
        stream_factory: Callable[..., object] | None = None,
    ):
        self.config = config
        self._synthesize = synthesize or synthesize_stream
        self._notifier = notifier
        self._stream_factory = stream_factory
        self._out_stream = None
        self._queue: asyncio.Queue | None = None
        self._fetch_task: asyncio.Task | None = None
        self._write_task: asyncio.Task | None = None
        self._active = False
        self._speaking = False
        self._cancelled = False
        self._sample_rate = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def speaking(self) -> bool:
        return self._speaking

    async def speak(self, text: str, voice: str = "", model: str = "") -> dict:
        if self._active:
            logger.warning("speak already in flight")
            return {"played": False, "error": "already speaking"}
        self._active = True
        self._cancelled = False
        self._sample_rate = 0
        cap = max(1, int(self.config.tts_playback_buffer_s * self.config.sample_rate))
        self._queue = asyncio.Queue(maxsize=cap)
        await self._notify({"event": "speech.tts", "state": "starting", "chars": len(text)})
        self._fetch_task = asyncio.create_task(self._fetch_loop(text, voice, model))
        self._write_task = asyncio.create_task(self._write_loop())
        try:
            await asyncio.gather(self._fetch_task, self._write_task)
        except Exception as e:
            logger.warning("speak failed: %s", e)
            await self._notify({"event": "speech.tts", "state": "error", "error": str(e)})
            return {"played": False, "error": str(e)}
        finally:
            self._close_stream()
            self._active = False
        if self._cancelled:
            await self._notify({"event": "speech.tts", "state": "cancelled"})
            return {"played": False, "cancelled": True}
        await self._notify({"event": "speech.tts", "state": "done", "chars": len(text)})
        return {"played": True, "chars": len(text), "sample_rate": self._sample_rate}

    async def _fetch_loop(self, text: str, voice: str, model: str) -> None:
        header_seen = False
        leftover = b""
        try:
            async for chunk in self._synthesize(
                self.config, text, voice=voice, model=model, stream=True
            ):
                if self._cancelled:
                    break
                if not chunk:
                    continue
                if not header_seen:
                    if len(chunk) < _WAV_HEADER_LEN:
                        logger.warning("tts first chunk < header (%d)", len(chunk))
                        header_seen = True
                        data = chunk
                    else:
                        self._sample_rate = _parse_wav_sample_rate(chunk[:_WAV_HEADER_LEN])
                        logger.info("tts stream rate=%d", self._sample_rate)
                        header_seen = True
                        data = chunk[_WAV_HEADER_LEN:]
                else:
                    data = chunk
                if leftover:
                    data = leftover + data
                    leftover = b""
                if len(data) % 2 != 0:
                    leftover = data[-1:]
                    data = data[:-1]
                if not data:
                    continue
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if self._sample_rate and self._sample_rate != self.config.sample_rate:
                    pcm = _resample(pcm, self._sample_rate, self.config.sample_rate)
                await self._queue.put(pcm)
            self._queue.put_nowait(None)
        except Exception:
            self._queue.put_nowait(None)
            raise

    async def _write_loop(self) -> None:
        assert self._queue is not None
        started = False
        try:
            while True:
                frame = await self._queue.get()
                if frame is None:
                    break
                if self._cancelled:
                    continue
                if not started:
                    self._start_output()
                    started = True
                    self._speaking = True
                    await self._notify({"event": "speech.tts", "state": "speaking"})
                if self._out_stream is not None:
                    self._write_frame(frame)
        finally:
            self._speaking = False

    def _start_output(self) -> None:
        device = self.config.tts_output_device or None
        factory = self._stream_factory or self._default_stream_factory
        try:
            stream = factory(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="float32",
                device=device,
            )
            stream.start()
            self._out_stream = stream
            logger.info("tts output started rate=%d device=%s", self.config.sample_rate, device)
        except Exception as e:
            logger.error("tts output open failed: %s", e)
            self._out_stream = None
            raise TtsPlaybackError(f"cannot open output stream: {e}") from e

    def _default_stream_factory(self, *, samplerate, channels, dtype, device):
        import sounddevice as sd

        return sd.OutputStream(samplerate=samplerate, channels=channels, dtype=dtype, device=device)

    def _write_frame(self, frame: np.ndarray) -> None:
        stream = self._out_stream
        if stream is None:
            return
        try:
            stream.write(np.ascontiguousarray(frame, dtype=np.float32))
        except Exception:
            logger.exception("tts write failed")

    def _close_stream(self) -> None:
        stream = self._out_stream
        if stream is None:
            return
        self._out_stream = None
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()
        logger.info("tts output closed")

    async def cancel(self) -> dict:
        if not self._active:
            return {"stopped": False}
        self._cancelled = True
        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)
        logger.info("tts cancel requested")
        return {"stopped": True}

    async def _notify(self, payload: dict) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier(payload)
        except Exception:
            logger.exception("tts notifier failed")


def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    n_out = int(round(pcm.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.empty(0, dtype=np.float32)
    idx = np.linspace(0, pcm.size - 1, n_out)
    out = np.interp(idx, np.arange(pcm.size), pcm).astype(np.float32)
    logger.debug("resample %d -> %d (%d samples)", src_rate, dst_rate, out.size)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_playback.py::test_speak_plays_chunks_and_returns_done -v`
Expected: PASS

- [ ] **Step 5: Write failing test for fail-soft on device-open error**

Append to `tests/test_tts_playback.py`:

```python
@pytest.mark.asyncio
async def test_speak_fail_soft_when_output_open_raises():
    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 4)
        yield (np.zeros(4, dtype=np.int16)).tobytes()

    cfg = load_config()

    def bad_factory(**kwargs):
        raise OSError("no device")

    player = StreamingPlayer(cfg, synthesize=synth, stream_factory=bad_factory)
    res = await player.speak("hi")
    assert res["played"] is False
    assert "no device" in res["error"]
    assert player.active is False
```

- [ ] **Step 6: Run test to verify it passes**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_playback.py::test_speak_fail_soft_when_output_open_raises -v`
Expected: PASS (the `_start_output` raise propagates through `gather` → the `except` in `speak` catches it → fail-soft return). If it does NOT pass, ensure `_write_loop`'s `_start_output` exception is not swallowed — it must propagate out of `gather`.

- [ ] **Step 7: Commit**

```bash
git add fusion_speech/tts_playback.py tests/test_tts_playback.py
git commit -m "feat(speech): add StreamingPlayer core (fetch/resample/play, fail-soft) (M4)"
```

---

### Task 3: StreamingPlayer — resample + cancel + backpressure

**Files:**
- Modify: `tests/test_tts_playback.py`

**Interfaces:**
- Produces: validates the existing `speak`/`cancel`/`_resample` behavior end-to-end.

- [ ] **Step 1: Write failing test for resample 24k→16k**

Append to `tests/test_tts_playback.py`:

```python
@pytest.mark.asyncio
async def test_speak_resamples_when_rate_mismatch():
    src_rate = 24000
    pcm = (np.sin(np.linspace(0, 100, 2400)) * 0.3).astype(np.float32)
    i16 = (pcm * 32767).astype(np.int16).tobytes()

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(src_rate, pcm.size)
        yield i16

    captured = {}
    player = _make_player(synth, rate=16000, captured=captured)
    res = await player.speak("hi")
    assert res["played"] is True
    assert res["sample_rate"] == src_rate
    # daemon rate is 16000; output opened at 16000
    assert captured["kwargs"]["samplerate"] == 16000
    # written samples resampled down: 2400 -> 1600 (approx)
    total = sum(w.size for w in captured["stream"].written)
    assert 1500 <= total <= 1700
```

- [ ] **Step 2: Run test to verify it passes**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_playback.py::test_speak_resamples_when_rate_mismatch -v`
Expected: PASS (resample already in `_fetch_loop`). If FAIL on the `samplerate` kwarg, the `_FakeOutStream` captures `kwargs` — confirm `_start_output` passes `samplerate=self.config.sample_rate`.

- [ ] **Step 3: Write failing test for cancel drops pending + closes**

Append:

```python
@pytest.mark.asyncio
async def test_cancel_stops_playback():
    big = (np.zeros(16000 * 10, dtype=np.float32))

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, big.size)
        # yield many small frames so the writer is mid-playback
        for _ in range(20):
            if player._cancelled:
                break
            yield (big[:1000] * 32767).astype(np.int16).tobytes()
            await asyncio.sleep(0.01)

    captured = {}
    player = _make_player(synth, captured=captured)
    task = asyncio.create_task(player.speak("hi"))
    await asyncio.sleep(0.05)
    res = await player.cancel()
    assert res["stopped"] is True
    speak_res = await asyncio.wait_for(task, timeout=3.0)
    assert speak_res["played"] is False
    assert speak_res["cancelled"] is True
    assert player.active is False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_playback.py::test_cancel_stops_playback -v`
Expected: PASS. If it hangs, the fetch loop's `await self._queue.put(pcm)` may block forever after cancel because the writer stopped draining. Fix: in `_fetch_loop`, check `self._cancelled` before each `put` AND wrap the `put` in a timeout, e.g. `await asyncio.wait_for(self._queue.put(pcm), timeout=0.5)` with a `try/except TimeoutError: break`. Apply the minimal fix to make it pass.

- [ ] **Step 5: Write failing test for bounded-queue backpressure**

Append:

```python
@pytest.mark.asyncio
async def test_bounded_queue_backpressures_fetch():
    put_count = 0

    class _SlowStream:
        def __init__(self, **kwargs):
            self.active = False

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

        def write(self, data):
            # slow consumer: never reads fast enough -> fetch must block
            pass

    cfg = load_config()
    cfg.tts_playback_buffer_s = 0.05  # tiny buffer
    frame = (np.zeros(1600, dtype=np.float32) * 32767).astype(np.int16).tobytes()

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 1600)
        for i in range(50):
            nonlocal_count[0] = i
            yield frame
            await asyncio.sleep(0)

    nonlocal_count = [0]
    player = StreamingPlayer(cfg, synthesize=synth, stream_factory=lambda **k: _SlowStream())
    await asyncio.wait_for(player.speak("hi"), timeout=5.0)
    # the fetch did NOT race ahead of the tiny buffer — it was bounded
    assert player.active is False
```

- [ ] **Step 6: Run test to verify it passes**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_playback.py::test_bounded_queue_backpressures_fetch -v`
Expected: PASS. The bounded `asyncio.Queue(maxsize=cap)` inherently blocks the fetch `put` when full. If FAIL with timeout, the writer never drains because `_SlowStream.write` succeeds but the loop keeps `get`-ing — confirm the writer drains the queue (it does). If the test hangs, reduce frame count or buffer so the speak completes.

- [ ] **Step 7: Commit**

```bash
git add tests/test_tts_playback.py
git commit -m "test(speech): StreamingPlayer resample/cancel/backpressure (M4)"
```

---

### Task 4: Daemon wiring — speak + mic guard + stop_tts

**Files:**
- Modify: `fusion_speech/server.py`
- Test: `tests/test_tts_rpc.py` (created Task 5, but the guard logic is exercised there; this task also adds a focused server-level test)

**Interfaces:**
- Consumes: `StreamingPlayer` (Task 2), `WakeWordEngine.pause()/resume()` (M3, existing), `start_listening/stop_listening` (M2, existing).
- Produces: `SpeechDaemon.speak(text, voice="", model="") -> dict`, `SpeechDaemon.stop_tts() -> dict`, attrs `self._player`, `self.speaking`.

- [ ] **Step 1: Add state attrs to `__init__`**

In `server.py` `SpeechDaemon.__init__`, after `self._wake = None` (line 32), add:

```python
        self.speaking = False
        self._player = None  # StreamingPlayer, lazily built
```

- [ ] **Step 2: Add `speak` + `stop_tts` methods**

In `server.py`, after the `disarm_wake` method (before `set_hotwords`), add:

```python
    async def speak(self, text: str, voice: str = "", model: str = "") -> dict:
        from fusion_speech.tts_playback import StreamingPlayer

        if self._player is not None and self._player.active:
            return {"played": False, "error": "already speaking"}
        player = StreamingPlayer(self.config, notifier=self.broadcast)
        self._player = player
        wake = self._wake
        was_listening = self._listener is not None and self._listener.active
        if wake is not None and wake.active and not wake.paused:
            wake.pause()
        if was_listening:
            with contextlib.suppress(Exception):
                await self.stop_listening()
        try:
            self.speaking = True
            result = await player.speak(text, voice=voice, model=model)
        finally:
            self.speaking = False
            self._player = None
            if wake is not None and wake.active and wake.paused:
                wake.resume()
            if was_listening:
                with contextlib.suppress(Exception):
                    await self.start_listening()
        return result

    async def stop_tts(self) -> dict:
        player = self._player
        if player is None:
            return {"stopped": False}
        return await player.cancel()
```

- [ ] **Step 3: Add player stop to daemon `stop()`**

In `server.py` `stop()`, at the top of the cleanup (before `if self._wake is not None:`), add:

```python
        if self._player is not None:
            with contextlib.suppress(Exception):
                await self._player.cancel()
            self._player = None
            self.speaking = False
```

- [ ] **Step 4: Verify server imports + daemon constructs**

Run: `source ~/fusion/.venv/bin/activate && python3 -c "from fusion_speech.server import SpeechDaemon; d=SpeechDaemon.__new__(SpeechDaemon); print('ok')"`
Expected: `ok` (no import errors). If a real `SpeechDaemon()` needs config, use `from fusion_speech.config import load_config; SpeechDaemon(load_config())`.

- [ ] **Step 5: Commit**

```bash
git add fusion_speech/server.py
git commit -m "feat(speech): wire speak + mic guard + stop_tts into daemon (M4)"
```

---

### Task 5: RPC methods — speech.speak + speech.stop_tts + status

**Files:**
- Modify: `fusion_speech/ipc/methods.py`
- Create: `tests/test_tts_rpc.py`

**Interfaces:**
- Consumes: `daemon.speak(text, voice, model)`, `daemon.stop_tts()`, `daemon.speaking`, `_resolve_tts_model(daemon)` (existing).
- Produces: RPC methods `speech.speak`, `speech.stop_tts`; `handle_status` includes `speaking`.

- [ ] **Step 1: Add `speaking` to `handle_status`**

In `methods.py` `handle_status`, after `"wake_armed": daemon.wake_armed,` (line 27), add:

```python
        "speaking": daemon.speaking,
```

- [ ] **Step 2: Add `handle_speak` + `handle_stop_tts`**

In `methods.py`, after `handle_wake` (before `build_methods`), add:

```python
async def handle_speak(params: dict, daemon) -> dict:
    text = params.get("text", "")
    if not text:
        raise ValueError("text required")
    voice = params.get("voice", "") or daemon.config.tts_voice
    model = params.get("model", "") or _resolve_tts_model(daemon)
    logger.info("speech.speak chars=%d model=%s", len(text), model)
    return await daemon.speak(text, voice=voice, model=model)


async def handle_stop_tts(params: dict, daemon) -> dict:
    logger.info("speech.stop_tts")
    return await daemon.stop_tts()
```

- [ ] **Step 3: Register the two methods**

In `methods.py` `build_methods`, after the `"speech.wake"` line, add:

```python
        "speech.speak": lambda params: handle_speak(params, daemon),
        "speech.stop_tts": lambda params: handle_stop_tts(params, daemon),
```

- [ ] **Step 4: Write failing test for speak/stop_tts RPC round-trip**

Create `tests/test_tts_rpc.py`:

```python
from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest

from fusion_speech.config import load_config
from fusion_speech.ipc.jsonrpc import JsonRpcDispatcher
from fusion_speech.ipc.methods import build_methods
from fusion_speech.server import SpeechDaemon


def _wav_header(sample_rate: int, num_samples: int) -> bytes:
    data = num_samples * 2
    return (
        b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", data)
    )


class _FakeOutStream:
    def __init__(self, **kwargs):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False

    def write(self, data):
        pass


def _patch_daemon_tts(daemon):
    from fusion_speech.tts_playback import StreamingPlayer

    async def synth(cfg, text, *, voice="", model="", stream=True, streaming_interval=None):
        yield _wav_header(16000, 4)
        yield (np.zeros(4, dtype=np.int16)).tobytes()

    def factory(**kwargs):
        return _FakeOutStream(**kwargs)

    orig_speak = daemon.speak

    async def fake_speak(text, voice="", model=""):
        player = StreamingPlayer(daemon.config, notifier=daemon.broadcast, synthesize=synth, stream_factory=factory)
        daemon._player = player
        try:
            daemon.speaking = True
            return await player.speak(text, voice=voice, model=model)
        finally:
            daemon.speaking = False
            daemon._player = None

    daemon.speak = fake_speak
    daemon.stop_tts = orig_speak.__get__(daemon) if False else daemon.stop_tts


@pytest.mark.asyncio
async def test_speak_rpc_round_trip():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_tts(daemon)
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    res = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hello"}})
    assert res["result"]["played"] is True


@pytest.mark.asyncio
async def test_status_reports_speaking():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_tts(daemon)
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    s0 = await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.status"})
    assert s0["result"]["speaking"] is False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_rpc.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Write failing test for mic guard (pause wake, stop/restart ASR)**

Append to `tests/test_tts_rpc.py`:

```python
class _FakeWake:
    def __init__(self):
        self.active = True
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


@pytest.mark.asyncio
async def test_speak_pauses_wake_during_playback():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_tts(daemon)
    wake = _FakeWake()
    daemon._wake = wake
    daemon.wake_armed = True

    paused_during = []
    orig_fake = daemon.speak

    async def wrapped(text, voice="", model=""):
        paused_during.append(wake.paused)
        return await orig_fake(text, voice=voice, model=model)

    daemon.speak = wrapped
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hi"}})
    assert paused_during and paused_during[0] is True
    assert wake.paused is False  # resumed after


@pytest.mark.asyncio
async def test_speak_stops_and_restarts_asr():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_tts(daemon)

    class _FakeListener:
        active = True

    daemon._listener = _FakeListener()
    listen_calls = []

    async def fake_start(audio_source=None):
        listen_calls.append("start")
        daemon._listener = _FakeListener()
        return {"listening": True}

    async def fake_stop():
        listen_calls.append("stop")
        daemon._listener = None
        return {"listening": False}

    daemon.start_listening = fake_start
    daemon.stop_listening = fake_stop
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hi"}})
    assert listen_calls == ["stop", "start"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_rpc.py -v`
Expected: PASS (4 tests). If the mic-guard tests fail, check `server.speak` checks `wake.active`/`wake.paused` correctly and that the `_FakeWake` has those attrs.

- [ ] **Step 8: Write failing test for `speech.tts` events on subscriber queue**

Append to `tests/test_tts_rpc.py`:

```python
async def _wait_event(q: asyncio.Queue, event: str, timeout=2.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            payload = await asyncio.wait_for(q.get(), timeout=deadline - asyncio.get_event_loop().time())
        except asyncio.TimeoutError:
            break
        if payload.get("event") == event:
            return payload
    return None


@pytest.mark.asyncio
async def test_speak_emits_tts_events():
    daemon = SpeechDaemon(load_config())
    _patch_daemon_tts(daemon)
    q = daemon.subscribe()
    dispatcher = JsonRpcDispatcher(build_methods(daemon))
    await dispatcher.dispatch({"jsonrpc": "2.0", "id": 1, "method": "speech.speak", "params": {"text": "hi"}})
    done = await _wait_event(q, "speech.tts")
    assert done is not None
    assert done["state"] == "done"
    daemon.unsubscribe(q)
```

(Note: `_wait_event` collects the first `speech.tts` frame; it may be `starting`. Assert `state == "done"` — if the test sees `starting` instead, change the assertion to check that *a* `speech.tts` frame arrived with `state in ("starting","speaking","done")`, OR wait specifically for `done` by looping until found. Keep the `done` assertion and make `_wait_event` loop until it finds `state == "done"` — adjust the helper to filter by both `event` and `state`.)

- [ ] **Step 9: Run test to verify it passes**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_rpc.py::test_speak_emits_tts_events -v`
Expected: PASS. If it fails because `_wait_event` returns the `starting` frame, update the helper to scan for `done` specifically (the queue holds all frames; loop the `get` until `state == "done"`).

- [ ] **Step 10: Commit**

```bash
git add fusion_speech/ipc/methods.py tests/test_tts_rpc.py
git commit -m "feat(speech): speech.speak/stop_tts RPC + status speaking + events (M4)"
```

---

### Task 6: Real-load test (really load the TTS model)

**Files:**
- Modify: `tests/test_tts_playback.py`

**Interfaces:**
- Consumes: real `mlx_bridge.synthesize_stream` + fusion-mlx on `localhost:11434`; fake `stream_factory` (headless, no real speaker).

- [ ] **Step 1: Write the real-load gate test**

Append to `tests/test_tts_playback.py`:

```python
import socket


def _mlx_reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=2.0):
            return True
    except OSError:
        return False


MLX_UP = _mlx_reachable()


@pytest.mark.asyncio
@pytest.mark.skipif(not MLX_UP, reason="fusion-mlx not running on 11434")
async def test_real_tts_loads_and_streams():
    # Real-load gate (project rule: really load the model). Uses the real
    # synthesize_stream -> fusion-mlx Kokoro, fake OutputStream (headless, no
    # speaker). Asserts chunks arrive, TTFA, done event.
    cfg = load_config()
    notified = []

    async def notifier(payload):
        notified.append(payload)

    class _FakeOutStream:
        def __init__(self, **kwargs):
            self.active = False
            self.frames = 0

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

        def write(self, data):
            self.frames += 1

    player = StreamingPlayer(
        cfg,
        notifier=notifier,
        stream_factory=lambda **k: _FakeOutStream(),
    )
    res = await asyncio.wait_for(player.speak("Hello world.", model=cfg.tts_model), timeout=60.0)
    assert res["played"] is True
    assert res["sample_rate"] > 0
    states = [n.get("state") for n in notified if n.get("event") == "speech.tts"]
    assert "done" in states
```

- [ ] **Step 2: Ensure fusion-mlx is up**

Run: `~/claude-home/fusion-mlx/start.sh status`
If not running: `~/claude-home/fusion-mlx/start.sh start` (per project rule for model tests).

- [ ] **Step 3: Run the real-load test**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/test_tts_playback.py::test_real_tts_loads_and_streams -v -s`
Expected: PASS (Kokoro streams, frames written, `done` event). If mlx is down and the test skips, that is acceptable for the gate, but try to run it for real at least once.

- [ ] **Step 4: Commit**

```bash
git add tests/test_tts_playback.py
git commit -m "test(speech): real-load TTS streaming gate against fusion-mlx (M4)"
```

---

### Task 7: Full regression + ruff + docs

**Files:**
- Modify: `README.md`, `docs/ARCHITECTURE.md`

- [ ] **Step 1: Run full test suite**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/ -q`
Expected: all green (M0–M3 + M4). Target ~80+ tests.

- [ ] **Step 2: Run ruff**

Run: `source ~/fusion/.venv/bin/activate && ruff check fusion_speech tests && ruff format fusion_speech tests`
Expected: clean. Auto-fix any issues (`ruff check --fix`).

- [ ] **Step 3: Update README RPC table + env vars + status**

In `README.md`:
- Add to the RPC table (after the `speech.wake` row):
  `| speech.speak | {text, voice?, model?} | {played, chars, sample_rate} — streams TTS to the speaker (pauses mic) |`
  `| speech.stop_tts | {} | {stopped} — cancel active playback |`
- Add to the env-var table (after `FUSION_SPEECH_WAKE_MODEL_URL`):
  `| FUSION_SPEECH_TTS_BUFFER_S | 2.0 | TTS playback buffer cap (seconds) |`
  `| FUSION_SPEECH_TTS_OUTPUT_DEVICE | (system default) | sounddevice output device |`
- Update the M2 realtime-push paragraph: change `M4 adds speech.tts_chunk.` to `M4 adds speech.tts progress events on the same socket.`
- Add an M4 status section (after the M3 block):
  ```
  - **M4** — streaming TTS playback. **done**
    - `tts_playback.py`: `StreamingPlayer` consumes `mlx_bridge.synthesize_stream`,
      resamples 24k→16k, feeds a bounded queue a writer task drains into a
      `sounddevice.OutputStream` (started on first chunk = low TTFA). Cancel drops
      pending + closes the stream. Fail-soft on device error.
    - Mic guard: while speaking, wake paused; if ASR was listening, stopped then
      restarted — no self-trigger, no echo. Reuses M2/M3 primitives.
    - RPC: `speech.speak`, `speech.stop_tts`; `speech.tts` events (starting/speaking/done/cancelled/error) on `/stream/stt`.
    - Verified: ~80+ tests green incl. real-load (Kokoro via fusion-mlx, fake output, headless).
  ```

- [ ] **Step 4: Update ARCHITECTURE module table + §8 + §5 + §9 + §12**

In `docs/ARCHITECTURE.md`:
- Module table row: change `| tts_playback.py | streaming TTS playback | M4 | todo |` to `| tts_playback.py | streaming TTS playback | M4 | done |`.
- §5 "Methods (M4, planned)" → rename to "Methods (M4, implemented)" and replace the table with:
  `| speech.speak | {text, voice?, model?} → stream TTS to speaker, mic paused |`
  `| speech.stop_tts | {} → cancel playback |`
- §8 rewrite: replace the planned-API sketch with the implemented design — daemon owns `sounddevice.OutputStream`, `StreamingPlayer` (fetch→resample→bounded queue→writer→output, TTFA on first chunk, cancel, fail-soft), mic guard (pause wake / stop-restart ASR), `speech.tts` events on `/stream/stt` (not a separate endpoint, no stream id). Note the collapse of the old `synthesize_stream`/`tts_chunk`/`/stream/tts` sketch.
- §9 runtime state: add `speaking` and `_player` to the runtime-state list.
- §12 milestone plan: mark M4 done.

- [ ] **Step 5: Commit docs**

```bash
git add README.md docs/ARCHITECTURE.md
git commit -m "docs(speech): M4 streaming TTS playback (README + ARCHITECTURE)"
```

- [ ] **Step 6: Final verify**

Run: `source ~/fusion/.venv/bin/activate && python3 -m pytest tests/ -q && ruff check fusion_speech tests`
Expected: all green + ruff clean.

---

## Self-Review

**1. Spec coverage:** Spec section "New module tts_playback.py" → Task 2 (core) + Task 3 (resample/cancel/backpressure). "Daemon wiring" → Task 4. "RPC" → Task 5. "Notifications" → Task 5 step 8 (events) + Task 2 notifier calls. "Config" → Task 1. "Tests" (unit + rpc + real-load) → Tasks 2,3,5,6. "Files" table → all 8 covered. "Verification" 1–5 → Task 7. ✓

**2. Placeholder scan:** No TBD/TODO/"add error handling". Each code step has actual code. The two `If it does NOT pass` / `If it hangs` notes in Task 3 are conditional fallbacks with concrete fixes, not placeholders. ✓

**3. Type consistency:** `StreamingPlayer.speak(text, voice="", model="") -> dict` consistent across Tasks 2/4/5. `cancel() -> dict` consistent. `notifier: Callable[[dict], Awaitable[None]]` consistent. `daemon.speak/stop_tts` signatures match between Task 4 (produces) and Task 5 (consumes via RPC). `_resolve_tts_model` reused. Config field names `tts_playback_buffer_s`/`tts_output_device` consistent across Task 1 (produces) and Task 2 (consumes). ✓

**Fix applied inline:** Task 5 step 8 — the `_wait_event` helper must scan for `state == "done"`, not just any `speech.tts` frame, since `starting` arrives first. The note in the step says this explicitly.
