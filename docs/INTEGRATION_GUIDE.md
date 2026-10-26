# fusion-speech Integration Guide

> Scenario-based recipes for integrating fusion-speech into a client app
> (Python or Swift over UDS JSON-RPC + WebSocket).

fusion-speech is a local-first speech daemon: STT, TTS, hotword wake, realtime
streaming. It delegates model inference to fusion-mlx and exposes a JSON-RPC 2.0
control plane over a Unix Domain Socket plus a WebSocket event channel. This
guide covers the recurring integration scenarios with copy-paste Python clients.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Connecting over UDS JSON-RPC](#2-connecting-over-uds-json-rpc)
3. [Scenario A — One-shot transcription (STT)](#3-scenario-a--one-shot-transcription-stt)
4. [Scenario B — One-shot synthesis (TTS)](#4-scenario-b--one-shot-synthesis-tts)
5. [Scenario C — Realtime streaming ASR](#5-scenario-c--realtime-streaming-asr)
6. [Scenario D — Wake word + auto-listen](#6-scenario-d--wake-word--auto-listen)
7. [Scenario E — Streaming TTS playback](#7-scenario-e--streaming-tts-playback)
8. [Event reference](#8-event-reference)
9. [Error handling](#9-error-handling)
10. [Model selection](#10-model-selection)
11. [Observability](#11-observability)
12. [Security boundary](#12-security-boundary)
13. [Scenario F — Cloud engines (exam pipeline)](#13-scenario-f--cloud-engines-exam-pipeline)

<!-- skeleton — sections filled below -->

---

## 1. Prerequisites

fusion-speech needs the inference engine running and the daemon itself up.

```bash
# fusion-mlx owns the model weights (STT/TTS) on localhost:11434
~/fusion/fusion-mlx/start.sh start
~/fusion/fusion-mlx/start.sh status

# fusion-speech daemon on 127.0.0.1:11465 (HTTP) + UDS JSON-RPC
./start.sh start
./start.sh status
./start.sh doctor          # health check: port + socket + audio-lane probe
./start.sh log -f          # tail logs
./start.sh stop
```

Health check (HTTP, local only):

```bash
curl http://127.0.0.1:11465/health
```

The daemon reads the fusion-mlx bearer key from `FUSION_MLX_API_KEY` (or
fusion-mlx's `settings.json`). If `./start.sh doctor` reports an unreachable
audio lane, confirm `FUSION_MLX_API_KEY` is set and fusion-mlx is up.

Default socket: `~/.fusion-speech/run/fusion-speech.sock` (file mode `0600`,
dir `0700`). Override the dir with `FUSION_SOCKET_DIR`.

---

## 2. Connecting over UDS JSON-RPC

The control plane is **newline-delimited JSON-RPC 2.0** over a Unix Domain
Socket. One request per line; one response per line. The daemon verifies the
peer UID on every connection (fail-closed): only a process running as the same
user as the daemon may call it — there is no shared bearer token to leak, and a
different user's socket connect is closed immediately.

Minimal re-usable client (Python, stdlib only — no `websockets` needed for RPC):

```python
import asyncio
import contextlib
import json
import logging
import os

logger = logging.getLogger("fs.client")

SOCK = os.path.expanduser("~/.fusion-speech/run/fusion-speech.sock")


class SpeechRpc:
    def __init__(self, sock: str = SOCK, timeout: float = 120.0):
        self._sock = sock
        self._timeout = timeout
        self._r = None
        self._w = None
        self._id = 0

    async def connect(self):
        self._r, self._w = await asyncio.open_unix_connection(self._sock, limit=2 ** 22)
        logger.info("connected to %s", self._sock)

    async def close(self):
        if self._w is not None:
            self._w.close()
            with contextlib.suppress(OSError, asyncio.CancelledError):
                await self._w.wait_closed()
        logger.info("disconnected")

    async def call(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self._w.write((json.dumps(req) + "\n").encode())
        await self._w.drain()
        line = await asyncio.wait_for(self._r.readline(), timeout=self._timeout)
        return json.loads(line.decode())


async def main():
    rpc = SpeechRpc()
    await rpc.connect()
    try:
        print(await rpc.call("speech.ping"))
        print(await rpc.call("speech.status"))
    finally:
        await rpc.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

The `limit=2**22` (4 MiB) readline cap matters for `speech.transcribe` /
`speech.synthesize`, whose responses carry base64 WAV payloads. A default
readline limit would truncate large audio responses.

A response is either `{"jsonrpc":"2.0","id":N,"result":{...}}` (success) or
`{"jsonrpc":"2.0","id":N,"error":{"code":C,"message":M}}` (failure). See
[Error handling](#9-error-handling) for the code map.

---

## 3. Scenario A — One-shot transcription (STT)

Send a complete audio clip, get back the transcript. Use this for pre-recorded
files or when you buffer audio client-side; for live mic streaming use
[Scenario C](#5-scenario-c--realtime-streaming-asr).

`speech.transcribe` params: `audio` (base64-encoded PCM/WAV bytes — required),
optional `language` (e.g. `"en"`, `"zh"`), `model`, `filename`. The daemon posts
the bytes to fusion-mlx `/v1/audio/transcriptions` and returns the parsed
result.

```python
import base64
import wave


def wav_to_b64(path: str) -> str:
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return base64.b64encode(frames).encode().decode("ascii")


async def transcribe(rpc: SpeechRpc, path: str, language: str = "en") -> str:
    resp = await rpc.call("speech.transcribe", {"audio": wav_to_b64(path), "language": language})
    if "error" in resp:
        raise RuntimeError(f"transcribe failed: {resp['error']}")
    result = resp["result"]
    logger.info("transcribed: %r model=%s", result.get("text"), result.get("model"))
    return result.get("text", "")
```

Result shape: `{text, language, duration, segments?}`. Input cap: 25 MB base64
(`MAX_AUDIO_B64`) — larger raises `-32000`.

Pass `engine: "azure"` to route through a registered cloud engine instead of
fusion-mlx — see [Scenario F](#13-scenario-f--cloud-engines-exam-pipeline).

**Known upstream limit:** fusion-mlx issue #668 — `mlx-audio` Whisper
`non_speech_tokens` `IndexError` makes `/v1/audio/transcriptions` return 500
for every model. Until the fix lands, `speech.transcribe` errors with
`-32000` `"AudioBridgeError (status 500): ..."`. Workaround: call fusion-mlx's
endpoint directly only if you have a non-Whisper STT path, or wait for #668.
The fusion-speech RPC path itself is correct — the failure is upstream.

---

## 4. Scenario B — One-shot synthesis (TTS)

Send text, get back a complete WAV. Use this when you want the audio bytes to
play/queue yourself; for daemon-managed speaker playback use
[Scenario E](#7-scenario-e--streaming-tts-playback).

`speech.synthesize` params: `text` (required), optional `voice`, `model`.
Returns `{audio: <base64 wav>, format: "wav", size}`. Text cap: 5000 chars.

```python
import base64


async def synthesize(rpc: SpeechRpc, text: str, voice: str = "") -> bytes:
    resp = await rpc.call("speech.synthesize", {"text": text, "voice": voice} if voice else {"text": text})
    if "error" in resp:
        raise RuntimeError(f"synthesize failed: {resp['error']}")
    result = resp["result"]
    wav = base64.b64decode(result["audio"])
    logger.info("synthesized %d bytes (format=%s)", len(wav), result["format"])
    return wav


# play with sounddevice (optional, local speaker)
def play(wav_bytes: bytes):
    import io
    import sounddevice as sd
    from scipy.io import wavfile

    rate, data = wavfile.read(io.BytesIO(wav_bytes))
    sd.play(data, rate)
    sd.wait()
```

If `model`/`voice` are omitted, the daemon uses `FUSION_SPEECH_TTS_MODEL` /
`FUSION_SPEECH_TTS_VOICE`, falling back to the first probed TTS model if the
configured id is not loaded in fusion-mlx (logged at WARNING). Pass
`engine: "azure"` to route through a registered cloud engine (voice then
defaults to `FUSION_SPEECH_AZURE_TTS_VOICE`).

---

## 5. Scenario C — Realtime streaming ASR

For live microphone capture: arm capture via RPC, then subscribe to the
WebSocket `/stream/stt` to receive `speech.asr` event frames as the daemon
segments and transcribes speech. The WebSocket is **receive-only** (the daemon
pushes frames; you do not send commands on it) and **local-only** — no auth
token, the socket binds `127.0.0.1`.

State machine for `speech.asr`:

```
listening ──(VAD speech onset)──▶ speaking ──(silence)──▶ transcribing ──▶ final
                                                                    │
                                                                    └──▶ error
```

- `listening` — capture armed, waiting for speech.
- `speaking` — VAD detected speech onset (informs UI to show "listening…").
- `transcribing` — a segment closed; the daemon is transcribing it on fusion-mlx.
- `final` — transcript ready: `{text, language, duration, seq}`. Short/noise
  segments arrive as `final` with `text: ""` and `skipped: true`.
- `error` — a segment's transcribe failed: `{error, seq}`.

```python
import asyncio
import json
import os

import websockets

SOCK = os.path.expanduser("~/.fusion-speech/run/fusion-speech.sock")


async def stream_asr(rpc: SpeechRpc, on_final):
    # 1. subscribe to events BEFORE arming so no frame is missed
    uri = f"ws+unix://{SOCK}:/stream/stt"
    async with websockets.connect(uri, max_size=2 ** 22) as ws:
        # 2. arm mic capture + streaming ASR
        await rpc.call("speech.start_listening")
        logger.info("listening armed")
        try:
            async for raw in ws:
                frame = json.loads(raw)
                if frame.get("event") != "speech.asr":
                    continue  # keepalive / wake / tts frames, see Event reference
                if frame.get("state") == "final" and not frame.get("skipped"):
                    await on_final(frame.get("text", ""), frame.get("seq"))
        finally:
            await rpc.call("speech.stop_listening")
            logger.info("listening stopped")


async def main():
    rpc = SpeechRpc()
    await rpc.connect()
    try:
        async def on_final(text, seq):
            logger.info("final[%d]: %r", seq, text)
        await stream_asr(rpc, on_final)
    finally:
        await rpc.close()
```

Keep the RPC socket and the WebSocket open in parallel: RPC arms/stops capture,
the WS delivers the frames. `speech.stop_listening` returns
`{listening: false, segments: <count>}`.

---

## 6. Scenario D — Wake word + auto-listen

The classic voice-assistant flow: always-on wake-word detection, and on a wake
the daemon auto-starts ASR for a listen window, transcribes one utterance, then
resumes wake. Your client subscribes to events and reacts to the wake + the
ensuing final transcript.

`speech.wake {arm: true}` arms openwakeword ONNX detection. On a wake the daemon
emits a `speech.wake` notification, pauses wake, auto-starts streaming ASR for
`FUSION_SPEECH_WAKE_AUTO_LISTEN_S` seconds (ends early on the first `final` +
silence), then resumes wake. Wake is paused during ASR so the user's own speech
does not re-trigger.

```python
async def assistant_loop(rpc: SpeechRpc):
    uri = f"ws+unix://{SOCK}:/stream/stt"
    async with websockets.connect(uri, max_size=2 ** 22) as ws:
        await rpc.call("speech.wake", {"arm": True})
        logger.info("wake armed (words=%s)", (await rpc.call("speech.status"))["result"].get("hotwords"))
        try:
            async for raw in ws:
                frame = json.loads(raw)
                ev = frame.get("event")
                if ev == "speech.wake":
                    logger.info("WAKE: %s", frame.get("word"))
                elif ev == "speech.asr" and frame.get("state") == "final" and not frame.get("skipped"):
                    logger.info("you said: %r", frame.get("text"))
                    # daemon auto-resumes wake after the listen window
        finally:
            await rpc.call("speech.wake", {"arm": False})
```

Swap wake words at runtime (pretrained names only — unknown names are dropped):

```python
await rpc.call("speech.set_hotwords", {"hotwords": ["hey_jarvis", "alexa"]})
```

Pretrained words: `alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy`, `timer`,
`weather`. Wake models download on first arm to
`~/.fusion-speech/models/wakeword/` (URL configurable via
`FUSION_SPEECH_WAKE_MODEL_URL` for a mirror). Fail-soft: if openwakeword or the
ONNX models are unavailable, `speech.wake` returns `{armed: false}` and the
daemon keeps running.

---

## 7. Scenario E — Streaming TTS playback

Let the daemon own the speaker: `speech.speak` streams TTS to the local audio
output (it pauses mic capture / wake while speaking so there is no echo or
self-trigger), and emits `speech.tts` progress events on the WebSocket.

`speech.speak` params: `text` (required), optional `voice`, `model`. Returns
`{played, chars, sample_rate}`. `speech.stop_tts` cancels active playback.

`speech.tts` event states: `starting` (`{chars}`) → `speaking` → `done`
(`{chars}`), or `cancelled`, or `error` (`{error, degraded}`).

```python
async def speak(rpc: SpeechRpc, text: str):
    uri = f"ws+unix://{SOCK}:/stream/stt"
    async with websockets.connect(uri, max_size=2 ** 22) as ws:
        result = (await rpc.call("speech.speak", {"text": text}))["result"]
        logger.info("speak dispatched played=%s", result.get("played"))
        async for raw in ws:
            frame = json.loads(raw)
            if frame.get("event") != "speech.tts":
                continue
            state = frame.get("state")
            logger.info("tts: %s", state)
            if state in ("done", "cancelled", "error"):
                if state == "error" and frame.get("degraded"):
                    logger.warning("tts degraded: %s", frame.get("error"))
                break
```

The `degraded: true` flag on a `speech.tts` error means wake/ASR could not be
cleanly resumed after playback (the daemon latches `tts.degraded` until a clean
resume). The TTS audio itself still played; the degradation is in the realtime
layers, not the output. Call `speech.wake {arm: true}` again to reset, or
`speech.reprobe` if the audio lane is also stale.

`speech.speak` over RPC is one-playback-at-a-time (a second call while active
returns `{"played": false, "error": "already speaking"}` — safe to retry after
the `done`/`cancelled` event). A streaming voice agent that feeds TTS sentence
by sentence should use the WS voice session instead (`/stream/audio`): there,
concurrent `speak` frames queue FIFO per session (depth
`FUSION_SPEECH_TTS_QUEUE_MAX`, overflow drops the oldest and emits
`speech.queue_overflow`) so sentence N+1 starts the moment N finishes — no
client-side serialization gate. See the "Voice Sessions" section of the README
for the full protocol.

---

## 8. Event reference

All event frames are delivered on the `/stream/stt` WebSocket as JSON, one per
message. The `event` field discriminates the type.

| Event | Frame fields | When emitted |
|---|---|---|
| `speech.asr` | `state` (`listening`/`speaking`/`transcribing`/`final`/`error`), `seq`, and on `final`: `text`, `language?`, `duration?`, `skipped?`, `engine?`, `words?` (word/offset_ms/duration_ms/confidence), `dialect?`, `dialect_detected?`; on `error`: `error` | realtime ASR lifecycle; one `listening` on arm, per-segment states during capture, `final` on each transcript. The `engine`/`words`/`dialect*` fields appear when a cloud ASR engine produced the segment |
| `speech.wake` | `word`, `ts` | a wake word fired (before the auto-listen handshake) |
| `speech.tts` | `state` (`starting`/`speaking`/`done`/`cancelled`/`error`), `chars?` (starting/done), `error?` + `degraded?` (error) | `speech.speak` playback progress |
| `speech.keepalive` | — | idle keepalive every `FUSION_SPEECH_WS_KEEPALIVE_S` (default 30s); distinct event so clients filter real frames from noise |
| `speech.barge_in` | `session`, `count`, `ts` | WS voice session: user spoke over TTS playback (full-duplex barge-in); playback cancelled, queued sentences dropped |
| `speech.queue_overflow` | `session`, `queue_depth`, `ts` | WS voice session: the per-session speak queue was full; the oldest queued sentence was dropped |
| `speech.shutdown` | `ts` | daemon is draining on stop; close the WS and reconnect after restart |
| `speech.restarted` | `restart_count`, `ts` | this boot is a crash-restart (not the first); emitted right after start so clients know prior state was lost |

A client should switch on `event` and, within `speech.asr`, switch on `state`.
Keepalive is intentionally its own type — do not treat a `speech.keepalive` as
an ASR frame.

---

## 9. Error handling

JSON-RPC 2.0 error responses carry `{code, message}` under `error`. The codes
the daemon returns:

| Code | Meaning | When |
|---|---|---|
| `-32700` | Parse error | malformed JSON on a request line |
| `-32600` | Invalid Request | missing `jsonrpc: "2.0"` field |
| `-32601` | Method not found | unknown `method` name |
| `-32603` | Internal error | unexpected dispatch-layer failure |
| `-32000` | Handler error | a handler raised, timed out (120s backstop), or hit an input bound |

Handler errors are formatted as `"<ExceptionType> [(status N)]: <message>"` —
e.g. `AudioBridgeError (status 404): synthesize failed: Model ... not found`.
The `status N` segment appears when the raised exception carries a `.status`
attribute (HTTP status from fusion-mlx). Examples:

- `ValueError: audio (base64) required` — missing `audio` on `speech.transcribe`.
- `ValueError: audio too large: 30000000 > 26214400 bytes` — over the 25 MB cap.
- `ValueError: text too long: 6000 > 5000 chars` — over the 5000-char cap.
- `AudioBridgeError (status 500): ...` — fusion-mlx returned 5xx (e.g. issue #668 STT 500).
- `ValueError: azure stt failed (AuthenticationFailure): ... reason=azure_auth retryable=False: ...`
  — cloud engine failure; the `reason=<slug> retryable=<bool>` suffix is the
  stable machine-readable part (see [Scenario F](#13-scenario-f--cloud-engines-exam-pipeline)).

Input bounds: `speech.transcribe` audio ≤ 25 MB base64 (`MAX_AUDIO_B64`);
`speech.speak` / `speech.synthesize` text ≤ 5000 chars (`MAX_TEXT`). A request
line is also capped at 1 MiB server-side — anything larger with no newline
closes the connection (DoS guard, audit F3).

Client pattern:

```python
resp = await rpc.call("speech.synthesize", {"text": text})
if "error" in resp:
    err = resp["error"]
    logger.error("rpc %s failed code=%s msg=%s", method, err.get("code"), err.get("message"))
    raise RuntimeError(err["message"])
result = resp["result"]
```

---

## 10. Model selection

When a `model` param is omitted, the daemon resolves one:

1. Use the configured default (`FUSION_SPEECH_TTS_MODEL` / `FUSION_SPEECH_STT_MODEL`).
2. If that id is **not** in the probed fusion-mlx model set, fall back to the
   first probed model of that modality (logged at WARNING).

This makes `speech.synthesize {}` work out-of-box even when fusion-mlx's loaded
model differs from the configured default. Inspect and refresh the model set:

```python
models = (await rpc.call("speech.list_models"))["result"]
print(models["audio_models"])          # e.g. ["kokoro", "openai/whisper-tiny", ...]

lane = (await rpc.call("speech.reprobe"))["result"]["audio_lane"]
# refreshes the cache after you load/unload models in fusion-mlx
```

`speech.status` also reports the resolved `stt_model` / `tts_model` and the
probed `mlx_audio_lane` (with `tts_models` / `stt_models` lists).

**Chinese setup:** set `FUSION_SPEECH_STT_MODEL=openai/whisper-large-v3` and
`FUSION_SPEECH_LANGUAGE=zh` (env or pass `language="zh"` per-call). Note
`mlx-community/whisper-*-mlx*` repos ship no tokenizer files and are unusable
for STT — use `openai/whisper-*`.

---

## 11. Observability

Two surfaces:

- **HTTP `/metrics`** — Prometheus text exposition (zero-dependency, hand-written)
  on `127.0.0.1:11465`. Scrape:

  ```bash
  curl http://127.0.0.1:11465/metrics
  ```

  Counters (`rpc_requests_total`, `rpc_errors_total`, `asr_segments_total`,
  `wake_fires_total`, `tts_speak_total`), histograms (`rpc_duration_seconds`,
  `asr_transcribe_seconds`), gauges (`subscribers`, `listening`, `wake_armed`,
  `speaking`). See `README.md` Metrics table for the full list.

- **`speech.status` RPC** — returns a runtime snapshot incl. a `metrics` summary
  (`uptime_seconds`, `rpc_requests_total`, `rpc_errors_total`,
  `asr_segments_total`, `wake_fires_total`, `tts_speak_total`) plus live state
  (`listening`, `wake_armed`, `speaking`, `restart_count`, `last_wake_word`).

```python
status = (await rpc.call("speech.status"))["result"]
logger.info("up %.0fs listening=%s wake=%s speaking=%s",
            status["uptime"], status["listening"], status["wake_armed"], status["speaking"])
```

---

## 12. Security boundary

fusion-speech is local-first by design. Integration clients should assume:

- **Local transport only.** HTTP and UDS bind `127.0.0.1` / the local socket.
  No network egress; the daemon never calls out except to fusion-mlx on
  `localhost:11434`.
- **Peer-UID verification (fail-closed).** Every UDS connection is checked via
  `LOCAL_PEERCRED` (macOS) / `SO_PEERCRED` (Linux); only the same-UID process
  may call RPC. A different user's connect is closed immediately. There is no
  shared token to leak.
- **WebSocket is receive-only and local.** `/stream/stt` carries no auth token
  and accepts no client commands — it only pushes event frames.
- **No secrets in logs.** The fusion-mlx bearer key (`FUSION_MLX_API_KEY`) is
  used in the HTTP `Authorization` header to fusion-mlx only; it is not written
  to the daemon log.
- **Bounded inputs.** Audio ≤ 25 MB, text ≤ 5000 chars, lines ≤ 1 MiB — all
  enforced server-side to prevent memory/DoS abuse of the socket.

Do not expose the UDS socket or HTTP port beyond the host. For a GUI
integration (e.g. fusion-studio), connect in-process on the same machine and
let the OS UID check gate access.

---

## 13. Scenario F — Cloud engines (exam pipeline)

For flows that need cloud-grade STT/TTS — the ai-speech-exam pipeline is the
reference case: word/timestamp alignment for scoring, dialect/accent detection
to raise human-review priority, read-aloud TTS. **Credentials never live in
your app**: the daemon holds them (`FUSION_SPEECH_AZURE_KEY` /
`FUSION_SPEECH_AZURE_REGION`, issued via fusion-identity) and you select the
engine by name. Check what is registered first:

```python
engines = (await rpc.call("speech.engines"))["result"]
# {"asr": ["local", "azure"], "tts": ["local", "azure"]} — "azure" only when
# the daemon was started with Azure credentials configured
```

### STT with word alignment + dialect detection

```python
import re

REASON_RE = re.compile(r"reason=(\S+) retryable=(True|False)")


async def exam_transcribe(rpc: SpeechRpc, wav_b64: str) -> dict:
    resp = await rpc.call(
        "speech.transcribe",
        {"audio": wav_b64, "engine": "azure"},
    )
    if "error" in resp:
        return {"ok": False, **_parse_reason(resp["error"]["message"])}
    result = resp["result"]
    # words: [{"word", "offset_ms", "duration_ms", "confidence"}, ...]
    # dialect: detected locale, e.g. "ar-EG" — requires LID candidates
    #          (FUSION_SPEECH_AZURE_STT_LANGUAGES, e.g. "ar-EG,ar-SA,en-US")
    return {
        "ok": True,
        "text": result.get("text", ""),
        "words": result.get("words", []),
        "dialect": result.get("dialect", ""),
        "review_priority": "high" if result.get("dialect_detected") else "normal",
    }


def _parse_reason(message: str) -> dict:
    m = REASON_RE.search(message)
    if not m:
        return {"reason": "unknown", "retryable": False}
    return {"reason": m.group(1), "retryable": m.group(2) == "True"}
```

### Failure-reason branching for the audio job pipeline

Branch on the reason slug — auth/config problems never requeue, transient ones do:

| Reason | Retryable | Job-pipeline action |
|---|---|---|
| `azure_auth`, `azure_not_configured`, `azure_sdk_missing` | no | dead-letter; page on-call (credential/deployment issue) |
| `azure_bad_request`, `invalid_audio`, `azure_invalid_response` | no | dead-letter; inspect the job payload |
| `azure_unreachable`, `azure_timeout`, `azure_service` | yes | requeue with backoff |
| `azure_rate_limited` | yes | requeue with longer backoff (429) |

The daemon already retries internally (0.5/1/2 s backoff, retryable reasons
only); the slug reaching your app means the retries were exhausted. Metrics:
`cloud_requests_total{engine,kind}` / `cloud_errors_total{engine,kind,reason}`
on `/metrics`.

### Read-aloud TTS

```python
async def read_aloud(rpc: SpeechRpc, text: str, voice: str = "") -> bytes:
    params = {"text": text, "engine": "azure"}
    if voice:
        params["voice"] = voice  # e.g. "ar-EG-SalmaNeural"
    resp = await rpc.call("speech.synthesize", params)
    if "error" in resp:
        raise RuntimeError(resp["error"]["message"])
    return base64.b64decode(resp["result"]["audio"])  # 16 kHz mono WAV
```

Cloud engines also work in realtime paths: a WS voice session `start` frame
accepts `asr_engine`/`tts_engine` (e.g. `"azure"`), and the ensuing
`speech.asr` `final` frames carry the same `engine`/`words`/`dialect`/
`dialect_detected` fields. Set up the daemon with
`pip install -e ".[azure]"` plus the credential env vars; without them the
daemon stays local-only (fail-soft) and `speech.engines` lists just `local`.
