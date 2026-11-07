---
doc_id: fusion-speech/api-reference
product: fusion-speech
class: app
entry: service
doc_type: api-reference
title: { zh: "API 参考", en: "API Reference" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - fusion_speech/server.py
  - fusion_speech/ws_ingress.py
---

# API Reference (API 参考)

Two transports, one contract: JSON-RPC 2.0 over UDS (newline-delimited,
peer-UID verified) and HTTP/WebSocket on `127.0.0.1:11465`. The method/param
table in README "RPC API" is exhaustive; this page adds the transport rules.

## UDS JSON-RPC (UDS 通道)

- Socket: `~/.fusion-speech/run/fusion-speech.sock` (dir 0o700).
- Framing: one JSON-RPC request per line; one response line per request.
- Security: the daemon verifies the peer UID (fail-closed); a client from
  another user account is rejected.
- Errors: JSON-RPC error objects; cloud-engine failures embed a stable slug
  (`azure_auth`, `azure_unreachable`, `azure_rate_limited`, ... ) plus a
  `retryable` flag in the message.

## RPC methods (方法清单)

`speech.ping`, `speech.status`, `speech.transcribe`, `speech.synthesize`,
`speech.list_models`, `speech.engines`, `speech.start_listening`,
`speech.stop_listening`, `speech.set_hotwords`, `speech.wake`, `speech.speak`,
`speech.stop_tts`, `speech.reprobe`. Full params/returns: README "RPC API".

Wake semantics: only pretrained wake-word names are accepted
(alexa / hey_mycroft / hey_jarvis / hey_rhasspy / timer / weather); unknown
names are dropped, not errored. `speech.wake {arm}` toggles always-on
detection; a fire emits a `speech.wake` notification and auto-starts ASR for
`FUSION_SPEECH_WAKE_AUTO_LISTEN_S`.

## HTTP (HTTP 通道)

| Path | Purpose |
|---|---|
| `GET /health` | liveness + fusion-mlx audio-lane readiness |
| `GET /metrics` | Prometheus text exposition (hand-written) |

## WebSocket `/stream/stt` (实时推送)

Subscribe-only event socket. Server frames: `speech.asr`
(listening → speaking → transcribing → final, or error), `speech.wake`,
`tts` progress events, and `{"event": "speech.keepalive"}` every
`FUSION_SPEECH_WS_KEEPALIVE_S` idle seconds — keepalive is a distinct event
type so clients never confuse it with audio events.

## WebSocket `/stream/audio` (语音会话)

One connection = one voice session, one ASR stream at a time. 16 kHz mono PCM
in binary frames (`int16` LE default, `float32` via encoding).

Client frames: `start` (mode full_duplex/half_duplex, tts_sink ws/local,
engine selection, barge_in, reprompt_min_confidence), binary PCM, `speak`
(text/voice/model, optional `ref_audio`/`ref_text` cloning params), 
`tts_cancel`, `stop`.

Server frames: the same `speech.*` events as `/stream/stt` plus
`speech.session` state transitions, `speech.barge_in`,
`speech.reprompt`, `speech.queue_overflow` (`{queue_depth}`), 
`speech.session.error`, and binary int16 PCM while TTS streams to the socket.

Session rules:

- `speak` while synthesis is active is queued FIFO (depth
  `FUSION_SPEECH_TTS_QUEUE_MAX`); overflow drops the oldest queued frame.
- Full-duplex: the client runs AEC and plays TTS locally; server VAD stays
  live, so user speech over playback fires `speech.barge_in`, drops pending
  audio, and returns to listening.
- Half-duplex: ASR pauses during playback (mic-era behavior).
- State machine: `listening → speaking → interrupted → listening`, importable
  standalone as `fusion_speech.session.VoiceSession`.

## Engine registry (引擎注册表)

`speech.engines` lists registered ASR/TTS engines. `"local"` is the
fusion-mlx delegation. Cloud engines register at startup only when their
credentials are configured (Azure: `FUSION_SPEECH_AZURE_KEY` + `_REGION`,
fail-soft). Per-session selection by name: `asr_engine`/`tts_engine` on
`start`, or `engine` on `speech.transcribe`/`speech.synthesize`.
