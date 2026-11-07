---
doc_id: fusion-speech/get-started
product: fusion-speech
class: app
entry: service
doc_type: get-started
title: { zh: "快速开始", en: "Get Started" }
audience: engineer
status: draft
version: 0.1.6
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - start.sh
  - pyproject.toml
---

# Get Started (快速开始)

## Prerequisites (前置条件)

- macOS on Apple Silicon; Python 3.12. In the fusion monorepo:
  `cd ~/fusion && source .venv/bin/activate`.
- fusion-mlx running on :11434 — fusion-speech loads no models itself; it
  posts audio to `/v1/audio/transcriptions` and reads PCM from
  `/v1/audio/speech`.
- A microphone for local capture (or skip it and feed PCM over WebSocket).
- Wake word detection needs the `[audio]` extra (onnxruntime + openwakeword).

- macOS（Apple Silicon）+ Python 3.12（monorepo 共享 `.venv`）。
- 需要 fusion-mlx（:11434）在运行；唤醒检测需 `[audio]` 扩展。

## Install (安装)

```bash
cd ~/fusion
source .venv/bin/activate
pip install -e fusion-core
cd fusion-speech
pip install -e ".[test]"      # add [audio] for wake word, [azure] for cloud engines
```

## Start the daemon (启动守护进程)

```bash
~/fusion/fusion-mlx/start.sh start    # inference backend first
./start.sh start                      # daemon on 127.0.0.1:11465 + UDS
./start.sh status
./start.sh doctor                     # environment + audio lane check
./start.sh log -f                     # tail logs (~/.fusion-speech/logs)
./start.sh stop
```

start.sh self-checks the port against `architecture/port-registry.yaml`
(11465). Bypass in sandboxes with `FUSION_SKIP_REGISTRY_CHECK=1`.

start.sh 会对照 port-registry 自检端口（11465），沙箱内可用
`FUSION_SKIP_REGISTRY_CHECK=1` 跳过。

## Verify (验证)

```bash
curl http://127.0.0.1:11465/health
curl http://127.0.0.1:11465/metrics | head
```

Smoke-test the RPC lane:

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

Expect `{"jsonrpc": "2.0", "id": 1, "result": {"pong": true, ...}}`.

## First transcription (第一次识别)

```python
import base64, json, os, asyncio

async def main():
    audio = base64.b64encode(open("sample.wav", "rb").read()).decode()
    r, w = await asyncio.open_unix_connection(
        os.path.expanduser("~/.fusion-speech/run/fusion-speech.sock"))
    req = {"jsonrpc": "2.0", "id": 2, "method": "speech.transcribe",
           "params": {"audio": audio, "language": "en"}}
    w.write((json.dumps(req) + "\n").encode()); await w.drain()
    print(json.loads(await r.readline()))
    w.close()

asyncio.run(main())
```

For Chinese STT set `FUSION_SPEECH_STT_MODEL=openai/whisper-large-v3` and
`FUSION_SPEECH_LANGUAGE=zh` before starting the daemon.

中文识别需在启动前设置 `FUSION_SPEECH_STT_MODEL=openai/whisper-large-v3` 和
`FUSION_SPEECH_LANGUAGE=zh`。

## What you no longer have to do (你不再需要做什么)

- You do not need to manage STT/TTS models or GPU memory in your app — the
  daemon + fusion-mlx own that; your client is one UDS socket.
  Limitation: your process must run as the same user (UDS peer-UID check).
- You do not need a separate metrics stack dependency — `GET /metrics` is
  Prometheus-format text with zero extra libraries.

- 应用侧不再管理模型与显存：客户端只需一个 UDS 连接；约束是同用户进程（对端 UID 校验）。
- 无需引入 metrics 依赖：`/metrics` 为零依赖 Prometheus 文本格式。

## Next steps (下一步)

- `docs/api-reference.md` — RPC methods + WS protocols
- `docs/config-reference.md` — all `FUSION_SPEECH_*` variables
- `docs/troubleshooting.md` — common failures
- `docs/INTEGRATION_GUIDE.md` — Python/Swift client recipes
