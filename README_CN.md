# fusion-speech

> [中文](README_CN.md) | [English](README.md)

> Fusion 生态的本地优先低延迟语音守护进程。

本地优先语音服务：语音转文字（STT）、文字转语音（TTS）、热词唤醒、实时流式。
运行于 Apple Silicon。通过 HTTP 把模型推理委托给 fusion-mlx，仅添加 fusion-mlx
缺失的实时层（麦克风采集、唤醒、流式 ASR、流式 TTS 播放）。

> 注：本文为 [`README.md`](README.md) 的中文镜像，英文版为权威源。版本号、
> 命令、环境变量、RPC 方法名、指标名、文件路径保持英文原文（技术术语不译）。

## 概述

- **守护进程**：asyncio FastAPI 服务，监听 `127.0.0.1:11465`（HTTP `/health`），
  另开 Unix Domain Socket 提供 JSON-RPC 2.0 IPC
  （`~/.fusion-speech/run/fusion-speech.sock`）。
- **委托**：STT/TTS 模型推理在 fusion-mlx（`localhost:11434`）中运行。
  fusion-speech 把音频 POST 到 `/v1/audio/transcriptions`，从 `/v1/audio/speech`
  读取 PCM 流。本进程不加载任何模型权重。
- **实时层（M2-M4）**：环形缓冲麦克风采集、VAD、分块流式 ASR、流式 TTS 播放、
  热词唤醒。
- **IPC**：基于 UDS 的换行分隔 JSON-RPC 2.0，带 peer-UID 校验（fail-closed），
  仅同用户进程可调用守护进程。

## 架构

```
┌──────────────┐   JSON-RPC/UDS    ┌───────────────┐   HTTP   ┌────────────┐
│ fusion-studio│ ────────────────▶ │ fusion-speech │ ───────▶ │ fusion-mlx │
│  (Swift UI)  │ ◀──────────────── │   (daemon)    │ ◀─────── │ (models)   │
└──────────────┘  notifications    └───────────────┘  PCM/    └────────────┘
                                     mic│VAD│wake     audio
                                        │
                                  realtime layers (M2-M4)
```

推理边界：fusion-mlx 拥有权重和 OpenAI 兼容的 `/v1/audio/*` 端点。fusion-speech
拥有麦克风到这些端点之间的一切。完整分层与 fusion-* 集成图见
`docs/ARCHITECTURE.md`。

## 快速开始

```bash
# 从 fusion monorepo 根目录
source .venv/bin/activate
pip install -e fusion-core
pip install -e ".[test]"        # 在 fusion-speech/ 下执行

# 先启动 fusion-speech 委托的推理引擎
~/fusion/fusion-mlx/start.sh start

# 启动语音守护进程
./start.sh start
./start.sh status
./start.sh doctor

# 日志
./start.sh log -f

# 停止
./start.sh stop
```

冒烟测试 RPC 通道：

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

## 配置

所有配置走环境变量（无配置文件）。默认值在 `fusion_speech/config.py`。

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `FUSION_SPEECH_PORT` | `11465` | HTTP 端口（仅 localhost） |
| `FUSION_SOCKET_DIR` | `~/.fusion-speech/run` | UDS socket 目录（0o700） |
| `FUSION_SPEECH_STT_MODEL` | `openai/whisper-tiny` | STT 模型 id（多语言含中文用 `openai/whisper-large-v3`） |
| `FUSION_SPEECH_TTS_MODEL` | `kokoro` | TTS 模型 id（fusion-mlx 别名） |
| `FUSION_SPEECH_TTS_VOICE` | `af_heart` | TTS 音色 |
| `FUSION_SPEECH_LANGUAGE` | `en` | 默认 STT 语言（配合 `openai/whisper-large-v3` 设 `zh`） |
| `FUSION_SPEECH_SAMPLE_RATE` | `16000` | 采集采样率 —— 硬锁定 16000（VAD/wake/openwakeword 契约）；非 16000 启动失败 |
| `FUSION_SPEECH_VAD_THRESHOLD` | `0.3` | VAD 语音阈值 |
| `FUSION_SPEECH_CHUNK_MS` | `300` | 流式分块大小 |
| `FUSION_SPEECH_WAKE_WORD` | `hey_jarvis` | 默认唤醒词（预训练：alexa/hey_mycroft/hey_jarvis/hey_rhasspy/timer/weather） |
| `FUSION_SPEECH_WAKE_THRESHOLD` | `0.5` | 唤醒得分阈值（0-1） |
| `FUSION_SPEECH_WAKE_AUTO_LISTEN_S` | `8.0` | 唤醒后自动监听窗口 |
| `FUSION_SPEECH_WAKE_DEBOUNCE_S` | `1.0` | 唤醒触发间冷却 |
| `FUSION_SPEECH_WAKE_MODEL_DIR` | `~/.fusion-speech/models/wakeword` | 唤醒 ONNX 模型缓存目录 |
| `FUSION_SPEECH_WAKE_MODEL_URL` | GitHub releases v0.5.1 | 唤醒模型下载 base URL（可改镜像） |
| `FUSION_SPEECH_TTS_BUFFER_S` | `2.0` | TTS 播放缓冲上限（秒） |
| `FUSION_SPEECH_TTS_OUTPUT_DEVICE` | （系统默认） | sounddevice 输出设备 |
| `FUSION_SPEECH_WS_KEEPALIVE_S` | `30.0` | `/stream/stt` WebSocket 空闲 keepalive 间隔（秒） |
| `FUSION_SPEECH_WS_SUBSCRIBER_QUEUE` | `64` | 每订阅者推送队列深度（满则丢最旧） |
| `FUSION_SPEECH_WAKE_CHUNK_SAMPLES` | `1280` | 固定唤醒帧大小 —— 必须 1280（openwakeword 80ms 契约）；非 1280 启动失败 |
| `FUSION_SPEECH_PROBE_INTERVAL_S` | `60.0` | 后台音频通道重探间隔（秒） |
| `FUSION_SPEECH_LOG_LEVEL` | `INFO` | 日志级别 |
| `FUSION_SPEECH_LOG_DIR` | `~/.fusion-speech/logs` | 日志目录 |
| `FUSION_MLX_API_KEY` | （来自 settings.json） | fusion-mlx bearer key |

## RPC API

基于 UDS 的 JSON-RPC 2.0，换行分隔。Peer UID 校验（fail-closed）。

| 方法 | 参数 | 返回 |
|---|---|---|
| `speech.ping` | `{}` | `{pong: true, timestamp}` |
| `speech.status` | `{}` | running, socket_path, port, uptime, mlx_audio_lane, stt_model, tts_model, listening, hotwords |
| `speech.transcribe` | `{audio: base64, language?, model?, filename?}` | `{text, language, duration, segments?}` |
| `speech.synthesize` | `{text, voice?, model?}` | `{audio: base64 wav, format: "wav", size}` |
| `speech.list_models` | `{}` | `{total, audio_models}` |
| `speech.start_listening` | `{}` | `{listening: true}` —— 激活麦克风采集 + 流式 ASR |
| `speech.stop_listening` | `{}` | `{listening: false, segments: <int>}` |
| `speech.set_hotwords` | `{hotwords: [...]}` | `{hotwords: [...]}` —— 注册唤醒词（仅预训练名；未知词丢弃） |
| `speech.wake` | `{arm: true\|false}` | `{armed: bool, words: [...]}` —— 激活/解除唤醒检测 |
| `speech.speak` | `{text, voice?, model?}` | `{played, chars, sample_rate}` —— 流式 TTS 到扬声器（暂停麦克风） |
| `speech.stop_tts` | `{}` | `{stopped}` —— 取消正在进行的播放 |
| `speech.reprobe` | `{}` | `{audio_lane}` —— 按需重探 fusion-mlx 音频通道（刷新 `list_models` 缓存） |

M2 实时推送：连接 WebSocket `/stream/stt` 接收 `speech.asr` 通知帧
（`listening` → `speaking` → `transcribing` → `final`，或 `error`）。M3 在同一
socket 上新增 `speech.wake` 通知（随后为监听窗口自动启动 ASR）。M4 在同一 socket
上新增 `speech.tts` 进度事件。keepalive 是独立事件类型 ——
`{"event": "speech.keepalive"}`，每 `FUSION_SPEECH_WS_KEEPALIVE_S` 秒空闲一次，
区别于 `speech.asr`，客户端可据此区分 keepalive 与真实音频事件。

### 唤醒（M3）

`speech.wake {arm: true}` 激活常驻唤醒词检测（openwakeword，ONNX）。唤醒时守护进程
向 `/stream/stt` 订阅者发 `speech.wake` 通知，暂停唤醒，复用同一麦克风源为
`FUSION_SPEECH_WAKE_AUTO_LISTEN_S` 秒自动启动流式 ASR，随后停止 ASR 恢复唤醒 ——
经典助手流程。ASR 期间暂停唤醒，自身说话不会自触发。Fail-soft：openwakeword 或
ONNX 模型不可用时，`speech.wake` 返回 `{armed: false}`，守护进程继续运行。

唤醒模型（唤醒词 `.onnx` + `melspectrogram.onnx` + `embedding_model.onnx`）首次激活
时从 `FUSION_SPEECH_WAKE_MODEL_URL` 下载到 `FUSION_SPEECH_WAKE_MODEL_DIR`
（默认：openWakeWord GitHub releases v0.5.1）。预训练词：`alexa`、`hey_mycroft`、
`hey_jarvis`、`hey_rhasspy`、`timer`、`weather`。

Handler 错误映射到 `-32000`，格式为 `"<ExceptionType> [(status N)]: <message>"`
（如 `AudioBridgeError (status 404): synthesize failed: Model ... not found`）。

模型解析：未给 `model` 参数时，守护进程用配置默认值
（`FUSION_SPEECH_TTS_MODEL` / `FUSION_SPEECH_STT_MODEL`）；若该 id 不在探测到的
fusion-mlx 模型集中，回退到该模态首个被探测的音频模型（WARNING 日志）。这让
`speech.synthesize {}` 即使 fusion-mlx 加载的模型与配置默认值不同也能开箱即用。

错误码遵循 JSON-RPC 2.0：`-32700` 解析错误、`-32600` 非法请求、`-32601` 方法未找到、
`-32603` 内部错误、`-32000` handler 错误。

## 指标

HTTP 端口（`127.0.0.1:11465`）上的 `GET /metrics` 返回 Prometheus 文本格式
（零依赖，手写 —— 未引入 `prometheus_client`）。每个 RPC handler 都计时计数；监听器、
唤醒引擎、TTS 播放器报告状态 gauge；ASR/TTS 桥报告吞吐。

| 指标 | 类型 | 标签 | 度量内容 |
|---|---|---|---|
| `rpc_requests_total` | Counter | `method` | 每方法 RPC 调用数 |
| `rpc_errors_total` | Counter | `method` | 每方法 RPC handler 错误数 |
| `rpc_duration_seconds` | Histogram | `method` | RPC handler 延迟（秒） |
| `asr_segments_total` | Counter | — | ASR 段关闭数（VAD 段关闭） |
| `asr_transcribe_seconds` | Histogram | — | 每段转写延迟 |
| `asr_segments_dropped_total` | Counter | — | R4 背压丢弃的段数 |
| `wake_fires_total` | Counter | — | 唤醒词触发数 |
| `tts_speak_total` | Counter | — | `speech.speak` 调用数 |
| `subscribers` | Gauge | — | 当前 WS/事件订阅者数 |
| `listening` | Gauge | — | ASR 监听器激活时为 1，否则 0 |
| `wake_armed` | Gauge | — | 唤醒引擎激活时为 1，否则 0 |
| `speaking` | Gauge | — | TTS 播放器激活时为 1，否则 0 |

`speech.status` 也返回指标摘要（`uptime`、`rpc_total`、`errors_total`）。抓取方式：

```bash
curl http://127.0.0.1:11465/metrics
```

## 集成

- **fusion-mlx**（硬依赖）：拥有 `/v1/audio/transcriptions`（STT）和
  `/v1/audio/speech`（TTS）。fusion-speech 启动时探测音频通道，并在 `/health` 和
  `speech.status` 中报告就绪状态。
- **fusion-core**（in-tree 共享底座）：`mlx_client.FusionMLXClient` 做 HTTP，
  `http.app.create_app` 做 FastAPI 工厂，`config` 解析 base URL / API key，
  `logging` 做日志设置。
- **fusion-studio**（Swift 桌面端，M5）：通过 UDS JSON-RPC 调用守护进程；注册
  SpeechBridge + 麦克风权限。上游改动 —— 已提 issue dahai80/fusion-studio#337；
  Swift PR 按上游优先工作流随后跟进。
- **port-registry**：在 `architecture/port-registry.yaml` 占用 `11465`（M5）。

仅修改本工程目录。上游 fusion-mlx / fusion-studio / port-registry 的改动先提 issue，
再提 PR。

场景化集成配方（transcribe、synthesize、wake、流式 ASR/TTS 的 Python/Swift UDS
客户端）见 [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md)。开发者环境
搭建、测试、代码风格与上游 issue→PR 工作流见
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)。

## 测试

单元 + 对抗性门禁套件全离线运行（不加载模型、不起活守护进程）：

```bash
cd /Users/dahai/fusion && source .venv/bin/activate
cd fusion-speech
FUSION_MLX_API_KEY=dahai168 pytest tests/ -q          # 183 passed, 6 skipped
ruff check fusion_speech tests                         # clean
```

5 个 skipped 测试带 `real_load` 标记 —— 它们启动真实 fusion-speech 守护进程
（真实端口 11465 + 真实 socket），背后是真实 fusion-mlx（11434），故为可选。设
`FUSION_SPEECH_REAL_LOAD=1`（并确保 fusion-mlx 已起）来运行：

```bash
~/fusion/fusion-mlx/start.sh start
FUSION_SPEECH_REAL_LOAD=1 FUSION_MLX_API_KEY=dahai168 \
  pytest tests/test_m6_real_load_integration.py -m real_load -v
```

real-load 套件是基于 UDS JSON-RPC 的 5 步全链路门禁：

1. `speech.ping`/`speech.status` —— 存活 + 音频通道已探测。
2. `speech.transcribe` —— 经 fusion-mlx 对内置英文 fixture 做真实 STT。**上游
   fusion-mlx STT 500 时跳过**（issue #668：`mlx-audio` Whisper
   `non_speech_tokens` `IndexError` 使 `/v1/audio/transcriptions` 对每个模型返回
   500）。#668 落地修复后即刻变成真实断言；RPC 路径本身已验证正确。
3. `speech.synthesize` —— 真实 TTS，断言非空 WAV（`RIFF` + PCM body）。
4. `speech.speak` —— 流式播放，轮询 `speech.status` 直至稳定。
5. `speech.wake` —— 经真实 openwakeword ONNX 激活/解除；无头静音不得误触发唤醒。

测试 fixture 在 `tests/fixtures/`（作为最终产物保留）。UDS 客户端 helper 是
`tests/_uds_client.py`；`real_daemon` session fixture（经 `start.sh` 拉起、真实
socket/端口）在 `tests/conftest.py`。

开发者环境搭建、测试命令、lint 与代码风格见
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)。

## 状态

里程碑进度：

- **M0** — 工程骨架（本发布）：守护进程、UDS JSON-RPC、fusion-mlx 音频通道探测、
  `/health`、`start.sh` 生命周期、单元测试。**done**
- **M1** — fusion-mlx 委托桥（经 HTTP 的 transcribe/synthesize）。**done**
  - TTS 已 real-load 验证：经 UDS JSON-RPC 的 `speech.synthesize` → fusion-mlx
    Qwen3-TTS，5.3 s 内 388 KB WAV；靠探测驱动的模型回退开箱即用。
  - STT 已 real-load 验证：经 UDS JSON-RPC 的 `speech.transcribe` → fusion-mlx
    `openai/whisper-tiny`，`assistant_bank_en.wav` → 1.5 s 内得到 "Welcome to First
    National Bank! How may I assist you today?"。
  - 需要一个上游 fusion-mlx 发现修复（issue #649）：`_is_hf_cache_mlx_compatible`
    拒绝 HF 格式音频仓库（`openai/whisper-tiny` 带 HF `model.safetensors` + 完整
    tokenizer，无 MLX metadata，仓库不在 `mlx-community/` 下）。`mlx_audio` 原生
    能加载 HF 格式音频权重，故修复新增一个音频模型分支。注：`mlx-community/whisper-*-mlx*`
    仓库不带任何 tokenizer 文件 → 对 `mlx_audio.stt` 不可用（空 vocab →
    `IndexError`）；用 `openai/whisper-*`。追踪：issue #649 + fusion-mlx 上的 PR，
    按上游优先工作流。
- **M2** — 实时麦克风采集 + 流式 ASR。**done**
  - 环形缓冲：无锁 SPSC numpy 环，过载丢最旧（采集永不阻塞）。AudioSource 封装
    `sounddevice` 16 kHz 单声道（懒加载，无头安全）。
  - VAD：两级能量 RMS 门 + Silero ONNX（`silero_vad_16k_op15.onnx`）。64 采样滚动
    上下文前缀必需 —— 缺之概率对所有音频坍缩到 ~0.001。模型缺失时 fail-soft 回退
    到纯能量。
  - StreamingListener：drain 镜像采样，VAD 按语音起始 / 语音后静音分段，段 PCM →
    16-bit WAV → `transcribe_stream` → `speech.asr` `final` 通知。<0.1s 段跳过。
  - 实时推送：WebSocket `/stream/stt`，每订阅者队列 fan-out（满则丢最旧）；
    `speech.start_listening`/`stop_listening` RPC。
  - 已验证：55 测试绿，含对语音 WAV fixture 的 real-load Silero，以及端到端
    start → capture → transcribe → final-to-subscriber 集成测试。
- **M3** — 热词唤醒。**done**
  - `wake_word.py`：openwakeword ONNX 关键词 spotting（默认 `hey_jarvis`）。复用
    M2 麦克风环；drain 读 1280 采样（80ms）帧 → `openwakeword.Model.predict`。得分
    ≥ 阈值触发一次 `on_wake`，随后去抖冷却阻止重复。`pause()`/`resume()` 在活跃
    ASR 期间挂起检测（无 SPSC 读竞争 —— 暂停的 drain 停止读环）。Fail-soft：
    缺 openwakeword/models → `enabled=False`，守护进程无损。
  - 模型缓存：唤醒词 + `melspectrogram.onnx` + `embedding_model.onnx` 首次激活时
    从 openWakeWord GitHub releases 下载到 `~/.fusion-speech/models/wakeword/`
    （URL 可配镜像）。
  - RPC：`speech.wake {arm}` 激活/解除；`speech.set_hotwords {hotwords}` 运行时
    换词（仅预训练名）。共存：唤醒 → `speech.wake` 通知 → 暂停唤醒 → 为监听窗口
    自动 `start_listening` → `stop_listening` → 恢复唤醒。
  - 已验证：72 测试绿，含 real-load 测试（下载 hey_jarvis ONNX，喂 2s 静音，断言
    无误唤醒）。
- **M4** — 流式 TTS 播放。**done**
  - `tts_playback.py`：`StreamingPlayer` 消费 `mlx_bridge.synthesize_stream`，
    24k→16k 重采样，喂有界队列，写任务排入 `sounddevice.OutputStream`（首块即启 =
    低 TTFA）。取消丢弃 pending + 关流。设备错误 fail-soft。
  - 麦克风守卫：说话时唤醒暂停；若 ASR 正在监听，先停后启 —— 无自触发、无回声。
    复用 M2/M3 原语。
  - RPC：`speech.speak`、`speech.stop_tts`；`speech.tts` 事件
    （starting/speaking/done/cancelled/error）在 `/stream/stt`。
  - 已验证：~80+ 测试绿，含 real-load（经 fusion-mlx 的 Kokoro，假输出，无头）。

### 加固（P0-P3 审计轮次）

对抗性审计（`audit/fusion-speech-audit-0827.md`，13 项发现跨
FATAL/LOGIC/ARCH/PERF/MAINT）已全修。无公开契约改动；本轮收紧安全、正确性与
资源处理：

- **F1 (FATAL)** — 唤醒永不触发：`wake_word._resolve_score_keys` 现读真实
  openwakeword `class_mapping`/`models` 键（`hey_jarvis_v0.1`）按前缀匹配，取代
  `splitext(word)[0]`（后者把 `.1` 剥掉）。
- **F2 (FATAL)** — IPC 鉴权 fail-open：`_verify_peer_uid` 在 socket 缺失时拒绝
  （fail-closed，原为 `return True`）。
- **F3 (FATAL)** — 无界行 DoS：`handle_client` 把行封顶 1 MiB，溢出则关连接
  （bytearray 缓冲，无 O(n²) 拷贝）。
- **M1** — HTTP 重试：`mlx_bridge` 重试瞬态失败（5xx / `TransportError` /
  `TimeoutException`），退避 0.5/1.0/2.0s；4xx 不重试；流重试仅连接（无重复播放）。
- **M2** — 输入大小限制：`speech.transcribe` 拒绝 >25 MB 音频，`speech.speak`/
  `synthesize` 拒绝 >5000 字符文本（`-32000`）。
- **M3** — socket unlink TOCTOU：`server.start` 对陈旧 socket unlink 抑制
  `FileNotFoundError`。
- **L1/L2/L3** — 唤醒自动监听窗口在 ASR `final` 时可中断；唤醒恢复对 `None` 做守卫；
  订阅者队列有界 64（丢最旧）。
- **A1** — 在途 transcribe 任务被跟踪、sem 限流（2），并在 `stop()` 时取消/清理
  （无孤儿）。
- **A2/A3/P2-1** — 实时采集回调写 view（无逐帧分配）；共享环不被借用监听器关闭；
  `write()` 中环溢出路径不再在 PortAudio 线程上记 I/O 日志（仅计数器；溢出在
  `stop()` 和 `stats()` 中暴露）。环通知器在生产中未设 —— 消费者轮询环
  （`_drain_loop` 每 20ms），故通知器永不从实时回调调用。
- **P1** — TTS 重采样用 `scipy.signal.resample_poly`（抗混叠）；scipy 缺失时
  fail-soft 到 `np.interp`。队列上限钳到 [1, 10s]。
- **P2** — VAD `feed` 用预分配缓冲（无二次 realloc）。
- **P3** — `collect_pcm` 按整块裁剪（无第二次全拷贝）。

已验证：93 测试绿，含对抗性门禁套件（`tests/test_audit_fixes.py`）与 real-load
门禁（openwakeword ONNX 静音、fusion-mlx STT/TTS）。

### 架构审计（5 波，20 项发现）

架构审计（`audit/fusion-speech-audit-report-0827.md`，结论：未达商用就绪）已全
部修复。20 项发现：5 架构 (H)、7 运行时风险 (R)、8 工程缺陷 (E)。无公开 RPC 契约
改动；唯一新增（`speech.reprobe`）非破坏性。组件拆分见 `docs/ARCHITECTURE.md` §3a，
逐波分解见 §12（M4-加固）。

- **Wave 2 — god-object 拆分 (H1/H3/H4)**：`SpeechDaemon` 拆为薄 facade + 5 个叶
  服务（`AsrService`/`WakeService`/`TtsService`/`AudioBus`/`EventBus`）位于
  `fusion_speech/components/`。麦克风所有权集中到 `AudioBus`（acquire/release，
  无共享源竞态 peek）。引擎模块的懒加载提升为顶层（import 图验证无环）。
- **Wave 3 — 并发正确性 (H5/R3/R4/E7)**：每服务 `asyncio.Lock` 守护状态变更操作；
  facade `_handshake_lock` 串行化 wake→ASR 握手（消除双唤醒 check-then-act 撕裂）。
  ASR-final 握手帧经每握手 token 注册表带外投递（慢 TTS 丢最旧不再丢 final 帧）。
  `_on_close` 任务创建有界（Semaphore(4)，溢出丢弃 + 日志）。新增 6 个并发测试。
- **Wave 4 — 可观测性 + 关停 + 持久化 (E2/R6/E8/E6)**：零依赖 `/metrics`（见上）。
  有界关停排空：广播 `speech.shutdown`，等 WS 关闭带 10s 上限，取消掉队者。
  `start.sh` stop 宽限 3s → 15s 以吸收排空。持久化
  `~/.fusion-speech/run/state.json`（`start_time`/`restart_count`）；崩溃重启时
  守护进程广播 `speech.restarted`。
- **Wave 5 — config/probe/error/wake-frame (E4/E1/E3/R5/H2/R7/E5)**：
  `SpeechConfig.__post_init__` fail-loud 校验（sample_rate==16000、channels==1、
  端口范围、阈值、固定 1280 唤醒帧）—— 非 16000 启动现改为失败，而非静默腐蚀
  VAD/wake。`speech.reprobe` RPC + 后台重探循环（消除陈旧 `audio_lane`）。畸形
  HTTP-200 body（非 JSON STT、零速率 WAV TTS）抛带 status 的 `AudioBridgeError`，
  而非裸 `JSONDecodeError`/垃圾播放。`speech.speak` 后唤醒/ASR 恢复失败广播带
  `degraded: true` 的 `speech.tts` 错误事件并锁存 `tts.degraded`（干净恢复时清除）。
  keepalive 是独立 `speech.keepalive` 事件类型。唤醒 drain 读固定 1280 采样帧
  （原为可变 `available % 1280` → openwakeword 概率腐蚀）。keepalive/队列深度现
  配置驱动。

已验证：147 测试绿（`tests/test_wave5.py`、`tests/test_config_validation.py`、
`tests/test_r7_fixed_chunk_read.py`、`tests/test_concurrency.py`、
`tests/test_metrics.py`、`tests/test_state.py`、`tests/test_r6_graceful_drain.py`
+ 先前门禁套件），ruff clean。

- **M5** — fusion-studio 集成
  - 端口 `11465` 注册到 `architecture/port-registry.yaml`（check-ports CI 对
    fusion-speech 绿）。
  - 已提上游 issue：dahai80/fusion-studio#337 —— 请求 `SpeechBridge` Swift 类
    （UDS JSON-RPC 客户端）+ 麦克风权限 + `/stream/stt` 事件订阅。Swift 实现不在
    本工程范围（按工作流上游 issue 优先 → PR）。
- **M6** — 测试、集成、文档。**done**
  - Real-load 全链路集成门禁（`tests/test_m6_real_load_integration.py`）：
    基于 UDS JSON-RPC 的 5 步 —— ping/status、transcribe、synthesize、speak、
    wake 激活/解除 —— 经 `real_load` 标记可选（见[测试](#测试)）。
  - UDS 客户端 helper `tests/_uds_client.py`（换行分隔 JSON-RPC 2.0，4 MiB
    readline 上限以容纳大 base64 载荷）。`tests/conftest.py` 的 `real_daemon` session
    fixture 在真实端口 11465 + 真实 socket 上拉起守护进程，剥除 autouse 的每测试
    隔离变量。
  - 已提上游阻塞：fusion-mlx issue #668 —— `mlx-audio` Whisper
    `non_speech_tokens` `IndexError` 使 STT 对每个模型返回 500；步骤 2 跳过
    （非 xfail），故门禁上游侧保持绿，修复落地后翻转为真实断言。TTS 路径不受影响。
  - 已验证：148 测试绿，5 跳过（real_load 门禁），ruff clean。

- **0.1.1** — 双语文档（中文 `README_CN.md`）、场景化
  [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md) 与开发者
  [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)；0.1.0rc1 之上的补丁。无运行时
  改动。已知上游限制：fusion-mlx issue #668（STT 500）—— 见测试。

License: Apache-2.0.
