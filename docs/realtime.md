# 实时转写 / Realtime Transcription

实时和离线共用一份 R2T2 权重，每个连接独立保存音频、解码状态和说话人缓存。GPU 使用 vLLM，CPU 使用 Rust；启动与资源配置见[部署说明](deployment.md)。

Realtime and offline share the same R2T2 weights; each connection independently stores audio, decoding state, and speaker cache. GPU uses vLLM, CPU uses Rust; see [deployment instructions](deployment.md) for startup and resource configuration.

## WebSocket 协议 / WebSocket Protocol

连接 `ws://host:17003/v1/stream`。启用鉴权时使用 `Authorization: Bearer <API_KEY>`，浏览器可使用 `?token=...`。首条消息发送 `{}`，或带热词上下文：

Connect to `ws://host:17003/v1/stream`. When authentication is enabled, use `Authorization: Bearer <API_KEY>`; browsers can use `?token=...`. First message sends `{}`, or with hotword context:

```json
{"context":"网易有道，Qwen，R2T2"}
```

收到 `ready:true` 后发送 **16 kHz、单声道、int16 little-endian PCM** 二进制帧。建议每帧 160 ms（5120 字节），单帧最多 1 秒。结束时发送文本 `end`，继续接收至 `done:true`。不接受语言或采样参数，模型自动识别语言。

After receiving `ready:true`, send **16 kHz, mono, int16 little-endian PCM** binary frames. Recommended frame size is 160 ms (5120 bytes), maximum 1 second per frame. To finish, send text `end` and continue receiving until `done:true`. Does not accept language or sampling parameters; the model automatically detects language.

```json
{"delta":"Hello","audio_ms":960,"inference_ms":45.2,"done":false,"utterance":0,"utterance_end":false}
{"delta":".","audio_ms":1280,"inference_ms":48.1,"done":false,"utterance":0,"utterance_end":true}
{"utterance":0,"speaker":"说话人1"}
{"delta":"","audio_ms":1440,"inference_ms":3.0,"done":true,"utterance":1,"utterance_end":true,"text":"Hello."}
```

- 直接追加 `delta`，已发布文字不改写；最终 `text` 是完整文本。
- `audio_ms` 是已处理音频位置，不是文字时间戳；空增量可作为处理进度。
- `inference_ms` 包含该请求的调度和推理时间，不是端到端延迟。
- `utterance` 为停顿分隔的语句编号，`utterance_end` 表示该语句结束。

- `delta` is appended directly; previously emitted text is not rewritten; final `text` is the complete transcript.
- `audio_ms` is the processed audio position, not a text timestamp; empty deltas can serve as processing progress.
- `inference_ms` includes scheduling and inference time for that request, not end-to-end latency.
- `utterance` is the pause-separated utterance number; `utterance_end` indicates the end of that utterance.

错误返回 `{"code":"capacity_exceeded","error":"..."}` 并关闭连接。每会话最长 1 小时，输入队列最多 10 秒；超时、积压或断线会取消请求并释放名额。

Errors return `{"code":"capacity_exceeded","error":"..."}` and close the connection. Each session lasts at most 1 hour, input queue at most 10 seconds; timeout, backlog, or disconnection cancels the request and frees the slot.

## 说话人标签 / Speaker Labels

Nemotron 使用 1.04 秒缓冲模式，覆盖到语句尾部后发送该语句的主讲者标签。每个有文字的语句一个标签，通常比文字晚约 1–2 秒；全部标签先于 `done` 发送。`speaker:null` 表示无法判断或分离失败，文字转写继续。

Nemotron uses a 1.04-second buffer mode, sending the primary speaker label for an utterance after covering its tail. One label per utterance with text, typically arriving 1–2 seconds after the text; all labels are sent before `done`. `speaker:null` indicates unable to determine or separation failed; text transcription continues.

编号仅在本连接内有效，没有词级时间戳。无停顿的抢话会归到同一语句的主讲者，背景噪声可能推迟停顿检测和标签。保存录音后，以离线重新识别和说话人分配结果为准。

Numbers are only valid within this connection; there are no word-level timestamps. Overlapping speech without pauses is attributed to the same utterance's primary speaker; background noise may delay pause detection and labeling. After saving the recording, offline re-recognition and speaker assignment results are authoritative.

## 延迟与限制 / Latency and Limitations

GPU 默认每 160 ms 解码，CPU 每 640 ms 解码，可用 `R2T2_CHUNK_SECONDS` 调整；输入帧长度与解码间隔独立。音频窗口最多 16 秒，超过后移除最早 8 秒及相应文本前缀。语音之后至少 320 ms 的低能量音频触发语句收尾，结束连接前也会补齐未确认文字。

GPU decodes every 160 ms by default, CPU every 640 ms; adjustable with `R2T2_CHUNK_SECONDS`; input frame length and decode interval are independent. Audio window is at most 16 seconds; when exceeded, the earliest 8 seconds and corresponding text prefix are removed. At least 320 ms of low-energy audio after speech triggers utterance finalization; unconfirmed text is also completed before connection close.

GPU 默认支持 4 个会话，CPU 默认 1 个。GPU 离线最多并行识别 8 段，CPU 串行执行且不可中途抢占。共享推理、对齐和说话人模型会竞争资源，需按实际并发负载验收。首个解码窗口长度不等于首字延迟保证，混合语言、口音和重叠发言仍可能识别不完整。

GPU supports 4 sessions by default, CPU 1. GPU offline recognizes up to 8 segments in parallel; CPU executes serially and cannot be preempted mid-execution. Shared inference, alignment, and speaker models compete for resources; acceptance testing must be done with actual concurrent load. First decode window length does not guarantee first-word latency; mixed languages, accents, and overlapping speech may still result in incomplete recognition.

## 验证 / Verification

```bash
uv run --no-sync python -m scripts.benchmark.realtime_smoke zh.wav en.wav \
  --url http://127.0.0.1:17003 --concurrency 4 \
  --output benchmark_results/r2t2-acceptance.json
```

CPU 将并发改为 1，鉴权通过 `API_KEY` 环境变量提供。脚本覆盖真实节奏回放、语言切换、长音频、满载拒绝和断线回收；它检查接口与会话隔离，不代替人工准确率评测。

For CPU, change concurrency to 1; authentication is provided via the `API_KEY` environment variable. The script covers real-pace playback, language switching, long audio, full-capacity rejection, and disconnection recovery; it checks interface and session isolation but does not replace manual accuracy evaluation.

解码算法源自 [NetEase Youdao R2T2](https://github.com/netease-youdao/Confucius4-R2T2)，源码归属与许可见 `deploy/R2T2-NOTICE`。

The decoding algorithm is derived from [NetEase Youdao R2T2](https://github.com/netease-youdao/Confucius4-R2T2); source attribution and licensing are in `deploy/R2T2-NOTICE`.
