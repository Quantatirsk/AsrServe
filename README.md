# Qwen3 ASR

基于 Confucius4-R2T2 的实时与离线语音识别服务，支持 NVIDIA GPU、Linux CPU（amd64/arm64）和 macOS Apple Silicon CPU。

- 实时与离线共用一份 R2T2 权重；离线独立识别原始录音。
- Nemotron 提供说话人分离，Qwen3-ForcedAligner 提供字词时间戳。
- 提供 OpenAI 兼容转写接口、阿里云兼容接口和浏览器录音页面。

## Docker 启动

先按[部署说明](docs/deployment.md)准备模型，然后启动：

```bash
# 默认 GPU：compose.yml
docker compose up -d --build

# CPU：compose.cpu.yaml
docker compose -f compose.cpu.yaml up -d --build
```

两份配置分别使用，不要叠加。镜像统一为 `quantatrisk/qwen3-asr:latest`；切换后端需要重新构建。

默认地址为 `http://localhost:17003`：录音页面 `/realtime`，API 文档 `/docs`，健康检查 `/stream/v1/asr/health`。

`.env` 可选；需要鉴权、离线模式或调整显存/线程时，复制 `.env.example` 并取消相应注释。

## 文件转写

```bash
curl http://localhost:17003/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=confucius4-r2t2 \
  -F response_format=verbose_json \
  -F enable_speaker_diarization=true \
  -F word_timestamps=true
```

离线始终用 Nemotron 的活动区间切段。`enable_speaker_diarization=false` 仅关闭文字说话人归属和说话人信息；只有请求字词时间戳时才做对齐。成功检测但活动为空时保留整段音频、按最大长度切分交给 ASR，避免将漏检当成静音；Nemotron 推理失败则报错。实时低能量收尾检测保持独立。

配置 `API_KEY` 后添加 `Authorization: Bearer <API_KEY>`。`model` 参数不切换模型，服务始终使用 R2T2。说话人段落表示主讲者，重叠活动另存于 `speaker_segments`；分离说话人不等于分离干净音轨。

离线会合并连续的同一说话人片段，合并后时间跨度最多 30 秒，中间的自然停顿不单独拆段。中间插话片段合计不足 2 秒（不计片段间停顿）、随后回到同一主讲者且各处间隔最多 1 秒时，并入该主讲者段落，不要求主讲者累计发言满 4 秒；保留文字和字词绝对时间。`word_timestamps` 只控制公开字词时间戳，不改变说话人分组。 不足 2 秒的连续 Unknown 片段，在间隔不超过 1 秒且候选名单支持时优先并入前一说话人，否则尝试后一说话人；没有依据或持续较长时保留 Unknown。原始说话人活动仍保存在 `speaker_segments`。

实时接口 `/v1/stream` 使用 16 kHz 单声道 PCM，协议见[实时转写](docs/realtime.md)。

## 本地开发

需要 Python 3.11–3.12、uv 和 FFmpeg；CPU 后端还需要 Rust。

```bash
uv sync --frozen --extra cuda  # Linux CPU 改为 --extra cpu；macOS 去掉 --extra cuda
uv run --no-sync python start.py
uv run --no-sync python -m pytest tests
```

CPU 原生构建、模型缓存和平台限制见[部署说明](docs/deployment.md)；验收脚本见 [scripts/benchmark](scripts/benchmark/README.md)。

## 上游

- [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2)：语音识别；源码归属见 `deploy/R2T2-NOTICE`，权重遵循独立 MODEL_LICENSE。
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)：强制对齐。
- [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)：说话人分离。
