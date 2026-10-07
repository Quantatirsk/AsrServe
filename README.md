<div align="center">

<h1>AsrServe</h1>
<h3>Ready-to-use Local Speech Recognition API Service</h3>

Self-hosted realtime and offline speech recognition. AsrServe always ships state-of-the-art open models on the most efficient inference path available, and treats recognition quality as a first-class citizen.

---

![Static Badge](https://img.shields.io/badge/Python-3.11--3.12-blue?logo=python)
![Static Badge](https://img.shields.io/badge/CUDA-13.0-%2376B900?logo=nvidia&logoColor=white)
![Static Badge](https://img.shields.io/badge/vLLM-0.30-%23EE4C2C)
![Static Badge](https://img.shields.io/badge/macOS-Apple_Silicon-black?logo=apple)

</div>

## Live Demo Site

- **Web Demo**: https://asr.ieeio.com

## Demo

https://github.com/user-attachments/assets/ca6ed8ca-033d-4c58-85bb-2016d66151ea

[Watch or download the original video](https://media.cdn.ieeio.com/qwenasr_client_demo.mp4)

## Contact Author

- **Email**: [pengzhia@gmail.com](mailto:pengzhia@gmail.com)
- **WeChat**:

<img src="./demo/contact.jpg" alt="WeChat QR code" width="220">

## Release 1.0.5

`v1.0.5` restores punctuation throughout file transcriptions, exposes recording context and hotwords, and adds OpenAI Realtime transcription.

- **File punctuation**: CT-Transformer restores sentence boundaries using whole-file text context before alignment and speaker assignment. It runs on CPU and preserves the original words, case, spaces and numbers. The pinned punctuation checkpoint is downloaded on first startup if it is not already cached.
- **File hotwords**: `/v1/audio/transcriptions` accepts `hotwords` and uses `prompt` for recording context, for both uploads and `audio_address`. Their combined limit is 2048 characters; they guide recognition without forcing replacements.
- **OpenAI Realtime**: `/v1/realtime?intent=transcription` streams text as audio arrives, supports manual `commit`, `clear` and multiple turns, and uses the existing API key authentication. See the [protocol and SDK examples](docs/openai-realtime.md) for the supported transcription flow.
- **Audio and alignment**: FFmpeg normalizes input audio. Optional uniform word timing avoids loading ForcedAligner; responses identify these timestamps as estimates.
- **Native macOS service**: Apple Silicon can run the service through a user LaunchAgent, with install, update, restart and uninstall commands.
- **Images**: `quantatrisk/asrserve:1.0.5-gpu` (amd64), `:1.0.5-cpu` (amd64/arm64) and `:1.0.5-ascend` (arm64, Ascend 910B branch).

## Release 1.0.4

`v1.0.4` replaces the whole model stack on both the backend and the browser client. In our own tests against `v1.0.3`, recognition accuracy improved by about **20%** and end-to-end efficiency by about **40%**.

Breaking changes:

- **Project rename**: `qwen3-asr` is now **AsrServe**. The GitHub repository, Python package and Docker images (`quantatrisk/asrserve`) use the new name; old GitHub URLs redirect automatically.
- **ASR model**: Qwen3-ASR 1.7B/0.6B is replaced by [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2). Offline and realtime share one R2T2 engine (vLLM on CUDA, vendored Rust on CPU); the `model` request field no longer switches models.
- **Speaker diarization**: CAM++ is replaced by [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (up to 8 speakers). Realtime streams now also carry per-utterance speaker labels.
- **Removed models**: FSMN VAD, the three CAM++ models and the Qwen3-ASR checkpoints are gone. Nemotron speech activity drives offline segmentation.
- **Removed API**: the Alibaba Cloud compatible REST API (`/stream/v1/asr*`) is removed. Use the OpenAI-compatible `/v1/audio/transcriptions` and the native `/v1/stream` WebSocket.
- **Deployment**: images are published on Docker Hub as `quantatrisk/asrserve:gpu` (amd64), `:cpu` (amd64/arm64) and `:ascend` (arm64, Ascend 910B branch), plus `1.0.4-*` version tags; `build.sh` builds from source. Compose files are `compose.yml` (GPU) and `compose.cpu.yml` (CPU) and no longer build. The only mount is `./models`. `docker-compose*.yml`, `deploy/prepare.sh` and the model export option are removed. The old `quantatrisk/qwen3-asr` Docker Hub repository is no longer updated.
- **Runtime**: the CUDA image moves to CUDA 13.0 and vLLM 0.30; the service listens on port `17003`.

Older release notes: [GitHub Releases](https://github.com/Quantatirsk/AsrServe/releases).

## Features

- **SOTA models, quality first**: Confucius4-R2T2 recognition, CT-Transformer offline punctuation, Nemotron speaker diarization and Qwen3-ForcedAligner word timestamps today; the stack moves whenever a better open model appears
- **First-class macOS support**: runs natively on Apple Silicon through a bundled Rust inference backend, no Docker or GPU required
- **Runs anywhere else too**: NVIDIA GPU (vLLM) and Linux CPU (amd64/arm64)
- **Realtime and offline** in one service, sharing one set of R2T2 weights
- **Speaker diarization** for files and live streams via Nemotron
- **Word timestamps** via Qwen3-ForcedAligner
- **OpenAI compatible** `/v1/audio/transcriptions`, works with the OpenAI SDK
- **OpenAI Realtime transcription** `/v1/realtime`, with incremental text, manual commit and multiple turns per connection
- **Browser recording page** at `/realtime`

## Quick Start

```bash
docker compose up -d                        # GPU: quantatrisk/asrserve:gpu
docker compose -f compose.cpu.yml up -d     # CPU: quantatrisk/asrserve:cpu (amd64/arm64)
```

Images are pulled from Docker Hub on first start; upgrade with `docker compose pull && docker compose up -d`. To build from source, run `./build.sh` (CPU: `TARGET=cpu ./build.sh`). Ascend 910B lives on the [`ascend-910b`](https://github.com/Quantatirsk/AsrServe/tree/ascend-910b) branch.

On macOS, run natively instead (see [Deployment](docs/deployment.md#原生-cpu)):

```bash
uv sync --frozen && ./scripts/build-rust.sh
HF_HOME="$PWD/models/huggingface" uv run --no-sync python start.py   # http://localhost:8000
```

The first start downloads the pinned models into `./models`. Default URL: `http://localhost:17003` (recording page `/realtime`, API docs `/docs`, health `/health`).

`.env` is optional; copy `.env.example` and uncomment what you need (API key, offline mode, GPU memory, CPU threads).

For offline hosts, pre-download on a networked machine, copy the repository including `models/`, and set `HF_HUB_OFFLINE=1`:

```bash
./scripts/prepare-models.sh                                   # uses the :gpu image, no GPU needed
IMAGE=quantatrisk/asrserve:cpu ./scripts/prepare-models.sh   # or the :cpu image
```

## File Transcription

```bash
curl http://localhost:17003/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=confucius4-r2t2 \
  -F 'hotwords=光谷水投,武汉新城,东湖高新区' \
  -F response_format=verbose_json \
  -F enable_speaker_diarization=true \
  -F word_timestamps=true
```

With `API_KEY` set, add `Authorization: Bearer <API_KEY>`. Speaker paragraphs follow the main speaker; raw overlapping activity is returned in `speaker_segments`.

`hotwords` supplies names and terminology; `prompt` supplies recording context. Both work for uploads and `audio_address`. Values are trimmed and combined in `prompt`, `hotwords` order with a newline, up to 2048 characters; longer requests return HTTP 400. Every ASR chunk receives the context as recognition guidance, without forced replacements or keyword weights.

File transcription restores punctuation using whole-file text context before alignment and speaker assignment. CT-Transformer runs on CPU through FunASR, preserving the original text, case, spaces, numbers and existing sentence endings. Its pinned ModelScope checkpoint is downloaded into `models/`; details are in [Deployment](docs/deployment.md).

## Realtime Transcription

Connect to `ws://localhost:17003/v1/stream` and send 16 kHz mono int16 PCM. See the [realtime protocol](docs/realtime.md).

For OpenAI clients, connect to `/v1/realtime?intent=transcription` and send Base64 PCM16 at 24 kHz. Text streams as audio arrives; the client commits each turn. Prompts and keywords guide recognition. See [OpenAI Realtime transcription](docs/openai-realtime.md) for SDK examples, authentication and supported features.

## Documentation

- [Deployment](docs/deployment.md): configuration, offline setup, native CPU, limits
- [Realtime protocol](docs/realtime.md)
- [OpenAI Realtime transcription](docs/openai-realtime.md)
- [Acceptance scripts](scripts/benchmark/README.md)

## Acknowledgements

- [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2): speech recognition; source attribution in `deploy/R2T2-NOTICE`, weights under their own MODEL_LICENSE
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR): forced alignment
- [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization): speaker diarization
- [QwenASR](https://github.com/huanglizhuo/QwenASR): vendored CPU Rust backend
