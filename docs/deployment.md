# 部署

## Docker

需要 Docker Compose 2.24+。GPU 还需 NVIDIA 驱动和 NVIDIA Container Toolkit，支持 Linux amd64；CPU 支持 Linux amd64/arm64，x86 要求 x86-64-v3（含 AVX2/FMA）。macOS/Windows 可使用 Docker Desktop 的 Linux 容器，Windows 尚未实机验收。

默认 GPU 使用 `compose.yml`。首次部署先构建、下载模型，再启动：

```bash
docker compose build
# 模型准备时可写，正式服务中 Nemotron 仍只读挂载
docker compose run --rm --no-deps \
  -v ./models/nemotron-3-diarization:/app/models/nemotron-3-diarization \
  --entrypoint python asr -m app.utils.download_models
docker compose up -d
docker compose logs -f asr
```

CPU 对以上每条命令添加 `-f compose.cpu.yaml`，例如 `docker compose -f compose.cpu.yaml build`。两份配置独立使用，镜像都叫 `quantatrisk/qwen3-asr:latest`；切换后端执行对应的 `up -d --build`，避免复用另一后端的同名镜像。已准备完整模型时可跳过下载。

默认端口 17003。浏览器录音需要 localhost 或 HTTPS；反向代理须支持 WebSocket Upgrade，并给长录音请求足够的上传大小和超时时间。

## 配置

无需 `.env` 即可使用默认值。自定义时复制 `.env.example` 为 `.env`，只取消需要的注释。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `ASR_PORT` | `17003` | 宿主机端口 |
| `ASR_GPU` | `0` | 宿主机 GPU 编号 |
| `API_KEY` | 空 | 公共接口鉴权 |
| `HF_HUB_OFFLINE` | `0` | 模型齐全后设为 `1` 禁止下载 |
| `HF_ENDPOINT` | 官方地址 | 可选 Hugging Face 镜像 |
| `R2T2_GPU_MEMORY_UTILIZATION` | `0.30` | R2T2 显存比例 |
| `FORCED_ALIGNER_GPU_MEMORY_UTILIZATION` | `0.15` | 对齐模型显存比例 |
| `R2T2_CPU_THREADS` | `8` | Rust CPU 线程 |
| `OPENBLAS_NUM_THREADS` | GPU `4`、CPU `8` | 矩阵运算线程 |

显存比例均相对于整张显卡，需另给 Nemotron FP32 和运行时留空间。CPU 线程数按目标机器调节。

应用参数通过 `.env` 传入容器；仅在宿主机 shell 中 export 不会自动传入。Compose 固定 `DEVICE`，会话数默认 GPU 4、CPU 1。需要时可在 `.env` 添加 `R2T2_MAX_SESSIONS`、`R2T2_MAX_MODEL_LEN`（16384）、`R2T2_ENFORCE_EAGER`（0）或 `R2T2_CHUNK_SECONDS`（GPU 0.16、CPU 0.64）。内部鉴权可设置 `R2T2_INTERNAL_TOKEN`。

## 模型与运行数据

- `models/huggingface`：R2T2 和强制对齐模型缓存。
- `models/modelscope`：VAD 与标点模型缓存。
- `models/nemotron-3-diarization`：Nemotron，正式服务只读挂载。
- `logs`：应用日志；`.cache/vllm`：GPU 编译缓存。

临时音频留在容器内。离线部署复制完整 `models/` 后，在 `.env` 设置 `HF_HUB_OFFLINE=1`；模型缺失时启动失败。

R2T2 revision 固定为 `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`，Nemotron revision 为 `f667ed73aee57d40cc39428eb768b4fd87a0a29e`。Python 依赖由 `uv.lock` 锁定。

## 原生 CPU

需要 Python 3.11–3.12、uv、Rust、FFmpeg；Linux 还需 libsndfile 和 OpenBLAS 开发库（Debian/Ubuntu：`libsndfile1 libopenblas-dev`）。macOS 支持 Apple Silicon。

```bash
uv sync --frozen --extra cpu  # macOS 去掉 --extra cpu
./scripts/build-rust.sh
DEVICE=cpu ./scripts/prepare-models.sh
HF_HOME="$PWD/models/huggingface" \
MODELSCOPE_CACHE="$PWD/models/modelscope/hub" \
DEVICE=cpu OPENBLAS_NUM_THREADS=8 uv run --no-sync python start.py
```

原生服务端口为 8000。使用自定义 `CARGO_TARGET_DIR` 时，另设置 `R2T2_CPU_LIBRARY_PATH` 指向构建的动态库。原生 Linux GPU 使用 `uv sync --frozen --extra cuda`；macOS 默认 CPU，Linux 默认 CUDA，不自动降级。

## 验证与边界

```bash
docker compose ps
curl http://localhost:17003/stream/v1/asr/health
```

配置鉴权时添加 `Authorization: Bearer <API_KEY>`。文件转写示例见 [README](../README.md)，实时与并发验收见 [benchmark](../scripts/benchmark/README.md)。

启动器依次加载私有推理引擎与公共 API，任一进程异常则关闭全部子进程。私有引擎只监听容器内 `127.0.0.1:8001`；健康检查等待模型就绪，启动宽限期为 600 秒。

CPU 推理不可中途抢占，同时跑实时与离线会增加延迟。历史 M5 Pro 短样本纯识别 RTF 约 0.08–0.13；Linux amd64 完整 5 分钟录音约耗时 310 秒，不能按 Mac 结果承诺实时性能。Linux arm64 已验证构建和动态库加载，完整模型链路尚未验收。

Nemotron 最多支持 8 个说话人。混合语言可能漏词，重叠发言仍受单路 ASR 限制；时间戳边界合法不代表人工对齐准确。模型质量和长时并发须在目标硬件用真实录音验收。
