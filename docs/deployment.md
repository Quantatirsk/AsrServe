# 部署

## Docker

需要 Docker Compose 2.24+。GPU 还需 NVIDIA 驱动和 NVIDIA Container Toolkit，支持 Linux amd64；CPU 支持 Linux amd64/arm64，x86 要求 x86-64-v3（含 AVX2/FMA）。macOS/Windows 可使用 Docker Desktop 的 Linux 容器，Windows 尚未实机验收。

Compose 只引用镜像，不含构建参数：

```bash
docker compose up -d                        # GPU
docker compose -f compose.cpu.yml up -d     # CPU
docker compose logs -f asr
```

| 镜像 | 架构 | 版本标签 |
| --- | --- | --- |
| `quantatrisk/asrserve:gpu` | linux/amd64 | `1.0.4-gpu` |
| `quantatrisk/asrserve:cpu` | linux/amd64、linux/arm64 | `1.0.4-cpu` |

本地没有镜像时自动从 Docker Hub 拉取，升级执行 `docker compose pull && docker compose up -d`。需要固定版本时把 compose 的 `image` 改为版本标签。从源码构建：`./build.sh` 生成 `:gpu`，`TARGET=cpu ./build.sh` 生成本机架构的 `:cpu`，同名标签会覆盖拉取的镜像。首次启动自动下载缺失模型，健康检查宽限 600 秒。

### 离线部署

在有网络的机器拉取镜像并预下载模型，再用 `docker save`/`docker load` 迁移镜像，把整个仓库目录（含 `models/`）拷到目标机：

```bash
./scripts/prepare-models.sh                                   # 用 :gpu 镜像
IMAGE=quantatrisk/asrserve:cpu ./scripts/prepare-models.sh   # 用 :cpu 镜像
```

脚本在服务镜像内下载并校验，只挂载 `./models`，不需要 GPU；两个镜像下载的模型相同。需要镜像站时加 `HF_ENDPOINT=https://hf-mirror.com`。目标机在 `.env` 设置 `HF_HUB_OFFLINE=1`，模型缺失时启动失败而不是联网。

默认端口 17003，需要改端口时直接修改 compose 的 `ports`。浏览器录音需要 localhost 或 HTTPS；反向代理须支持 WebSocket Upgrade，并给长录音请求足够的上传大小和超时时间。

## 配置

无需 `.env` 即可使用默认值。自定义时复制 `.env.example` 为 `.env`，只取消需要的注释。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `ASR_GPU` | `0` | 宿主机 GPU 编号 |
| `API_KEY` | 空 | 公共接口鉴权 |
| `HF_HUB_OFFLINE` | `0` | 模型齐全后设为 `1` 禁止下载 |
| `HF_ENDPOINT` | 官方地址 | 可选 Hugging Face 镜像 |
| `R2T2_GPU_MEMORY_UTILIZATION` | `0.30` | R2T2 显存比例 |
| `FORCED_ALIGNER_GPU_MEMORY_UTILIZATION` | `0.15` | 对齐模型显存比例 |
| `R2T2_CPU_THREADS` | `8` | Rust CPU 线程 |
| `OPENBLAS_NUM_THREADS` | GPU `4`、CPU `8` | 矩阵运算线程 |

显存比例均相对于整张显卡，需另给 Nemotron FP32 和运行时留空间。CPU 线程数按目标机器调节。

应用参数通过 `.env` 传入容器；仅在宿主机 shell 中 export 不会自动传入。镜像固定 `DEVICE`，会话数默认 GPU 4、CPU 1。需要时可在 `.env` 添加 `R2T2_MAX_SESSIONS`、`R2T2_MAX_MODEL_LEN`（16384）、`R2T2_ENFORCE_EAGER`（0）或 `R2T2_CHUNK_SECONDS`（GPU 0.16、CPU 0.64）。内部鉴权可设置 `R2T2_INTERNAL_TOKEN`。

## 模型与运行数据

唯一挂载 `./models:/app/models`：`models/huggingface` 存放 R2T2 与强制对齐模型，`models/nemotron-3-diarization` 存放 Nemotron。日志用 `docker compose logs`，临时音频与编译缓存留在容器内。

R2T2 revision 固定为 `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`，Nemotron revision 为 `f667ed73aee57d40cc39428eb768b4fd87a0a29e`。Python 依赖由 `uv.lock` 锁定。

## 原生 CPU

需要 Python 3.11–3.12、uv、Rust、FFmpeg；Linux 还需 libsndfile 和 OpenBLAS 开发库（Debian/Ubuntu：`libsndfile1 libopenblas-dev`）。macOS 支持 Apple Silicon。

```bash
uv sync --frozen --extra cpu  # macOS 去掉 --extra cpu
./scripts/build-rust.sh
HF_HOME="$PWD/models/huggingface" \
DEVICE=cpu OPENBLAS_NUM_THREADS=8 uv run --no-sync python start.py
```

原生服务端口为 8000，首次启动同样自动下载模型到 `models/`。使用自定义 `CARGO_TARGET_DIR` 时，另设置 `R2T2_CPU_LIBRARY_PATH` 指向构建的动态库。原生 Linux GPU 使用 `uv sync --frozen --extra cuda`；macOS 默认 CPU，Linux 默认 CUDA，不自动降级。

## 验证与边界

```bash
docker compose ps
curl http://localhost:17003/health
```

配置鉴权时添加 `Authorization: Bearer <API_KEY>`。文件转写示例见 [README](../README.md)，实时与并发验收见 [benchmark](../scripts/benchmark/README.md)。

启动器依次加载私有推理引擎与公共 API，任一进程异常则关闭全部子进程。私有引擎只监听容器内 `127.0.0.1:8001`；健康检查等待模型就绪，启动宽限期为 600 秒。

CPU 推理不可中途抢占，同时跑实时与离线会增加延迟。历史 M5 Pro 短样本纯识别 RTF 约 0.08–0.13；Linux amd64 完整 5 分钟录音约耗时 310 秒，不能按 Mac 结果承诺实时性能。Linux arm64 已验证构建和动态库加载，完整模型链路尚未验收。

Nemotron 最多支持 8 个说话人。混合语言可能漏词，重叠发言仍受单路 ASR 限制；时间戳边界合法不代表人工对齐准确。模型质量和长时并发须在目标硬件用真实录音验收。
