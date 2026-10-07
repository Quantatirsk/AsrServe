# 部署 / Deployment

## Docker

需要 Docker Compose 2.24+。GPU 还需 NVIDIA 驱动和 NVIDIA Container Toolkit，支持 Linux amd64；CPU 支持 Linux amd64/arm64，x86 要求 x86-64-v3（含 AVX2/FMA）。macOS/Windows 可使用 Docker Desktop 的 Linux 容器，Windows 尚未实机验收。

Requires Docker Compose 2.24+. GPU also requires NVIDIA drivers and NVIDIA Container Toolkit, supports Linux amd64; CPU supports Linux amd64/arm64, x86 requires x86-64-v3 (with AVX2/FMA). macOS/Windows can use Docker Desktop's Linux containers, Windows not yet verified on real hardware.

Compose 只引用镜像，不含构建参数：

Compose only references images, does not include build parameters:

```bash
docker compose up -d                        # GPU
docker compose -f compose.cpu.yml up -d     # CPU
docker compose logs -f asr
```

| 镜像 / Image | 架构 / Architecture | 版本标签 / Version Tag |
| --- | --- | --- |
| `quantatrisk/asrserve:gpu` | linux/amd64 | `1.0.5-gpu` |
| `quantatrisk/asrserve:cpu` | linux/amd64、linux/arm64 | `1.0.5-cpu` |
| `quantatrisk/asrserve:ascend` | linux/arm64（Ascend 910B） | `1.0.5-ascend` |

Ascend 910B 使用 [`ascend-910b`](https://github.com/Quantatirsk/AsrServe/tree/ascend-910b) 分支的 Compose 文件及部署说明。

本地没有镜像时自动从 Docker Hub 拉取，升级执行 `docker compose pull && docker compose up -d`。需要固定版本时把 compose 的 `image` 改为版本标签。从源码构建：`./build.sh` 生成 `:gpu`，`TARGET=cpu ./build.sh` 生成本机架构的 `:cpu`，同名标签会覆盖拉取的镜像。首次启动自动下载缺失模型，健康检查宽限 600 秒。

Images are automatically pulled from Docker Hub if not present locally. To upgrade, run `docker compose pull && docker compose up -d`. To pin a version, change the `image` in compose to a version tag. Build from source: `./build.sh` produces `:gpu`, `TARGET=cpu ./build.sh` produces `:cpu` for the local architecture; same-name tags overwrite pulled images. First start automatically downloads missing models; health check has a 600-second grace period.

### 离线部署 / Offline Deployment

在有网络的机器拉取镜像并预下载模型，再用 `docker save`/`docker load` 迁移镜像，把整个仓库目录（含 `models/`）拷到目标机：

On a machine with network access, pull the image and pre-download models, then use `docker save`/`docker load` to transfer the image, and copy the entire repository directory (including `models/`) to the target machine:

```bash
./scripts/prepare-models.sh                                   # 用 :gpu 镜像 / Use :gpu image
IMAGE=quantatrisk/asrserve:cpu ./scripts/prepare-models.sh   # 用 :cpu 镜像 / Use :cpu image
```

脚本在服务镜像内下载并校验，只挂载 `./models`，不需要 GPU；两个镜像下载的模型相同。需要镜像站时加 `HF_ENDPOINT=https://hf-mirror.com`。目标机在 `.env` 设置 `HF_HUB_OFFLINE=1`，模型缺失时启动失败而不是联网。

The script downloads and verifies models inside the service image, only mounts `./models`, and does not require a GPU; both images download the same models. To use a mirror, add `HF_ENDPOINT=https://hf-mirror.com`. On the target machine, set `HF_HUB_OFFLINE=1` in `.env` so that startup fails instead of attempting to connect to the network if models are missing.

默认端口 17003，需要改端口时直接修改 compose 的 `ports`。浏览器录音需要 localhost 或 HTTPS；反向代理须支持 WebSocket Upgrade，并给长录音请求足够的上传大小和超时时间。

Default port is 17003; to change the port, directly modify `ports` in compose. Browser recording requires localhost or HTTPS; reverse proxies must support WebSocket Upgrade and provide sufficient upload size and timeout for long recording requests.

## 配置 / Configuration

无需 `.env` 即可使用默认值。自定义时复制 `.env.example` 为 `.env`，只取消需要的注释。

No `.env` is required to use default values. To customize, copy `.env.example` to `.env` and uncomment only what you need.

| 参数 / Parameter | 默认值 / Default | 用途 / Purpose |
| --- | --- | --- |
| `ASR_GPU` | `0` | 宿主机 GPU 编号 / Host GPU number |
| `API_KEY` | 空 / empty | 公共接口鉴权 / Public API authentication |
| `HF_HUB_OFFLINE` | `0` | 模型齐全后设为 `1` 禁止下载 / Set to `1` to disable downloads after models are complete |
| `HF_ENDPOINT` | 官方地址 / official URL | 可选 Hugging Face 镜像 / Optional Hugging Face mirror |
| `R2T2_GPU_MEMORY_UTILIZATION` | `0.30` | R2T2 显存比例 / R2T2 GPU memory fraction |
| `FORCED_ALIGNER_GPU_MEMORY_UTILIZATION` | `0.15` | 对齐模型显存比例 / Alignment model GPU memory fraction |
| `R2T2_CPU_THREADS` | `8` | Rust CPU 线程 / Rust CPU threads |
| `OPENBLAS_NUM_THREADS` | GPU `4`、CPU `8` | 矩阵运算线程 / Matrix operation threads |

显存比例均相对于整张显卡，需另给 Nemotron FP32 和运行时留空间。CPU 线程数按目标机器调节。

GPU memory fractions are relative to the entire GPU card; additional space must be reserved for Nemotron FP32 and runtime. Adjust CPU thread count based on the target machine.

应用参数通过 `.env` 传入容器；仅在宿主机 shell 中 export 不会自动传入。镜像固定 `DEVICE`，会话数默认 GPU 4、CPU 1。需要时可在 `.env` 添加 `R2T2_MAX_SESSIONS`、`R2T2_MAX_MODEL_LEN`（16384）、`R2T2_ENFORCE_EAGER`（0）或 `R2T2_CHUNK_SECONDS`（GPU 0.16、CPU 0.64）。内部鉴权可设置 `R2T2_INTERNAL_TOKEN`。

Application parameters are passed to the container via `.env`; exporting in the host shell alone does not automatically pass them through. The image fixes `DEVICE`; session count defaults to 4 for GPU and 1 for CPU. If needed, add `R2T2_MAX_SESSIONS`, `R2T2_MAX_MODEL_LEN` (16384), `R2T2_ENFORCE_EAGER` (0), or `R2T2_CHUNK_SECONDS` (GPU 0.16, CPU 0.64) to `.env`. Internal authentication can be set with `R2T2_INTERNAL_TOKEN`.

## 模型与运行数据 / Models and Runtime Data

唯一挂载 `./models:/app/models`：`models/huggingface` 存放 R2T2 与强制对齐模型，`models/nemotron-3-diarization` 存放 Nemotron，`models/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` 存放 CT-Transformer。日志用 `docker compose logs`，临时音频与编译缓存留在容器内。

Only mount is `./models:/app/models`: `models/huggingface` stores R2T2 and forced alignment models, `models/nemotron-3-diarization` stores Nemotron. Logs are accessed via `docker compose logs`; temporary audio and compilation caches remain inside the container.

R2T2 revision 固定为 `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`，Nemotron revision 为 `f667ed73aee57d40cc39428eb768b4fd87a0a29e`。Python 依赖由 `uv.lock` 锁定。

CT-Transformer 复用此前的约 292 MB 标点权重，固定 ModelScope `v2.0.4` 并校验三个必需文件的 SHA-256。文件转写默认在 CPU 上恢复段内及句尾标点，不额外占用 GPU 显存。已有匹配缓存可直接复用，缺失时首次启动自动下载；`HF_HUB_OFFLINE=1` 同时禁止该模型的下载。升级此修复需重建镜像或同步更新后的依赖。

CT-Transformer reuses the previous ~292 MB punctuation weights, pinned to ModelScope `v2.0.4` with SHA-256 verification of three required files. File transcription restores intra-segment and end-of-sentence punctuation by default on CPU, without additional GPU memory. Existing matching caches are reused directly; missing models are automatically downloaded on first start. `HF_HUB_OFFLINE=1` also disables downloads for this model. Upgrading this fix requires rebuilding the image or syncing the updated dependencies.

## 原生 CPU / Native CPU

需要 Python 3.11–3.12、uv、Rust、FFmpeg；Linux 还需 libsndfile 和 OpenBLAS 开发库（Debian/Ubuntu：`libsndfile1 libopenblas-dev`）。macOS 支持 Apple Silicon。

Requires Python 3.11–3.12, uv, Rust, FFmpeg; Linux also requires libsndfile and OpenBLAS development libraries (Debian/Ubuntu: `libsndfile1 libopenblas-dev`). macOS supports Apple Silicon.

```bash
uv sync --frozen --extra cpu  # macOS 去掉 --extra cpu / macOS: omit --extra cpu
./scripts/build-rust.sh
HF_HOME="$PWD/models/huggingface" \
DEVICE=cpu OPENBLAS_NUM_THREADS=8 uv run --no-sync python start.py
```

原生服务端口为 8000，首次启动同样自动下载模型到 `models/`。使用自定义 `CARGO_TARGET_DIR` 时，另设置 `R2T2_CPU_LIBRARY_PATH` 指向构建的动态库。原生 Linux GPU 使用 `uv sync --frozen --extra cuda`；macOS 默认 CPU，Linux 默认 CUDA，不自动降级。

Native service port is 8000; first start also automatically downloads models to `models/`. When using a custom `CARGO_TARGET_DIR`, also set `R2T2_CPU_LIBRARY_PATH` to point to the built dynamic library. Native Linux GPU uses `uv sync --frozen --extra cuda`; macOS defaults to CPU, Linux defaults to CUDA, with no automatic fallback.

## 验证与边界 / Verification and Limitations

```bash
docker compose ps
curl http://localhost:17003/health
```

配置鉴权时添加 `Authorization: Bearer <API_KEY>`。文件转写示例见 [README](../README.md)，实时与并发验收见 [benchmark](../scripts/benchmark/README.md)。

When authentication is configured, add `Authorization: Bearer <API_KEY>`. See [README](../README.md) for file transcription examples; see [benchmark](../scripts/benchmark/README.md) for realtime and concurrency acceptance testing.

启动器依次加载私有推理引擎与公共 API，任一进程异常则关闭全部子进程。私有引擎只监听容器内 `127.0.0.1:8001`；健康检查等待模型就绪，启动宽限期为 600 秒。

The launcher loads the private inference engine and public API sequentially; if any process fails, all child processes are shut down. The private engine only listens on `127.0.0.1:8001` inside the container; health check waits for model readiness with a 600-second startup grace period.

CPU 推理不可中途抢占，同时跑实时与离线会增加延迟。历史 M5 Pro 短样本纯识别 RTF 约 0.08–0.13；Linux amd64 完整 5 分钟录音约耗时 310 秒，不能按 Mac 结果承诺实时性能。Linux arm64 已验证构建和动态库加载，完整模型链路尚未验收。

CPU inference cannot be preempted mid-execution; running realtime and offline simultaneously increases latency. Historical M5 Pro short-sample pure recognition RTF is approximately 0.08–0.13; Linux amd64 full 5-minute recording takes approximately 310 seconds—realtime performance cannot be guaranteed based on Mac results. Linux arm64 build and dynamic library loading are verified; full model pipeline not yet accepted.

Nemotron 最多支持 8 个说话人。混合语言可能漏词，重叠发言仍受单路 ASR 限制；时间戳边界合法不代表人工对齐准确。模型质量和长时并发须在目标硬件用真实录音验收。

Nemotron supports up to 8 speakers. Mixed languages may miss words; overlapping speech is still limited by single-channel ASR; valid timestamp boundaries do not guarantee accurate manual alignment. Model quality and long-duration concurrency must be verified with real recordings on target hardware.

## macOS 登录自启 / macOS Login Auto-Start

使用用户级 LaunchAgent，开机登录后启动，退出登录时停止；不需要 `sudo`。
先完成原生 CPU 的依赖安装和 Rust 构建，然后安装服务：

Uses a user-level LaunchAgent that starts after login and stops on logout; no `sudo` required.
First complete native CPU dependency installation and Rust build, then install the service:

```bash
uv sync --frozen
./scripts/build-rust.sh
ALIGNMENT_MODE=uniform HF_HOME="$PWD/models/huggingface" \
  ./scripts/start-native.sh --download-models
uv run --no-sync python scripts/macos-service.py install
```

安装器立即启动服务，默认 CPU、8 个 Rust 线程、1 个实时会话和 `uniform` 对齐。
安装器将 Git 已跟踪文件、编译好的 Rust 库和 `.env` 复制到
`~/Library/Application Support/AsrServe`，在那里创建独立 `.venv`。
模型通过 macOS APFS 克隆复制到运行目录，避免重复占用初始数据块。
这避开后台进程访问 `Documents` 的隐私限制，不需要全磁盘访问权限。
启动不会安装依赖或编译；更新源码或 `.env` 后重新运行安装器。

The installer immediately starts the service, defaulting to CPU, 8 Rust threads, 1 realtime session, and `uniform` alignment.
The installer copies Git-tracked files, the compiled Rust library, and `.env` to
`~/Library/Application Support/AsrServe`, where it creates an independent `.venv`.
Models are copied to the runtime directory via macOS APFS cloning to avoid duplicating initial data blocks.
This avoids privacy restrictions on background processes accessing `Documents`; full disk access is not required.
Startup does not install dependencies or compile; re-run the installer after updating source code or `.env`.

`ALIGNMENT_MODE=forced` 使用 Qwen ForcedAligner；`uniform` 按文本单元均分音频片段时长，
中文按字、英文按词，不下载、校验或加载 ForcedAligner。
均分时间戳不反映实际停顿，也可能影响说话人切换处的字词归属。
Verbose JSON 的 `word_timestamp_method` 和响应头 `X-Word-Timestamp-Method`
分别标记 `forced_alignment` 或 `uniform_fallback`，后者是主动选择的估算模式。
普通启动默认 `forced`，LaunchAgent 默认 `uniform`。

`ALIGNMENT_MODE=forced` uses Qwen ForcedAligner; `uniform` divides audio segment duration equally by text units,
by character for Chinese, by word for English, without downloading, verifying, or loading ForcedAligner.
Equal-division timestamps do not reflect actual pauses and may affect word attribution at speaker transitions.
Verbose JSON's `word_timestamp_method` and response header `X-Word-Timestamp-Method`
indicate `forced_alignment` or `uniform_fallback` respectively; the latter is an intentionally selected estimation mode.
Normal startup defaults to `forced`; LaunchAgent defaults to `uniform`.

重新安装可以更改对齐模式和线程数：

Reinstalling can change alignment mode and thread count:

```bash
uv run --no-sync python scripts/macos-service.py install --alignment-mode forced --threads 8
```

LaunchAgent 明确设置的参数优先于仓库 `.env`；其余配置由复制后的 `.env` 提供。
以下路径均相对于运行目录：模型存放于 `models/`，后台输出在 `logs/launchd.stdout.log`、`logs/launchd.stderr.log`，
应用日志使用现有轮转设置。长时间运行时定期清理 launchd 的输出日志。
本机脚本固定公共端口为 `0.0.0.0:17003`，内部推理端口仍为 `8001`。局域网访问可在 `.env` 配置 `API_KEY`。

Parameters explicitly set by LaunchAgent take precedence over the repository `.env`; remaining configuration is provided by the copied `.env`.
The following paths are all relative to the runtime directory: models are stored in `models/`, background output is in `logs/launchd.stdout.log` and `logs/launchd.stderr.log`,
and application logs use the existing rotation settings. Periodically clean up launchd output logs during long-running operation.
The local script fixes the public port to `0.0.0.0:17003`; the internal inference port remains `8001`. For LAN access, configure `API_KEY` in `.env`.

```bash
./scripts/start-native.sh --healthcheck
launchctl print "gui/$(id -u)/com.asrserve.native"
launchctl kickstart -k "gui/$(id -u)/com.asrserve.native"
uv run --no-sync python scripts/macos-service.py uninstall
```

卸载只停止自启并删除 plist，保留模型、配置和日志。休眠期间不提供推理服务。

Uninstall only stops auto-start and deletes the plist; models, configuration, and logs are preserved. Inference service is not provided during sleep.
