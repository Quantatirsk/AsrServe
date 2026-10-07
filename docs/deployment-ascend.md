# Ascend 910B 部署

## 运行布局

一个容器、两个 Python 环境、两个进程：

| 进程 | 解释器 | 职责 |
| --- | --- | --- |
| 私有 `127.0.0.1:8001` | `/opt/ascend-python` | R2T2 的 Ascend AsyncLLM，实时与离线共享权重 |
| 公共 `:8000`，映射 `17003` | `/opt/api-venv/bin/python` | API、CPU Nemotron 与 CT-Transformer、分段、估算时间戳与说话人归属 |

启动器先用 NPU 解释器检查设备，私有引擎就绪后再启动 API；任一进程退出会清理整个进程组。

基础镜像固定为 vLLM Ascend `v0.27.1rc1`（digest 见 `Dockerfile.ascend`），自带 CANN 9.1.0、vLLM 0.27.1、torch 2.10.0、torch_npu 2.10.0.post4。CPU venv 使用本仓库 `uv.lock`，不覆盖 NPU 环境。

## 构建与启动

在 910B 宿主机（需 Ascend 驱动）执行：

```bash
docker compose -f compose.ascend.yml up -d
docker compose -f compose.ascend.yml logs -f asr
```

Compose 只引用 `quantatrisk/asrserve:ascend`（Docker Hub 发布 linux/arm64，版本标签 `1.0.4-ascend`；x86 宿主机或修改代码后用 `TARGET=ascend ./build.sh` 本地构建），挂载 `./models` 与宿主驱动（`/usr/local/Ascend/driver`、`/usr/local/dcmi`、`npu-smi`），直通 `/dev/davinci0` 及管理设备。首次启动自动下载 R2T2、Nemotron 与 CT-Transformer 标点模型到 `./models`，不下载 Aligner。

文件转写支持 `prompt`、`hotwords` 并恢复整份文件的句读；CPU API 环境包含固定版本的 FunASR、ModelScope 和 libsoxr，NPU 推理环境不安装这些标点组件。OpenAI `/v1/realtime` 通过私有流式协议使用已有 NPU R2T2，提供与主分支相同的手动提交、多轮和热词提示；24→16 kHz 连续重采样在 API 进程完成。SDK 配置和事件见 [OpenAI Realtime 转写](openai-realtime.md)。

离线部署时，在有网络的机器预下载后拷贝 `models/`，并在 `.env` 设置 `HF_HUB_OFFLINE=1`：

```bash
IMAGE=quantatrisk/asrserve:ascend ./scripts/prepare-models.sh
```

预下载不访问 NPU；没有 Ascend 镜像的机器可用 `:gpu` 或 `:cpu` 镜像，会额外下载一个不使用的 Aligner。

## 默认参数

NPU 默认值已写在代码与镜像中，无需配置：BF16、eager、16384 上下文、0.85 NPU 内存预算、0.64 秒实时解码间隔。需要调整时在 `.env` 设置 `R2T2_GPU_MEMORY_UTILIZATION`、`R2T2_MAX_MODEL_LEN`（须大于 4096）或 `R2T2_CHUNK_SECONDS`。这些默认值尚未经 910B 容量实测。

## 时间戳与说话人

910B 固定 `ALIGNMENT_MODE=uniform`，设置 `forced` 会启动报错。均分在每个识别段内进行，没有声学依据，会影响跨说话人边界的归属精度。

- 每次离线请求只运行一次 Nemotron；活动并集用于分段，原始重叠活动用于归属，`speaker_segments` 保留原始活动。
- 开启说话人标签时内部仍会估算词区间，即使不公开词时间戳；空活动整段分块识别，模型失败直接报错。
- OpenAI verbose JSON 返回 `word_timestamp_method="uniform_fallback"`；所有格式通过 `X-Word-Timestamp-Method` 响应头标明方法。

## 验证

```bash
uv sync --frozen --extra cpu
uv run --frozen python -m pytest -q tests/test_ascend_smoke.py tests/test_single_container.py
curl http://localhost:17003/health
```

`test_ascend_smoke.py` 用模拟设备与引擎检查双环境、NPU 不可用报错、BF16/eager 配置、标签与词时间戳组合、跳过 Aligner 和估算标识。真实 NPU 推理、目标宿主性能与长时并发仍需实机验收。
