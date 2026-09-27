# R2T2 + Nemotron：Ascend 910B

本分支以 `dev@3af1c97` 为功能基线，删除旧 Qwen3-ASR、FSMN、CAM++、远程 OpenAI vLLM 适配及其启动参数，不提供兼容入口。

## 运行布局

一个容器，两个独立 Python 环境、两个进程：

| 进程 | 解释器 | 职责 |
| --- | --- | --- |
| 私有 `127.0.0.1:8001` | `R2T2_PYTHON=/opt/ascend-python` | R2T2 的 Ascend AsyncLLM，实时与离线共享权重 |
| 公共 `:8000`，映射 `17003` | `API_PYTHON=/opt/api-venv/bin/python` | API、CPU Nemotron、分段、估算时间戳与说话人归属 |

两个进程的 `DEVICE` 都声明主模型的 `npu:0`，仅私有进程校验并使用 NPU。Nemotron 单独使用 `SPEAKER_DIARIZATION_DEVICE=cpu`。启动器通过 NPU 解释器检查设备，等待私有引擎就绪再启动 API；任一进程退出会清理整个进程组。

镜像固定为当前最新带版本号的 **vLLM Ascend v0.27.1rc1**（2026-09-27 查询，RC 候选版本），不跟随每日变化的 nightly 标签。digest：

`sha256:4f75925e8f38e6d964768ae340c2433d08003e96fb37ceebc8cab10ec1a9ee06`

NPU 环境保留镜像整套依赖：CANN 9.1.0、vLLM 0.27.1、torch 2.10.0、torch_npu 2.10.0.post4、Triton Ascend 3.2.2、Transformers 5.14.1。CPU venv 使用 dev 的锁文件和 Transformers 提交 `27166ea03f12c940f23176a904ab1d2ff1a3dcbb`，不覆盖 NPU 环境。启动时显式启用 Ascend 平台和模型注册插件。

依据：[镜像仓库](https://quay.io/repository/ascend/vllm-ascend?tab=tags)、[版本构建配置](https://github.com/vllm-project/vllm-ascend/blob/v0.27.1rc1/Dockerfile)、[发行版依赖](https://github.com/vllm-project/vllm-ascend/blob/v0.27.1rc1/requirements.txt)、[官方同架构 ASR 教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html)。教程不是本项目 R2T2 的实机验收报告。

## 构建与启动

平台方提供驱动、设备节点和模型缓存挂载。维护者仅有容器 shell 时，不需要安装宿主驱动或运行第二个容器。

```bash
# 有 Docker 权限的构建/部署机器；显式指定文件，避免选中 dev 的 compose.yml。
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d
```

在有网络的准备机器下载固定版本权重：

```bash
DEVICE=npu:0 ALIGNMENT_MODE=uniform bash scripts/prepare-models.sh
```

准备脚本在 Linux 使用 CPU extra，下载 R2T2 与 Nemotron，不下载 Aligner。Compose 挂载 `models/huggingface/hub` 和 `models/nemotron-3-diarization`。离线运行时向容器传入 `HF_HUB_OFFLINE=1`，缺失或不完整模型会导致启动失败。

仅有容器 shell 时，在本镜像及平台挂载已就绪的前提下：

```bash
cd /app
/opt/api-venv/bin/python start.py
# 另一个 shell 检查两个进程
/opt/api-venv/bin/python start.py --healthcheck
```

默认采用 BF16、eager、16384 上下文、0.85 NPU 内存预算和 0.64 秒实时解码间隔。可调整 `R2T2_MAX_MODEL_LEN`、`R2T2_GPU_MEMORY_UTILIZATION`（vLLM 通用参数名）、`R2T2_CHUNK_SECONDS`。`R2T2_MAX_MODEL_LEN` 必须大于 4096；这些默认值未经 910B 容量实测。

## 时间戳与说话人语义

910B 当前使用 `ALIGNMENT_MODE=uniform`；设置 `forced` 会明确报错。均分发生在每个识别段内，输出段内相对秒数，后续仅偏移、缩放一次。API 原文不做标点补充或改写。

- 每次离线请求只运行一次 Nemotron；活动并集用于分段，原始重叠活动用于归属。
- 关闭说话人标签仍运行 Nemotron；关闭标签及词时间戳时不生成词区间。
- 开启标签需要内部估算词区间，即使不公开词时间戳；空活动保留整段分块识别，模型失败直接报错。
- OpenAI verbose JSON 的 `word_timestamp_method="uniform_fallback"` 明示估算。OpenAI 各格式通过 `X-Word-Timestamp-Method` 响应头公开方法；纯文本/SRT/VTT 正文不增添非协议内容。
- 均分没有声学证据，会影响跨说话人边界归属；不是强制对齐精度。`speaker_segments` 保留 Nemotron 原始活动。

## 本地烟测范围

```bash
uv sync --frozen
uv run --frozen python -m pytest -q tests/test_ascend_smoke.py tests/test_single_container.py tests/test_r2t2_offline.py tests/test_diarized_pipeline.py tests/test_segment_bounds.py tests/test_speaker_diarizer.py tests/test_speaker_attribution.py tests/test_shared_inference.py tests/test_realtime.py tests/test_api_contract.py
```

`test_ascend_smoke.py` 用模拟设备/推理引擎检查双环境、NPU 不可用报错、BF16/eager/优先级配置、四种标签/词时间戳组合、不加载 Aligner、非零段偏移及时间缩放、估算标识。其余复用 dev 的分段、空活动/失败、临时文件清理、实时取消和进程生命周期测试。

本地结果：98 项测试及 53 个子测试通过，Compose 配置、Python 编译、shell 语法及 diff 空白检查通过。

升级 v0.27.1rc1 后复测：27 项测试及 7 个子测试通过；`docker build --check -f Dockerfile.ascend .` 已解析新镜像元数据并通过，无警告。Compose 配置与 diff 空白检查通过。

本次仅做本地代码与配置烟测；未构建或运行 Ascend 镜像，未执行真实 NPU 推理、对齐精度、目标宿主性能或生产验证。
