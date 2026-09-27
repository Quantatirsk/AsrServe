# Nemotron-3-Diarization 的 CPU 可用性与效率

> 研究日期：2026-09-27。上游代码检查使用本仓库固定的 Transformers commit `27166ea03f12c940f23176a904ab1d2ff1a3dcbb`。本节公开资料与后续本机实测分开记录。

## CPU 可用性

**Transformers 实现具备 CPU 执行路径，未发现必须依赖 CUDA 的模型计算；是否满足实时并发要求需要实测。** NVIDIA 模型卡的硬件列表和速度数据面向 NVIDIA GPU，不能据此宣称厂商已经给出 CPU 性能保证。

证据：

- 模型 attention 存在普通 PyTorch `matmul`、`softmax` 的 eager 实现，也声明支持 SDPA；缓存与位置张量随输入的 `device` 创建。FlashAttention 是可选后端，不是运行前提。[固定版本模型源码](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py)
- 上游离线/流式测试使用 `torch_device`，并非 GPU-only 测试；测试工具允许显式 `TRANSFORMERS_TEST_DEVICE=cpu`，没有可用加速设备时选择 CPU。这证明测试代码设计允许 CPU，不代表本次已运行上游全部测试。[模型测试](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/tests/models/nemotron3_diarization/test_modeling_nemotron3_diarization.py)、[设备选择](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/testing_utils.py)
- 当前项目已用 `.to(device)` 和 float32 加载 Nemotron，`DEVICE=cpu` 可走这条路径；但 NPU 主模型与 CPU Nemotron 的独立设备配置尚未接线。[当前加载代码](../../app/utils/speaker_diarizer.py)

CPU 验证应显式使用 `.to("cpu")` 和 float32，记录实际 attention 后端；`device_map="auto"` 可能选择 GPU，不能拿它当 CPU 测试证据。先测试未编译版本，无需为了确认能否运行引入 NeMo、ONNX 或另一套模型实现。

## 官方速度数字不能当 CPU 基准

NVIDIA 官方模型卡报告的单样本速度如下；**硬件是 Blackwell RTX PRO 5000，框架是 NeMo，精度是 BF16**：

| 模式的输入缓冲时长 | RTFx，eager | RTFx，compiled |
|---|---:|---:|
| 离线式 30.4 秒 | 1340 | 4385 |
| 流式 1.04 秒 | 38 | 164 |
| 流式 0.64 秒 | 25 | 113 |
| 流式 0.32 秒 | 12.5 | 54 |

来源：[NVIDIA 模型卡 — Inference Speed Evaluation](https://huggingface.co/nvidia/Nemotron-3-Diarization#inference-speed-evaluation)。本次查阅未找到官方 CPU RTF、吞吐或并发容量表；上表也不是本仓库 Transformers float32 的运行速度。

Transformers 文档另给出 A100 上编译后相对 eager 的提升：流式每步 float32 为 1.2 倍、BF16 为 4.4 倍；离线 488 秒录音分别为 1.3 倍和 2.8 倍。该优化示例采用固定形状与 CUDA graph，不能套用为 CPU 提速承诺。文档还指出，缓存逐步变化会让直接 `torch.compile` 重复编译，因此编译不是默认的 CPU 验证步骤。[固定版本 Transformers 优化说明](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/docs/source/en/model_doc/nemotron3_diarization.md)

## 衡量效率时需要区分的数字

- `RTF = 处理耗时 / 音频时长`，越小越好；`RTFx` 是它的倒数，越大越好。单路平均 RTF 小于 1 只是跟得上音频的必要容量条件，仍需检查逐步 p95/p99 与积压。
- 1.04 / 0.64 / 0.32 秒是**输入块加前瞻的缓冲等待**，不含计算和队列耗时。把缓冲改短会提高调用频率，不能保证 CPU 上端到端延迟更小。[Transformers 流式说明](https://huggingface.co/docs/transformers/main/model_doc/nemotron3_diarization)
- 离线大块吞吐与流式小块速度要分别测；缓存逐渐填满后每次计算的上下文更多，几秒短音频不能代表长会话稳态。记录冷启动、模型加载、暖机后时间，以及实际语音输入的预处理和后处理。
- CPU 线程数、CPU 型号、PyTorch/BLAS、并发会话、精度、attention 后端均影响结果。Mac 的 CPU 实测只能证明该机器的能力，不能替代 910B 服务器所配鲲鹏或 x86 CPU 的验收。

对 Ascend 部署的当前判断应是：**CPU 辅助链有可执行依据，效率尚须目标机器验证。** 先用现有实现获得离线与流式基线，再决定是否有必要调线程、编译或迁移 Nemotron 本身；不从 GPU 数字反推需要多少 CPU 核。

## 本机 CPU 实测

2026-09-27，在 Apple M5 Pro（18 核、64 GiB RAM）上，以 PyTorch 2.10.0、float32、显式 `device=cpu` 运行仓库实际代码；未使用 MPS、CUDA、量化或 `torch.compile`。使用仓库固定 revision 的本地模型。输入为缓存中的 ModelScope 双人语音 `2speakers_example.wav`，16 kHz 单声道，51.66275 秒；文件 SHA-256 见原始结果。

每种线程数先分别暖机离线和流式路径，再测量三次。离线包含 `librosa.load`、特征提取、模型和分段后处理；流式调用当前 `SpeakerStream`，按 160 ms 包连续回放但不 sleep，包含特征提取、模型、缓存和活动区间处理，不包含网络、ASR、音频实际到达等待或多路争用。下面耗时均为三次中位数，线程数指 PyTorch intra-op 配置，不能等同于进程仅使用这么多操作系统线程。

| PyTorch 线程数 | 离线耗时 | 离线 RTF | 流式累计计算耗时 | 流式 RTF | 流式有效分块计算 p95 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.370 s | 0.0072 | 6.877 s | 0.133 | 156.8 ms |
| 4 | 0.249 s | 0.0048 | 5.349 s | 0.104 | 113.5 ms |
| 6 | 0.246 s | 0.0048 | 5.398 s | 0.104 | 112.3 ms |

分块 p95 是每次回放中有新增输出帧的 `advance` 调用耗时的 95 分位，再取三次中位数；不包含末尾 flush，累计时间则包含。每次回放有 71 个这样的调用，输出 5165 帧。离线三种线程配置均返回两个说话人、非空区间与有限概率；这证明真实模型成功运行，**没有人工标注 DER，不能据此断言分离准确率**。

首次 `warmup`（包含首次 Transformers 类导入和模型加载，不包含此前 Python/torch 启动）为 1.88 秒。整个测试进程峰值 RSS 约 1.04 GiB，包含 Python、依赖、权重、缓存和两条执行路径，不是模型权重大小或每路增量内存。

**本机结论：CPU 可用，单路效率有余量。** 4 线程流式计算约为音频时长的 10.4%（约 9.7 倍实时速度）；离线约 208 倍实时速度。离线批量处理与流式反复小块处理的效率差异很大，不能用离线数字推算流式容量。6 线程相比 4 线程没有明显收益。

这只是单段约 52 秒音频的单路结果，不是长会话或并发压测，不能用 `1/RTF` 直接承诺十路并发。CPU 在该样本上跟得上实时，并不消除默认约 1.04 秒输入缓冲的等待。迁移到 910B 后，还需在实际宿主 CPU 上同时运行 API、Nemotron 和 R2T2 请求链验证。

复测命令（换成目标机器上相同或代表性音频）：

```bash
uv run --no-sync python -m scripts.benchmark.nemotron_cpu /path/to/16k-mono.wav \
  --threads 1 4 6 --repeats 3 --output benchmark_results/nemotron-cpu.json
```

[可复用脚本](../../scripts/benchmark/nemotron_cpu.py)；[本次逐轮原始结果](nemotron-cpu-m5pro.json)。脚本通过断言检查模型加载、输出概率和流式错误，不会把流式自动降级为无标签误报成成功。

## MachineLearning 远程 CPU 实测

2026-09-27，MachineLearning（主机名 vectorlab），Intel Core i5-13600KF，14 核 / 20 逻辑线程，62.56 GiB 可见内存，Linux 6.8.0-139。使用现有服务镜像的独立临时容器；未更改或重启现有服务。测试容器没有 GPU 设备请求、没有 `/dev/nvidia*`，设置 `DEVICE=cpu`、空 `CUDA_VISIBLE_DEVICES`、`NVIDIA_VISIBLE_DEVICES=void`，禁用网络，模型目录只读挂载。没有 CPU 配额或绑核，允许调度到 CPU 0–19。

模型权重和 `speaker_diarizer.py`、`speakers.py` 的 SHA-256 均与本机一致；使用同一脚本、同一 51.66275 秒双人音频、相同的暖机和三次重复方法。远程 PyTorch 为 **2.13.0+cu130**，虽然安装包包含 CUDA，但实际张量设备为 **cpu**、float32；本机 PyTorch 为 2.10.0，所以差异不能仅归因于 CPU 硬件。

| PyTorch 线程数 | 离线耗时中位数 | 流式累计计算中位数 | 流式处理速度 / 播放速度 | 有效分块计算 p95 |
|---|---:|---:|---:|---:|
| 1 | 1.76 s | 35.18 s | 1.47 倍 | 798 ms |
| 4 | 0.53 s | 13.00 s | 3.97 倍 | 311 ms |
| 6 | 0.92 s | 18.97 s | 2.72 倍 | 430 ms |

**本次 4 线程最好：约 52 秒录音，离线约半秒，流式全部计算约 13 秒。** 这段音频的单路实时处理有余量，6 线程反而较慢；原因未单独诊断，不能武断归因为大小核或线程竞争。现有其他服务持续运行，4 线程离线三轮为 0.526 / 1.249 / 0.515 秒，存在抖动，报告保留全部原始结果。单线程虽平均跟得上播放，最慢分块约 0.87 秒，实时余量较小。

首次模型 warmup 5.88 秒；测试进程峰值 RSS 约 **1.63 GiB**（完整进程，非每路内存）。所有离线轮次均返回两个说话人、有效概率；流式均正常输出 5165 帧。没有标注数据，未计算 DER。缓冲等待、多路并发和长会话的限制仍同上一节，4 倍处理速度不能直接写成支持 4 路并发。

4 线程与 M5 Pro 对比：离线慢约 2.1 倍，流式慢约 2.4 倍。现阶段可以保留 CPU 方案，但 910B 实际宿主 CPU 仍需单独测试。

[远程逐轮原始结果](nemotron-cpu-machinelearning.json)。远程留存目录：`/tmp/nemotron-cpu-test.Nvr0TE`（脚本、音频、日志与结果）；测试容器执行结束后自动删除。镜像 ID：`sha256:4585ebe08175cdec1d82b76c7e1644d54c9195a300c1da03f81790d486c84543`。
