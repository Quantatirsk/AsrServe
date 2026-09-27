# Ascend 910B：当前 dev 与旧适配分支的迁移差异

> 更新：2026-09-27。功能基线：`dev@2cfb68f`；旧适配基线：`codex/ascend-910b-adaptation@5251d43`。
> 本文更新调研与验收要求，不表示已完成 NPU 代码适配或实机验证。整体方案见 [迁移可行性研究](./ascend-910b-feasibility.md)。

## 1. 当前判断

可以开始以当前 `dev` 为基线迁移，选择性接回旧分支的 Ascend 镜像、设备挂载与启动经验；无需先重做业务架构。**R2T2 在 910B 上的模型加载、完整接口、实时调度和对齐器仍需验证。** 旧分支已有部署配置，不代表当前 R2T2 + Nemotron 已可交付。

官方已有 Qwen3-ASR-1.7B 的 Atlas A2 单卡教程。R2T2 固定权重配置使用 `Qwen3ASRForConditionalGeneration` / `qwen3_asr`，因此有复用该后端的架构依据；这是迁移假设，**不能写成 R2T2 权重与项目流式链路已获得官方支持**。[官方教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html)、[R2T2 固定配置](https://huggingface.co/netease-youdao/Confucius4-R2T2/blob/185ce639118ad1362d049ca0d8ed04b6ec5cd6c9/config.json)

## 2. 需要同步的功能差异

| 项目 | 旧 910B 分支 | 当前 dev 与迁移要求 |
|---|---|---|
| ASR | 旧 Qwen3-ASR 技术栈 | 实时与离线共用 R2T2，保留已有私有推理进程接口 |
| 离线语音检测 | FSMN VAD | 每次离线请求只运行一次 Nemotron，以所有说话人活动区间的并集分段 |
| 说话人 | CAM++ 路径 | Nemotron；首期可用 CPU，但实际宿主性能待测 |
| 关闭说话人分离 | 旧行为不可直接沿用 | 仍运行 Nemotron 供分段；不归属文本、不输出说话人；仅请求词级时间戳时运行 Aligner |
| 开启说话人分离 | 旧归属逻辑 | 复用同一次 Nemotron 结果；当前实现需要内部词级对齐，即使不对外输出词时间戳 |
| 空活动结果 | 旧 VAD 兜底 | 保留整段音频，按最大时长固定切分并交给 ASR；不能当成“确定静音”直接丢弃 |
| 模型推理失败 | 旧逻辑不可直接沿用 | Nemotron 失败明确报错，不能伪装为空活动并继续 |
| 原始文本 | 曾有标点/规整后处理 | 标点补充已删除，不恢复 FunASR、CT-Transformer 或原文改写 |
| 实时收尾 | 独立实时链路 | 低能量停顿检测保留；它不是离线语音检测的替代品 |
| 长音频与段落 | 旧切分/合并 | 复用最大 60 秒分段、短插话归并和有依据的短 Unknown 边界合并规则 |

代码依据：[离线准备与结果组装](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/services/asr/long_audio.py)、[活动区间与 Nemotron](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/utils/speaker_diarizer.py)、[分段器](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/utils/audio_splitter.py)、[ASR 引擎](https://github.com/Quantatirsk/qwen3-asr/tree/2cfb68f/app/services/asr/engines)、[说话人归属](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/services/asr/speaker_attribution.py)。

FSMN 的加载、预加载、下载/完整性检查和配置已经删除，FunASR、ModelScope 及其仅为旧链路保留的直接依赖也已删除。**迁移清单不再包含这些模型。** 不要把旧分支的依赖文件整体覆盖到当前基线。[当前依赖](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/pyproject.toml)

## 3. 真正剩余的接线工作

### 3.1 进程与设备

当前启动器虽然运行两个进程，但二者都用 `sys.executable`，是**同一个 Python 环境**；私有进程只承载 R2T2，Nemotron 与 Aligner 位于公共 API 侧。旧分支的双环境启动经验可参考，不能认为当前已自动隔离依赖。[当前启动器](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/deploy/entrypoint.py)、[旧 Ascend 启动脚本](https://github.com/Quantatirsk/qwen3-asr/blob/5251d43/scripts/start-ascend-services.sh)

最小方案是在同一容器内保留现有两个进程，让 R2T2 使用选定的 Ascend 运行环境，API 侧的 Nemotron 使用 CPU 环境。需实际接入两个解释器、明确每个模型的设备选择；不能用全局 `DEVICE=cpu` 掩盖主模型未上 NPU。也不能原样沿用启动器中的 `VLLM_PLUGINS=""`：应按选定 vLLM Ascend 版本验证平台插件发现与加载。

### 3.2 依赖兼容

当前 dev 固定 Transformers 提交 `27166ea03f12c940f23176a904ab1d2ff1a3dcbb` 以运行 Nemotron；所查 vLLM Ascend `v0.27.1rc1` 要求 `transformers==5.14.1`。**这些基线的依赖约束不同，需要验证或隔离**；不能推导为所有版本永远无法共享环境，也不能未经回归就强行升级 Ascend 环境的单个包。[dev 依赖](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/pyproject.toml)、[Ascend 固定版本依赖](https://github.com/vllm-project/vllm-ascend/blob/v0.27.1rc1/requirements.txt)

当前 dev 已声明 `linux aarch64`，旧报告“锁文件完全不覆盖该平台”的结论已过时。仍需在目标 CPU 架构上核实 CPU wheel、音频库、Ascend 整套版本与安装结果；目标机器不能先验认定为鲲鹏。Ascend 的 vLLM、PyTorch、torch_npu、CANN 和驱动组合按选定发行版整体固定，不把当前 CUDA 锁文件直接用于 NPU。

### 3.3 对齐器

Pooling 或 `token_classify` 出现在通用支持/测试清单，不能证明 `Qwen3ASRForcedAlignerForTokenClassification` 的加载、音频输入或时间戳正确。需要单独验证当前对齐器及其实际后端。

首期可先跑通 `enable_speaker_diarization=False`、`word_timestamps=False` 的最小识别请求，但当前初始化仍会创建 Aligner，需要同步调整初始化和模型检查，不能仅关闭请求参数；这也不代表已交付说话人分离。如果实测无法使用对齐器，可评估旧分支已有的均分时间戳路径，**必须显式标明估算模式**，并验证时间戳误差、词语跨说话人边界的归属与用户可见结果；不能静默降级或宣称等同强制对齐。暂不为对齐器另建服务。

## 4. CPU 依据与边界

已有真实 Nemotron CPU 基准：MachineLearning，i5-13600KF、约 64 GiB RAM，51.66275 秒双人音频，float32、4 个 PyTorch intra-op 线程、暖机后三次中位数：

| 指标 | 结果 |
|---|---:|
| 离线处理 | 0.526 秒 |
| 流式累计计算 | 13.003 秒 |
| 测试进程峰值 RSS | 约 1.63 GiB |

记录位于 **dev 工作区**的 `docs/research/nemotron-cpu.md` 与 `nemotron-cpu-machinelearning.json`。这些不是 910B 宿主测试，也不代表长会话或多路容量。流式累计计算不包含音频实际到达等待、网络与 ASR；不能由约四倍实时速度承诺四路并发。

这些数据支持“先让 Nemotron 留在 CPU”，但没有证明其语音边界与 FSMN 准确率等价。无活动可能来自噪声、音乐、低声或短促语音，保留整段音频的空区间兜底必须继续存在。目标宿主需复测实际音频与并发，不能从离线耗时推算流式延迟。

## 5. 最小迁移顺序与验收

1. **锁定目标环境**：记录实际 CPU 架构、NPU 型号/显存、驱动/固件、CANN 与容器运行权限；固定镜像 digest、推理依赖和权重 revision。保留单容器交付约束。
2. **R2T2 最小 NPU 推理**：先验证官方同架构路径，再加载固定 R2T2 权重；检查实际 NPU 执行、原始文本与音频输入。继续验证项目所需上下文长度，不能只拿教程的 4096 配置代表当前默认 16384 可用。
3. **接回离线链路**：NPU R2T2 + CPU Nemotron；先关标签与词时间戳，再验证对齐器并打开对应能力。
4. **实时与混合负载**：验证现有滚动窗口、取消/断连、请求优先级与资源回收。解码间隔从保守值实测调节，不能把 CUDA 的延迟或调度效果直接照搬。

交付前至少记录以下结果：

- [ ] 标签开/关 × 词时间戳开/关四种组合；关闭两者不调用 Aligner，关闭标签不输出说话人信息（字段按现有协议为空或省略）。
- [ ] 一次离线请求只推理一次 Nemotron，切分与归属复用结果。
- [ ] 超过 60 秒纯静音、语音夹长静音、噪声、音乐、低声、短促语音；空活动保留原音频兜底，不能悄悄丢失片段。
- [ ] 重叠说话活动取并集、长录音每块不超过上限、时间轴不倒退、临时文件在成功/失败/取消后回收。
- [ ] Nemotron 加载或推理失败明确返回错误，不能作为空活动成功处理。
- [ ] 同一固定 ASR 输出经过段落归并后，原文与绝对时间戳不被改写；与现有 GPU 结果比较时允许独立推理差异并单独评估质量。
- [ ] 对齐器真实准确性；如启用估算模式，结果明确区分且用户可见，单独评估归属误差。
- [ ] 目标 CPU 的 Nemotron 离线/流式耗时、长会话 RSS、实际并发下 p95/p99 与积压。
- [ ] NPU 主模型加载、长输入、实时收尾、取消和断连；实时+离线混合负载下调度与显存回收。
- [ ] 容器重启、健康检查、API/私有进程互相退出处理，以及对外端口 17003。

**本次只更新文档。** 上述未完成项不能通过把旧测试结果换成新模型名称来勾选。
