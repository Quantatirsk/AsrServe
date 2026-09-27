# R2T2 服务迁移 Ascend 910B：前置条件与实施路径

> 更新：2026-09-27。功能基线：`dev` / `2cfb68f`；旧 Ascend 实现：`codex/ascend-910b-adaptation` / `5251d43`。
> 本文描述待实施的迁移方案。当前没有 R2T2 服务在 910B 上的实机验证结果；CUDA/CPU 测试不能替代 NPU 验收。

## 1. 当前结论

**已具备开始代码适配和最小实机验证的基础，尚不具备宣布 910B 生产可用的依据。** 以当前 `dev` 的 R2T2 + Nemotron 链路为功能基线，在 910B 分支复用必要的 Ascend 部署适配；不恢复旧识别模型、FSMN、CAM++ 或标点补充模块，不先重写一套模型服务。

沿用旧分支记录的交付约束：**一个容器镜像，维护者仅有容器 shell**。NPU 设备、驱动目录和持久化模型缓存由平台侧提供；不能把宿主驱动安装、修改容器设备映射或运行第二个容器当作维护者可自行完成的步骤。

| 前置条件 | 状态 | 边界 |
|---|---|---|
| 统一识别与离线分段链路 | 已实现并在 GPU 验证 | 实时/离线共用 R2T2；离线统一使用 Nemotron 活动区间 |
| Nemotron CPU 执行 | 已实测 | Mac 与 MachineLearning 单路测试通过，目标 910B 宿主 CPU 容量待测 |
| Ascend 部署骨架 | 旧分支已有 | 单容器、双 Python 环境和设备挂载可参考；仍是旧 Qwen3-ASR 链路 |
| R2T2 架构兼容依据 | 有，需实机验证 | 相同架构的官方教程不等于 R2T2 权重与本服务已获验证 |
| 目标硬件与软件组合 | 待确认 | Atlas 型号、CPU 架构、卡数/HBM、驱动、firmware、CANN、镜像 digest |
| NPU 运行接线、对齐与容量 | 待实施/验证 | 当前 `dev` 只接受 CPU/CUDA，不能直接设置 `DEVICE=npu:0` 启动 |

## 2. 当前功能基线：迁移时必须保留

离线流程是：音频解码 → **一次 Nemotron 推理** → 将所有说话人的活动区间取并集 → 按语音边界及最大 60 秒切段 → R2T2 识别 → 按需对齐和文字说话人归属。

| 开关/情况 | 当前行为 |
|---|---|
| `enable_speaker_diarization=false`、`word_timestamps=false` | 仍运行 Nemotron 供分段；不运行对齐推理，不输出说话人信息 |
| `enable_speaker_diarization=false`、`word_timestamps=true` | 同样分段，运行字词对齐；不输出说话人信息 |
| `enable_speaker_diarization=true` | 使用内部字词区间做文字归属；`word_timestamps` 只控制是否公开词时间戳 |
| Nemotron 成功但活动区间为空 | 保留完整音频，按最大长度切分交给 ASR；不把漏检直接判成纯静音 |
| Nemotron 离线推理失败 | 报错并清理临时文件；不静默回退成“无活动” |
| 重叠说话 | 分段使用活动并集，避免重复识别音频；原始说话人活动保留在 `speaker_segments` |

独立 FSMN VAD 的加载、预加载、下载、完整性检查、配置及调用已删除，FunASR/ModelScope 运行依赖已移除。句末标点补充已删除，保留 ASR 原文。**实时低能量停顿检测仍用于语句收尾，与离线分段职责不同，必须保留。**

关闭对齐推理不等于当前启动时不加载 Aligner：`R2T2Engine.__init__` 仍创建对齐器。迁移若选择均分方案，必须同时调整初始化和模型完整性检查，不能只改请求开关。

代码依据：[离线准备与结果组装](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/services/asr/long_audio.py)、[分段器](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/utils/audio_splitter.py)、[对齐调用条件](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/services/asr/r2t2_engine.py)。

## 3. 已有证据及其限制

### R2T2 与 Ascend

项目固定权重 revision 为 `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`，配置声明 `Qwen3ASRForConditionalGeneration` / `qwen3_asr`。这提供了复用 Qwen3-ASR 推理实现的依据。[模型配置](https://huggingface.co/netease-youdao/Confucius4-R2T2/blob/185ce639118ad1362d049ca0d8ed04b6ec5cd6c9/config.json)

vLLM Ascend 官方 Qwen3-ASR-1.7B 教程给出 Atlas 800I A2 64 GB 的单卡 BF16 部署示例。**推断：可以优先尝试加载同架构 R2T2；尚不能认定其权重、热词、增量解码和本项目协议兼容。** 教程的硬件容量不是本服务包含辅助模型后的最低配置承诺。[官方 ASR 教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html)

### Nemotron CPU

同一段 51.66275 秒双人音频，float32、4 个 PyTorch intra-op 线程，暖机后三次中位数：

| 机器 | 离线耗时 | 流式累计计算 | 进程峰值 RSS |
|---|---:|---:|---:|
| Apple M5 Pro，64 GiB RAM | 0.249 秒 | 5.349 秒 | 约 1.04 GiB |
| MachineLearning，i5-13600KF，约 64 GB RAM | 0.526 秒 | 13.003 秒 | 约 1.63 GiB |

依据为 `dev` 工作区的 `docs/research/nemotron-cpu.md`、配套原始 JSON 与 `scripts/benchmark/nemotron_cpu.py`。流式数字是计算累计时间，不包含音频到达、网络、排队或多路争用；RSS 是完整测试进程峰值，不是每路内存。两机 PyTorch 版本也不同，不能只按 CPU 型号归因。

这证明 CPU 路径真实可用，支持首期让 Nemotron 留在 CPU；**不证明与 FSMN 的语音边界准确率等价，也不能外推 910B 宿主的并发容量。** 目标 CPU 仍须复测噪声、音乐、低声、短促语音、重叠和长会话。

### Aligner

当前 GPU Aligner 使用 vLLM pooling / `token_classify`，是独立于 ASR `generate` 的路径。上游有对应模型实现，但本项目没有目标 910B 的加载、时间戳正确性或性能结果；通用 pooling 或 CI 列表不能代替该模型验收。[上游 Aligner 实现](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen3_asr_forced_aligner.py)

可先独立验证真实 Aligner；若目标栈不能满足要求，使用第 5 节的显式均分方案。首期不要求额外建设 CPU Aligner 服务，也不承诺所有辅助模型上 NPU。

## 4. 最小迁移范围

### 软件栈与单容器进程布局

2026-09-27 查阅的官方教程使用 `vllm-ascend:v0.23.0`；版本政策列出的 `0.23.0` / `0.23.0.post1` 对应 vLLM `0.23.0`、CANN `9.1.0`、PyTorch / torch_npu `2.10.0 / 2.10.0.post4`、Triton Ascend `3.2.2`。这是候选验证起点，**不是已验收的项目部署清单**。若缺少项目所需 API，再选完整匹配的后续发布组合或固定开发提交，不单独升级某个包。[官方版本政策](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)

目标环境应记录并冻结镜像 digest、CPU 架构、Python 和完整依赖清单；宿主 driver/firmware 必须与选定产品及 CANN 匹配，由平台方核实。当前 CUDA 镜像、CUDA extra 和 CPU 锁文件均不能直接覆盖 Ascend 基础环境。

建议保留一个容器内的两个进程：

| 进程 | 建议环境与职责 |
|---|---|
| 私有 R2T2 服务 `127.0.0.1:8001` | Ascend 基础 Python，运行共享 R2T2 NPU 推理，保留现有私有协议 |
| 公共 API `:8000`，默认映射 `17003` | 隔离 CPU venv，运行 API、音频处理、Nemotron 离线及实时标签；先接显式均分，真实对齐路径验收后再选择 |

Nemotron 当前由 API 网关承载，**不在私有 R2T2 引擎内**。独立环境可隔离其固定 Transformers 提交与 Ascend 依赖约束，但当前 `dev` 尚未完成此接线：两个进程都使用 `sys.executable`，不能将已有双进程等同于已实现双环境。[进程启动代码](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/deploy/entrypoint.py)、[实时网关](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/services/realtime/gateway.py)

| 修改位置 | 待完成内容 |
|---|---|
| `app/core/device.py`、设备指标 | 增加显式 NPU 校验与对应内存统计，设备不可用明确失败 |
| `deploy/entrypoint.py` | 分别选择两个解释器及环境，保留就绪等待和子进程失败联动退出；当前 `VLLM_PLUGINS=""` 需按 Ascend 插件加载要求调整并验证 |
| `app/utils/speaker_diarizer.py` | Nemotron 独立选择 CPU；不要继承主识别模型的 NPU 设备 |
| `app/services/realtime/engine.py` | 核对 processor、`AsyncLLM`、音频多模态输入、输出 token IDs、priority、abort/shutdown；先 eager，再逐项启用优化 |
| `app/services/asr/r2t2_engine.py`、模型资产检查 | 根据已选对齐方式初始化和检查模型；均分模式不加载未验收的 NPU Aligner |
| Ascend Docker/启动配置 | 复用旧分支部署约束，替换旧模型服务命令和依赖；预置固定 revision 权重并检查离线启动 |

共享协议常量会读取设备配置，API 内对齐器也按设备选择；因此不能只把 API 的 `DEVICE` 改为 `cpu`，就宣称所有接线完成。两进程的并发能力、配置报告与后端选择需一致。[共享协议](https://github.com/Quantatirsk/qwen3-asr/blob/2cfb68f/app/services/realtime/protocol.py)

## 5. Aligner 不可用时的均分语义

旧 Ascend 分支 `app/services/asr/uniform_alignment.py` 已有按**每个识别段**时长均分的算法：中文按 CJK 字符，连续其他字母/数字等组成单元，标点附在邻近单元上。它不使用声学证据，不能宣称精准对齐。迁移时复用算法，重新接入当前结果流：[旧实现](https://github.com/Quantatirsk/qwen3-asr/blob/5251d43/app/services/asr/uniform_alignment.py)。

1. 当前 `WordToken` 是**段内相对秒数**，旧实现生成绝对时间戳；不能直接复制。对于时长 `D`、`N` 个单元，第 `i` 个区间为 `[iD/N, (i+1)D/N]`，最后终点落在段末。空文本或零时长不生成区间。
2. 在段结果进入 `PreparedLongAudio.finish` 之前生成区间，条件保持 `word_timestamps or enable_speaker_diarization`。随后由现有链路统一偏移和缩放一次。
3. 不改写原始转写文本，保留全部文字与 `speaker_segments`。均分会把停顿摊入词时长，可能影响文字说话人归属；不确定归属仍允许 Unknown。
4. 明确标识估算方式。旧字段为 `word_timestamp_method="uniform_fallback"`，当前结果和序列化尚未接入；应补齐对外标识或在受限协议中明确文档说明。
5. 对齐模式是显式部署选择；已经选择真实 Aligner 后发生运行异常，不能静默改成均分。实时 `audio_ms` 仍表示处理进度，不改成词时间戳。

## 6. 实施顺序与验收门槛

1. **环境确认。** 收集目标设备和宿主软件版本；在平台提供的容器 shell 内检查 NPU 可见性、驱动库和模型缓存。冻结基础镜像及依赖，分别运行两个环境的依赖检查。缺设备或驱动挂载时由平台方补齐。
2. **最小 R2T2 推理。** 可先用官方 Qwen 教程确认环境，再加载固定 revision 的 R2T2。单卡 BF16、eager，验证短音频、中文/英文/混语、热词、静音及 60 秒边界。项目离线输出预算为 4096、默认总上下文为 16384，不能直接照抄教程的总长度 4096。
3. **接入原有实时/离线协议。** 保留私有 `/v1/config`、`/v1/transcribe`、`/v1/stream` 和公共兼容接口；验证长于滚动窗口的录音、结束补齐、追加文本不回写、断线取消、队列上限、会话隔离。先用低并发再验证调度优先级，不把 CUDA 默认并发当作 NPU 容量。
4. **接入 CPU Nemotron 与对齐方案。** 对齐四种开关组合；检查仅分段模式无说话人输出且不运行对齐。验证纯静音、语音夹长静音、重叠、低声、短促语音、噪声、音乐、长音频最大段长与零重复；空活动保留音频，模型失败明确报错且清理临时文件。均分补充非零偏移、缩放、中文/英文/标点和跨说话人边界检查。
5. **精度、容量和交付验收。** 用固定音频及人工标注评估文本、边界、说话人和时间戳；与 CUDA 结果比较但不要求跨后端逐字一致。长录音与实时并发下记录首字延迟、p95/p99、RTF、CPU/RSS/HBM、积压和错误率。验证无网启动、只读权重、重启、失败退出与健康检查。业务容量目标由实际需求确定。

第 2、3 步通过后才能报告“本台 910B 上 R2T2 链路可运行”；第 4、5 步通过后才能判断完整服务的准确性、容量和部署可用性。现阶段不承诺 NPU 性能、工期或 Nemotron 与 FSMN 边界等价。
