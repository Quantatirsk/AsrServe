# Issue #84 / PR #85 审阅与独立实现

审阅对象：[Issue #84](https://github.com/Quantatirsk/AsrServe/issues/84)、[PR #85](https://github.com/Quantatirsk/AsrServe/pull/85)，PR 提交 `c5190623e998b6c4c35ea6c84ae6e51ed2545cd0`，基线 `89fee7ace48d18fee8faa808020a680efeb78e4b`。仅将两个文件的差异作为审阅资料读取，没有检出、执行、合并或 cherry-pick 该 PR。

## 审阅结果

1. **P1：提交音频必然调用失败。** [第 141 行](https://github.com/Quantatirsk/AsrServe/blob/c5190623e998b6c4c35ea6c84ae6e51ed2545cd0/app/services/realtime/openai_realtime.py#L141)调用 `runtime_router.transcribe`，但 PR 基线及当前 `ASRRuntimeRouter` 均无此方法。它实际返回失败事件，无法完成一次转写；只检查语法和路由注册发现不了此问题。
2. **P1：配置 API_KEY 后仍允许匿名访问新入口。** [第 161 行](https://github.com/Quantatirsk/AsrServe/blob/c5190623e998b6c4c35ea6c84ae6e51ed2545cd0/app/services/realtime/openai_realtime.py#L161)直接接受 WebSocket，未使用项目已有的鉴权函数，绕过现有访问控制。
3. **P1：音频期间没有实时结果。** [第 118 行](https://github.com/Quantatirsk/AsrServe/blob/c5190623e998b6c4c35ea6c84ae6e51ed2545cd0/app/services/realtime/openai_realtime.py#L118)只向内存追加音频，直到 `commit` 才重采样并调用转写；`send_delta` 仅定义、从未调用。即使修正方法名，直播字幕或语音客户端仍得不到录音期间的增量。
4. **P1：音频缓存和会话无上限。** 同一 `bytearray` 接受持续追加，没有长度、PCM 偶数字节、会话时长或积压限制。连接不提交即可持续增长内存；重采样又在异步处理函数中同步执行，会阻塞其他连接。
5. **P2：配置和协议确认缺失。** [第 109 行](https://github.com/Quantatirsk/AsrServe/blob/c5190623e998b6c4c35ea6c84ae6e51ed2545cd0/app/services/realtime/openai_realtime.py#L109)只保存并记录配置，采样格式、语言和提示均未应用；没有 `session.created/updated`、`input_audio_buffer.committed/cleared` 等确认。无效 JSON、未知事件和部分音频错误也被忽略。等待确认或错误事件的客户端可能一直等待。
6. **P2：示例和功能说明与协议不符。** Issue 示例使用 `.hex()`，而处理函数按 Base64 解码，十六进制字符会被错误解码为另一份音频。Issue 中的 `transcription.text` / `transcription.completed` 不是此 WebSocket 转写流程的标准事件；所称说话人标签、字词时间戳也未实现。PR 还将完成结果的语言固定默认为 `en`，无法据此判断真实识别语言。

这两个文件的差异中未发现额外安装命令或外联代码；以上结论是可定位的功能、协议和访问控制问题，不推断作者意图。

## 独立实现

新代码位于 `app/services/realtime/openai_gateway.py` 和 `openai_protocol.py`，由路由注册 `/v1/realtime`。它使用现有私有 `open_stream` 客户端，复用已经运行的 R2T2 权重、GPU/CPU 流式能力和容量控制，不经过不存在的文件转写方法。

收到 Base64 PCM 后立即处理，使用具有连续状态的 libsoxr 重采样器将 24 kHz 转为 16 kHz；提交时补齐滤波尾部。输出使用实际增量和最终文本，按标准事件关联每一轮的 `item_id`。清除、断线、容量拒绝和后端失败均取消读取并关闭内核连接。新入口复用 API_KEY 校验，限制事件、输入量、时长和等待时间，并返回可恢复的客户端错误。

嵌套 GA 配置及旧版平铺转写配置经过校验，支持手动提交、同一连接多轮、上下文提示与关键词。服务端 VAD、对话生成等超出识别适配范围的配置明确返回错误。说话人和字词时间戳仍由现有原生/离线接口提供；完整使用方式见 [OpenAI Realtime 转写](openai-realtime.md)。

`soxr==1.0.0` 从已有 librosa 间接依赖声明为直接依赖，没有增加新识别模型或另一个推理服务。

## 验证

- 完整测试集通过：**191 项测试、85 项子测试**；新接口覆盖 17 项测试。新代码静态检查、差异空白检查及 `uv lock --check` 均通过。
- 自动化覆盖真实事件顺序、提交前增量、多轮文本隔离、清除与重采样重置、24 kHz 任意分帧后与整段重采样结果一致、滤波尾部、鉴权、无效输入、配置原子更新、旧版事件、容量拒绝、时长限制、后端失败、最终文本一致性和提交时断开回收。
- 使用 OpenAI Python SDK **2.32.0** 连接隔离的开发 API，再转发到已有真实 R2T2 内核。使用与 issue 提到的 Wyoming 客户端一致的嵌套会话配置和手动 `commit`，先清除一轮已进入内核的音频，再在同一连接转写中文 6.74 秒、英文 15.05 秒录音。
- 中文收到 13 个非空增量，首个增量约 1.85 秒；英文收到 36 个非空增量，首个增量约 1.50 秒。两轮均在 `commit` 前收到文字，最终文本等于所有增量拼接，上一轮 item 关联正确。这是当前环境中的链路验证，不是延迟保证或准确率评测。
- 验证使用临时开发 API，不替换当前运行的服务镜像；OpenAI SDK 只用于验证，没有请求 OpenAI 云端。

协议参考：[官方 Realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription)。Wyoming 客户端参考：[wyoming_openai](https://github.com/roryeckel/wyoming_openai)。
