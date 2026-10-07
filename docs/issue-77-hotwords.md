# Issue #77：录音文件热词支持

调研及适配日期：2026-10-07。需求来自 [issue #77](https://github.com/Quantatirsk/AsrServe/issues/77)。

## 结论与原因

可以直接适配现有内核，无需增加模型或模型权重。[R2T2 官方说明](https://github.com/netease-youdao/Confucius4-R2T2/blob/master/README.zh.md)明确支持上下文和热词提示。

修复前，公开的 `/v1/audio/transcriptions` 没有 `hotwords` 字段，已有 `prompt` 被列为兼容参数并忽略。但内部 `OfflineTranscriptionOptions.hotwords` 已经沿服务、运行时、长录音引擎传到每个识别块，最终以私有 HTTP 接口的 `context` 传给 `Model._prompt()`，写入聊天模板的 system 内容。CUDA 与 Rust CPU 解码均使用这份提示词。

因此，缺失的是公开接口的接入。这项能力通过模型上下文引导识别，不是带数值权重的热词词典，也不是识别完成后的文本替换。

## 已完成适配

`/v1/audio/transcriptions` 的 multipart 表单新增可选字符串 `hotwords`，同时让兼容参数 `prompt` 生效：

| 参数 | 用途 | 示例 |
|---|---|---|
| `hotwords` | 人名、产品名等词语提示，逗号或换行分隔 | `省核投,光谷水投,武汉新城` |
| `prompt` | 录音主题、术语和其他识别上下文 | `这是水务投资项目会议。` |

两者都可用于 `file` 上传和 `audio_address` URL 转写。去除两端空白后，按 `prompt`、`hotwords` 顺序用换行连接，整体作为上下文传入每个识别块。合计最多 2048 个字符（含连接换行），超限在启动转写前返回 HTTP 400。未提供时，上下文为空。公开接口、内部 HTTP 客户端、私有接口和实时配置共用同一个长度上限常量。

```bash
curl http://localhost:17003/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F 'hotwords=省核投,光谷水投,武汉新城' \
  -F 'prompt=这是水务投资项目会议。' \
  -F response_format=verbose_json
```

只使用 OpenAI 兼容参数的客户端也可仅传 `prompt`，例如 `省核投、光谷水投、武汉新城`。

## 验证

全套测试 **174 项及 85 个子测试通过**；静态检查和 `git diff --check` 通过。新增检查覆盖接口文档可发现性、文件与 URL 两种输入、单独及合并提示、Unicode/空白处理、2048 字符边界、超限在转写前拒绝，以及公开 API 经真实转写服务和运行时传参。分块测试确认每个计算块收到相同热词。

真实推理使用本机会议录音的相同前 60 秒：在隔离容器中运行修改后的公开 API、音频预处理、运行时、分块识别和 CPU 标点恢复，共享运行中服务的真实 CUDA R2T2 引擎；本次热词实验模拟了说话人活动为空的分块提示，关闭说话人归属和词时间戳。

| 请求 | 该专名的实际输出 |
|---|---|
| 不传提示 | `省河投` |
| 仅传 `hotwords`，含 `省核投` | `省核投` |
| 仅传 `prompt`，指定公司名 `省核投` | `省核投` |
| 同时传主题 `prompt` 和 `hotwords` | `省核投` |
| 再次不传提示 | `省河投` |

公开请求的提示与每块发送到私有识别接口的 `context` 完全一致；最后一次空提示请求与第一次原始识别结果完全一致，验证提示未串到后续请求。这个对照证明上下文实际参与识别；该录音没有人工专名标注，不能据此计算热词准确率。

证据保存在本机 `logs/issue-77-punctuation/` 的 `hotwords-real.json`、`hotwords-real.log`、`hotwords-all-tests.log` 和验证脚本中。此次完成源码适配及隔离验证，正式运行镜像尚未更新。
