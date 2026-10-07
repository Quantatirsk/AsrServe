# OpenAI Realtime 转写

WebSocket 地址为 `ws://host:17003/v1/realtime?intent=transcription`，HTTPS 部署时使用 `wss://`。本接口适配 OpenAI Realtime 的**手动结束音频转写**流程，复用已有 R2T2 流式内核；不加载另一份识别权重或新增模型。

## 会话与音频

服务首先发送 `session.created`。客户端发送以下配置，收到 `session.updated` 后发送音频：

```json
{
  "type": "session.update",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": {
          "model": "confucius4-r2t2",
          "prompt": "会议主题为武汉新城项目。",
          "keywords": ["光谷水投", "东湖高新区"]
        },
        "turn_detection": null
      }
    }
  }
}
```

音频必须是 **24 kHz、单声道、PCM16 little-endian**，并用 Base64 编码后放在 `input_audio_buffer.append.audio` 中；原始二进制、十六进制字符串、WAV 文件头和压缩音频均不能作为 PCM 发送。建议每次发送 160 ms（7680 字节）的原始 PCM。服务连续重采样到内核的 16 kHz，并在音频到达时发送文字增量。

作为本服务扩展，嵌套 `format.rate` 也允许 `16000`，这时直接传入内核。其他采样率或编码返回错误。兼容客户端传来的转写模型名称，但模型不会切换；会话确认始终返回实际使用的 `confucius4-r2t2`。

`prompt`、`keywords` 和可选 `language` / `languages` 合并为 R2T2 上下文，最多 2048 个字符。语言字段是识别提示，不能同时指定 `language` 和 `languages`；热词属于识别引导，没有强制替换或权重参数。配置可部分更新：提示更新在下一轮音频生效；改变采样率需要先 `commit` 或 `clear`。

## 事件与多轮转写

| 客户端事件 | 服务行为 |
| --- | --- |
| `session.update` | 校验并更新配置，返回 `session.updated`；无效更新保留原配置 |
| `input_audio_buffer.append` | 立即将音频送入流式识别，返回可用的文字增量 |
| `input_audio_buffer.commit` | 结束当前轮，返回 `input_audio_buffer.committed` 及最终转写 |
| `input_audio_buffer.clear` | 取消当前未提交音频和识别，释放内核会话，返回 `input_audio_buffer.cleared` |

文字事件使用 `conversation.item.input_audio_transcription.delta` 和 `conversation.item.input_audio_transcription.completed`，关联同一个 `item_id` 和 `content_index: 0`。直接追加 `delta`；成功时它们的拼接与最终 `transcript` 一致。每一轮使用独立的音频滤波器、识别状态和 `item_id`；提交确认的 `previous_item_id` 可用于关联已经提交的上一轮。`clear` 后不再使用那轮的文字或音频状态，但已发送到客户端的增量需要客户端自行撤回。

提交时还发送 `conversation.item.added`，完成后发送 `conversation.item.done`。等待最终转写后可以在同一 WebSocket 中继续下一轮。空音频或不足 100 ms 的音频不能提交；可以继续追加或清除。

支持旧版平铺配置 `input_audio_format: "pcm16"`、`input_audio_transcription`、`turn_detection: null`。使用 `OpenAI-Beta: realtime=v1` 且 `intent=transcription`，或发送 `transcription_session.update`，可启用旧版 `transcription_session.created/updated` 和 `conversation.item.created` 确认；旧版 PCM 固定为 24 kHz，不能在正在识别的一轮中切换事件版本。

## OpenAI Python SDK 示例

客户端单独安装 `openai`；服务运行不依赖 OpenAI SDK，也不调用 OpenAI 云端。以下示例将一个 **24 kHz、单声道、16 位 PCM WAV** 按录音节奏发送；实际麦克风接入可直接发送同样格式的采集帧。

```python
import asyncio
import base64
import os
import wave

from openai import AsyncOpenAI


async def main():
    async with AsyncOpenAI(
        api_key=os.getenv("API_KEY", "local-asr-key"),
        base_url="http://localhost:17003/v1",
    ) as client:
        async with client.realtime.connect(
            extra_query={"intent": "transcription"}, max_retries=0
        ) as connection:
            completed = asyncio.get_running_loop().create_future()

            async def receive():
                try:
                    async for event in connection:
                        if event.type == "conversation.item.input_audio_transcription.delta":
                            print(event.delta, end="", flush=True)
                        elif event.type == "conversation.item.input_audio_transcription.completed":
                            completed.set_result(event.transcript)
                            return
                        elif event.type in ("error", "conversation.item.input_audio_transcription.failed"):
                            raise RuntimeError(str(event.error))
                    raise RuntimeError("Connection closed before transcription completed")
                except Exception as error:
                    if not completed.done():
                        completed.set_exception(error)

            receiver = asyncio.create_task(receive())
            try:
                await connection.session.update(session={
                    "type": "transcription",
                    "audio": {"input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "confucius4-r2t2", "prompt": "会议录音"},
                        "turn_detection": None,
                    }},
                })
                with wave.open("recording-24k.wav", "rb") as audio:
                    assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (24000, 1, 2)
                    while pcm := audio.readframes(3840):
                        await connection.input_audio_buffer.append(
                            audio=base64.b64encode(pcm).decode("ascii")
                        )
                        await asyncio.sleep(0.16)
                await connection.input_audio_buffer.commit()
                transcript = await asyncio.wait_for(completed, 30)
                print("\n最终转写：", transcript)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)


asyncio.run(main())
```

## 鉴权、容量与范围

设置 `API_KEY` 后，握手需要 `Authorization: Bearer <API_KEY>`；OpenAI SDK 会自动发送该头。浏览器可以使用 `?token=...`，或 OpenAI 浏览器客户端的 `realtime`、`openai-insecure-api-key.<API_KEY>` 子协议；服务只协商返回 `realtime`。

每次 `append` 最多 480000 字节 PCM（24 kHz 的 10 秒音频）；待处理 JSON 事件最多 8 条，内核另有 10 秒音频队列。一个 WebSocket 累计接受的音频最长 1 小时。连接 30 秒未收到客户端事件、内核 30 秒无响应或持续积压会返回错误并关闭；断线、清除和失败均释放内核会话。GPU 默认 4 个活跃流、CPU 默认 1 个，与原生流式接口共享容量；空闲连接在实际追加音频前不占推理名额。

无效配置、Base64、事件或空提交返回标准 `error`，可在同一连接修正后继续。内核不可用、满载或推理失败会结束连接；已经建立的转写轮失败时发送 `conversation.item.input_audio_transcription.failed`。

本接口不提供对话生成、TTS、WebRTC、服务端 VAD、降噪或 logprobs；`turn_detection` 必须为 `null`。它也不输出标准协议未定义的说话人标签或字词时间戳。需要实时主讲者标签时使用[原生 `/v1/stream`](realtime.md)，需要完整文件断句、说话人归属和字词时间戳时使用 [`/v1/audio/transcriptions`](../README.md#file-transcription)。实时转写仍按内核的流式结果输出，离线全文标点恢复不会用于已发布的实时增量。

协议依据：[OpenAI Realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription)、[Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events)。
