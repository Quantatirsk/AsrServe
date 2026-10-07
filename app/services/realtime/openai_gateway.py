"""OpenAI transcription events over the existing shared R2T2 stream."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import uuid
from contextlib import suppress

import numpy as np
import soxr
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.security import validate_token_value, validate_websocket_token

from .client import get_capabilities, open_stream, receive_event
from .openai_protocol import (
    MAX_APPEND_BYTES,
    MAX_EVENT_BYTES,
    TranscriptionSessionConfig,
)
from .protocol import (
    BYTES_PER_SECOND,
    MAX_SECONDS,
    SAMPLE_RATE,
    StreamConfig,
    StreamError,
    supervise,
)

logger = logging.getLogger(__name__)


def _id(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex


class PCMResampler:
    """Keep the filter and fractional sample phase across append messages."""

    def __init__(self, rate: int):
        self.stream = (
            soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")
            if rate != SAMPLE_RATE
            else None
        )

    def push(self, data: bytes, *, final: bool = False) -> bytes:
        audio = np.frombuffer(data, dtype="<i2")
        if self.stream is None:
            return data
        output = self.stream.resample_chunk(audio.astype(np.float32), last=final)
        # Convert once, without libsoxr's random integer dithering. Filter overshoot
        # must saturate instead of wrapping around the PCM16 range.
        return np.clip(np.rint(output), -32768, 32767).astype("<i2").tobytes()


class _Turn:
    def __init__(self, session: _Session):
        self.session = session
        self.item_id = _id("item")
        self.previous_item_id = session.previous_item_id
        self.rate = session.config.audio.input.format.rate
        self.resampler = PCMResampler(self.rate)
        self.samples = 0
        self.text = ""
        self.committed = False
        self.failed = False
        self.context = None
        self.upstream = None
        self.reader = None

    async def start(self) -> None:
        context = open_stream(
            StreamConfig(
                context=self.session.config.audio.input.transcription.context()
            )
        )
        self.upstream, ready = await context.__aenter__()
        self.context = context
        self.max_samples = min(MAX_SECONDS, ready["max_session_seconds"]) * self.rate
        self.reader = asyncio.create_task(self.read())

    async def send_pcm(self, data: bytes) -> None:
        for start in range(0, len(data), BYTES_PER_SECOND):
            await asyncio.wait_for(
                self.upstream.send(data[start : start + BYTES_PER_SECOND]), 5
            )

    async def append(self, data: bytes) -> None:
        if self.samples + len(data) // 2 > self.max_samples:
            raise StreamError("session_limit", "Maximum turn audio duration exceeded")
        self.samples += len(data) // 2
        await self.send_pcm(self.resampler.push(data))

    async def read(self) -> None:
        try:
            while True:
                event = await receive_event(self.upstream)
                delta = event.get("delta")
                if not isinstance(delta, str) or type(event.get("done")) is not bool:
                    raise StreamError(
                        "upstream_error", "Invalid R2T2 stream event", 502
                    )
                self.text += delta
                if delta:
                    await self.session.send(
                        "conversation.item.input_audio_transcription.delta",
                        item_id=self.item_id,
                        content_index=0,
                        delta=delta,
                    )
                if event["done"]:
                    if not self.committed or event.get("text") != self.text:
                        raise StreamError(
                            "upstream_error",
                            "R2T2 stream completed inconsistently",
                            502,
                        )
                    await self.session.send(
                        "conversation.item.input_audio_transcription.completed",
                        item_id=self.item_id,
                        content_index=0,
                        transcript=self.text,
                    )
                    if not self.session.legacy:
                        await self.session.item(
                            self, "conversation.item.done", transcript=self.text
                        )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.fail(error)

    async def fail(self, error: Exception) -> None:
        if self.failed:
            return
        self.failed = True
        code = error.code if isinstance(error, StreamError) else "stream_failed"
        message = (
            str(error)
            if isinstance(error, StreamError)
            else "Realtime transcription failed"
        )
        try:
            await self.session.send(
                "conversation.item.input_audio_transcription.failed",
                item_id=self.item_id,
                content_index=0,
                error={"type": "server_error", "code": code, "message": message},
            )
        finally:
            if not self.session.failure.done():
                self.session.failure.set_result(error)

    async def finish(self) -> None:
        await self.send_pcm(self.resampler.push(b"", final=True))
        await asyncio.wait_for(self.upstream.send("end"), 5)
        await asyncio.wait_for(asyncio.shield(self.reader), 30)
        if self.session.failure.done():
            raise self.session.failure.result()

    async def close(self) -> None:
        if self.reader is not None:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.context is not None:
            context, self.context = self.context, None
            await context.__aexit__(None, None, None)


class _Session:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.session_id = _id("sess")
        self.config = TranscriptionSessionConfig()
        self.legacy = (
            "realtime=v1" in websocket.headers.get("openai-beta", "")
            and websocket.query_params.get("intent") == "transcription"
        )
        self.previous_item_id = None
        self.turn = None
        self.samples = 0
        self.lock = asyncio.Lock()
        self.failure = asyncio.get_running_loop().create_future()

    async def send(self, kind: str, **fields) -> None:
        async with self.lock:
            await asyncio.wait_for(
                self.ws.send_json({"type": kind, "event_id": _id("event"), **fields}), 5
            )

    async def error(
        self,
        code: str,
        message: str,
        *,
        event_id=None,
        param=None,
        error_type="invalid_request_error",
    ) -> None:
        await self.send(
            "error",
            error={
                "type": error_type,
                "code": code,
                "message": message,
                "param": param,
                "event_id": event_id,
            },
        )

    async def item(self, turn: _Turn, kind: str, *, transcript=None) -> None:
        await self.send(
            kind,
            previous_item_id=turn.previous_item_id,
            item={
                "id": turn.item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_audio", "transcript": transcript}],
            },
        )

    async def clear(self) -> None:
        if self.turn is not None:
            turn, self.turn = self.turn, None
            await turn.close()

    async def dispatch(self, event: dict) -> None:
        kind = event.get("type")
        if kind in ("session.update", "transcription_session.update"):
            data = event.get("session")
            if not isinstance(data, dict):
                raise StreamError("invalid_session", "session must be an object")
            try:
                config = self.config.update(data)
            except (ValidationError, ValueError) as error:
                raise StreamError("invalid_session", str(error)) from error
            legacy = self.legacy or kind == "transcription_session.update"
            if legacy and config.audio.input.format.rate != 24000:
                raise StreamError(
                    "invalid_session", "Legacy pcm16 sessions require 24 kHz audio"
                )
            if self.turn and legacy != self.legacy:
                raise StreamError(
                    "invalid_session",
                    "Commit or clear before changing the event schema",
                )
            if (
                self.turn
                and config.audio.input.format != self.config.audio.input.format
            ):
                raise StreamError(
                    "invalid_session",
                    "Commit or clear the current turn before changing its audio format",
                )
            self.config = config
            self.legacy = legacy
            prefix = "transcription_session" if self.legacy else "session"
            await self.send(
                prefix + ".updated",
                session=self.config.snapshot(self.session_id, legacy=self.legacy),
            )
        elif kind == "input_audio_buffer.append":
            encoded = event.get("audio")
            if (
                not isinstance(encoded, str)
                or len(encoded) > (MAX_APPEND_BYTES + 2) // 3 * 4
            ):
                raise StreamError(
                    "invalid_audio",
                    "audio must be Base64 PCM16, at most 10 seconds at 24 kHz per append",
                )
            try:
                audio = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise StreamError(
                    "invalid_audio", "audio must be valid Base64"
                ) from error
            if len(audio) % 2 or len(audio) > MAX_APPEND_BYTES:
                raise StreamError(
                    "invalid_audio",
                    "audio must contain complete little-endian PCM16 samples",
                )
            if not audio:
                return
            rate = self.config.audio.input.format.rate
            if (
                self.samples + len(audio) // 2 * SAMPLE_RATE / rate
                > MAX_SECONDS * SAMPLE_RATE
            ):
                raise StreamError(
                    "session_limit", "Maximum session audio duration exceeded"
                )
            if self.turn is None:
                self.turn = _Turn(self)
                await self.turn.start()
            await self.turn.append(audio)
            self.samples += len(audio) // 2 * SAMPLE_RATE / rate
        elif kind == "input_audio_buffer.commit":
            if self.turn is None or self.turn.samples < self.turn.rate // 10:
                raise StreamError(
                    "input_audio_buffer_commit_empty",
                    "Commit requires at least 100 ms of audio",
                )
            turn = self.turn
            turn.committed = True
            await self.send(
                "input_audio_buffer.committed",
                item_id=turn.item_id,
                previous_item_id=self.previous_item_id,
            )
            await self.item(
                turn,
                "conversation.item.created"
                if self.legacy
                else "conversation.item.added",
            )
            self.previous_item_id = turn.item_id
            try:
                await turn.finish()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await turn.fail(error)
                raise
            finally:
                await self.clear()
        elif kind == "input_audio_buffer.clear":
            await self.clear()
            await self.send("input_audio_buffer.cleared")
        else:
            raise StreamError(
                "unsupported_event",
                "Supported events: session.update, input_audio_buffer.append, commit and clear",
            )

    async def run(self) -> None:
        await get_capabilities()
        prefix = "transcription_session" if self.legacy else "session"
        await self.send(
            prefix + ".created",
            session=self.config.snapshot(self.session_id, legacy=self.legacy),
        )
        queue = asyncio.Queue(maxsize=8)

        async def receive():
            while True:
                message = await asyncio.wait_for(self.ws.receive(), 30)
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None or len(text.encode("utf-8")) > MAX_EVENT_BYTES:
                    await self.error(
                        "invalid_event", "Expected a bounded JSON text event"
                    )
                    continue
                try:
                    event = json.loads(text)
                except ValueError:
                    await self.error("invalid_event", "Invalid JSON")
                    continue
                if not isinstance(event, dict) or not isinstance(
                    event.get("type"), str
                ):
                    await self.error(
                        "invalid_event", "Event must be an object with a type"
                    )
                    continue
                event_id = event.get("event_id")
                if event_id is not None and (
                    not isinstance(event_id, str) or len(event_id) > 512
                ):
                    await self.error(
                        "invalid_event",
                        "event_id must be a string of at most 512 characters",
                    )
                    continue
                await asyncio.wait_for(queue.put(event), 5)

        async def consume():
            while True:
                event = await queue.get()
                try:
                    await self.dispatch(event)
                except StreamError as error:
                    if error.status >= 500:
                        raise
                    await self.error(
                        error.code, str(error), event_id=event.get("event_id")
                    )

        async def failed():
            raise await self.failure

        await supervise(receive(), consume(), failed())


async def handle_openai_realtime(websocket: WebSocket) -> None:
    valid, _ = validate_websocket_token(websocket)
    protocols = websocket.scope.get("subprotocols", [])
    if not valid:
        valid = any(
            validate_token_value(p.removeprefix("openai-insecure-api-key."))
            for p in protocols
            if p.startswith("openai-insecure-api-key.")
        )
    if not valid:
        await websocket.close(code=1008)
        return
    await websocket.accept(subprotocol="realtime" if "realtime" in protocols else None)
    session = _Session(websocket)
    try:
        if websocket.query_params.get("intent", "transcription") != "transcription":
            raise StreamError(
                "unsupported_session", "Only transcription sessions are supported"
            )
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception as error:
        if isinstance(error, StreamError):
            code, message = error.code, str(error)
        elif isinstance(error, TimeoutError):
            code, message = (
                "session_timeout",
                "Session timed out or consumer is too slow",
            )
        else:
            logger.exception("OpenAI transcription gateway failed")
            code, message = "stream_failed", "Realtime transcription failed"
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            if session.turn is not None:
                await session.turn.fail(error)
            error_type = (
                "invalid_request_error"
                if isinstance(error, StreamError) and error.status < 500
                else "server_error"
            )
            await session.error(code, message, error_type=error_type)
    finally:
        try:
            await session.clear()
        finally:
            with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
                await asyncio.wait_for(websocket.close(), 2)
