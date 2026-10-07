"""Public OpenAI events must stream, preserve audio and release native sessions."""

import asyncio
import base64
import json
import unittest
from contextlib import ExitStack, asynccontextmanager
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1.realtime import router
from app.core.config import settings
from app.services.realtime.openai_gateway import PCMResampler
from app.services.realtime.openai_protocol import (
    MAX_APPEND_BYTES,
    TranscriptionSessionConfig,
)
from app.services.realtime.protocol import MODEL_ID, StreamError

DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"


class FakeUpstream:
    def __init__(self, context):
        self.context = context
        self.queue = asyncio.Queue()
        self.audio = bytearray()
        self.text = ""
        self.closed = False
        self.fail = False
        self.hold_end = False
        self.bad_final = False
        self.end_sent = False
        self.frames = []

    async def send(self, data):
        if self.fail:
            await self.queue.put(
                {"error": "Inference failed", "code": "inference_failed"}
            )
        elif isinstance(data, bytes):
            self.frames.append(data)
            self.audio.extend(data)
            if not self.text:
                self.text = "Hello 中文"
                await self.queue.put({"delta": self.text, "done": False})
        elif data == "end":
            self.end_sent = True
            if not self.hold_end:
                self.text += "!"
                await self.queue.put(
                    {
                        "delta": "!",
                        "done": True,
                        "text": "wrong" if self.bad_final else self.text,
                    }
                )

    async def recv(self):
        return json.dumps(await self.queue.get())


class OpenAIRealtimeTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(settings, "API_KEY", None))
        self.upstreams = []
        self.capabilities = self.stack.enter_context(
            patch(
                "app.services.realtime.openai_gateway.get_capabilities",
                new_callable=AsyncMock,
                return_value={"max_session_seconds": 3600},
            )
        )
        self.next_error = None
        self.fail_next = False
        self.hold_end = False
        self.bad_final = False
        self.max_seconds = 3600

        @asynccontextmanager
        async def open_stream(config):
            if self.next_error:
                raise self.next_error
            upstream = FakeUpstream(config.context)
            upstream.fail = self.fail_next
            upstream.hold_end = self.hold_end
            upstream.bad_final = self.bad_final
            self.upstreams.append(upstream)
            try:
                yield upstream, {"max_session_seconds": self.max_seconds}
            finally:
                upstream.closed = True

        self.stack.enter_context(
            patch("app.services.realtime.openai_gateway.open_stream", open_stream)
        )
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def configure(self, ws, *, rate=16000, prompt="Names: Ada & 中文", **extra):
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": rate},
                            "transcription": {
                                "model": "gpt-4o-transcribe",
                                "prompt": prompt,
                            },
                            "turn_detection": None,
                            **extra,
                        }
                    },
                },
            }
        )
        event = ws.receive_json()
        self.assertEqual(event["type"], "session.updated")
        self.assertEqual(
            event["session"]["audio"]["input"]["transcription"]["model"], MODEL_ID
        )

    def append(self, ws, data=None):
        if data is None:
            data = np.arange(5120, dtype="<i2").tobytes()
        ws.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(data).decode(),
            }
        )
        return data

    def commit(self, ws):
        ws.send_json({"type": "input_audio_buffer.commit"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "conversation.item.done":
                return events
            if event["type"] == "error":
                self.fail(str(event))

    def test_streams_before_commit_and_supports_ordered_multiple_turns(self):
        with self.client.websocket_connect("/v1/realtime?intent=transcription") as ws:
            created = ws.receive_json()
            self.assertEqual(created["type"], "session.created")
            self.assertEqual(created["session"]["type"], "transcription")
            self.configure(ws)
            previous = None
            for _ in range(2):
                audio = self.append(ws)
                partial = ws.receive_json()  # Must arrive before any commit.
                self.assertEqual(partial["type"], DELTA)
                events = self.commit(ws)
                self.assertEqual(
                    [e["type"] for e in events],
                    [
                        "input_audio_buffer.committed",
                        "conversation.item.added",
                        DELTA,
                        COMPLETED,
                        "conversation.item.done",
                    ],
                )
                item = partial["item_id"]
                self.assertEqual(events[0]["previous_item_id"], previous)
                self.assertEqual(events[0]["item_id"], item)
                self.assertEqual(events[1]["previous_item_id"], previous)
                self.assertEqual(events[-1]["previous_item_id"], previous)
                self.assertEqual(events[-1]["item"]["id"], item)
                self.assertEqual(
                    events[-2]["transcript"], partial["delta"] + events[2]["delta"]
                )
                self.assertTrue(all(e.get("item_id", item) == item for e in events))
                previous = item
                self.assertEqual(bytes(self.upstreams[-1].audio), audio)
            self.assertTrue(
                all(u.context == "Names: Ada & 中文" for u in self.upstreams)
            )
        self.assertTrue(all(u.closed for u in self.upstreams))

    def test_clear_cancels_old_turn_and_resets_text_and_resampler(self):
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws)
            self.append(ws)
            first = ws.receive_json()["item_id"]
            ws.send_json({"type": "input_audio_buffer.clear"})
            self.assertEqual(ws.receive_json()["type"], "input_audio_buffer.cleared")
            self.assertTrue(self.upstreams[0].closed)
            self.append(ws)
            second = ws.receive_json()["item_id"]
            self.assertNotEqual(first, second)
            events = self.commit(ws)
            self.assertIsNone(events[0]["previous_item_id"])
            self.assertEqual(events[-2]["transcript"], "Hello 中文!")
        self.assertTrue(all(u.closed for u in self.upstreams))

    def test_24khz_audio_is_resampled_continuously_and_flushed_at_commit(self):
        audio = np.round(np.sin(np.arange(24000) * 0.1) * 16000).astype("<i2").tobytes()
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws, rate=24000)
            for start in range(0, len(audio), 638):
                self.append(ws, audio[start : start + 638])
            events = self.commit(ws)
            self.assertEqual(events[-2]["type"], COMPLETED)
        output = bytes(self.upstreams[0].audio)
        single = PCMResampler(24000)
        expected = single.push(audio) + single.push(b"", final=True)
        self.assertEqual(output, expected)
        self.assertEqual(len(output), 16000 * 2)
        self.assertTrue(
            all(
                0 < len(frame) <= 32000 and len(frame) % 2 == 0
                for frame in self.upstreams[0].frames
            )
        )

    def test_invalid_events_audio_and_empty_commit_are_recoverable(self):
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            for invalid in ("{", "[]", "null"):
                ws.send_text(invalid)
                self.assertEqual(ws.receive_json()["error"]["code"], "invalid_event")
            for event in (
                {"type": "input_audio_buffer.append", "audio": "%%"},
                {"type": "input_audio_buffer.append", "audio": "中文"},
                {"type": "input_audio_buffer.append", "audio": "YQ=="},
                {"type": "input_audio_buffer.commit"},
                {"type": "response.create"},
            ):
                ws.send_json(dict(event, event_id="client_error_1"))
                error = ws.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["error"]["event_id"], "client_error_1")
            self.assertEqual(self.upstreams, [])
            self.configure(ws)
            self.append(ws)
            ws.receive_json()
            self.assertEqual(self.commit(ws)[-2]["type"], COMPLETED)

    def test_unsupported_configuration_fails_atomically_and_remains_usable(self):
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            for data in (
                {"type": "realtime"},
                {"audio": {"input": {"format": {"rate": 48000}}}},
                {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}},
                {"audio": {"input": {"transcription": {"prompt": "x" * 2049}}}},
                {"input_audio_format": "g711_ulaw"},
            ):
                ws.send_json({"type": "session.update", "session": data})
                self.assertEqual(ws.receive_json()["error"]["code"], "invalid_session")
            self.configure(ws, prompt="valid")
            self.append(ws)
            ws.receive_json()
            self.commit(ws)
        self.assertEqual(self.upstreams[0].context, "valid")

    def test_updates_apply_to_next_turn_and_format_changes_require_clear(self):
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws, prompt="first")
            self.append(ws)
            ws.receive_json()
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {"input": {"transcription": {"prompt": "second"}}}
                    },
                }
            )
            self.assertEqual(ws.receive_json()["type"], "session.updated")
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {"audio": {"input": {"format": {"rate": 24000}}}},
                }
            )
            self.assertEqual(ws.receive_json()["error"]["code"], "invalid_session")
            self.commit(ws)
            self.append(ws)
            ws.receive_json()
            self.commit(ws)
        self.assertEqual([u.context for u in self.upstreams], ["first", "second"])

    def test_legacy_flat_transcription_session_and_event_names(self):
        with self.client.websocket_connect(
            "/v1/realtime?intent=transcription", headers={"OpenAI-Beta": "realtime=v1"}
        ) as ws:
            self.assertEqual(ws.receive_json()["type"], "transcription_session.created")
            ws.send_json(
                {
                    "type": "transcription_session.update",
                    "session": {
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": "whisper-1",
                            "prompt": "legacy",
                        },
                        "turn_detection": None,
                    },
                }
            )
            self.assertEqual(ws.receive_json()["type"], "transcription_session.updated")
            self.append(ws, b"\0\0" * 12000)
            self.assertEqual(ws.receive_json()["type"], DELTA)
            ws.send_json({"type": "input_audio_buffer.commit"})
            events = [ws.receive_json() for _ in range(4)]
            self.assertEqual(
                [e["type"] for e in events],
                [
                    "input_audio_buffer.committed",
                    "conversation.item.created",
                    DELTA,
                    COMPLETED,
                ],
            )
        self.assertEqual(self.upstreams[0].context, "legacy")

    def test_authentication_is_checked_before_backend_access(self):
        with patch.object(settings, "API_KEY", "secret-token-123"):
            for args in ({}, {"headers": {"Authorization": "Bearer wrong-token-123"}}):
                with self.assertRaises(WebSocketDisconnect) as error:
                    with self.client.websocket_connect("/v1/realtime", **args):
                        pass
                self.assertEqual(error.exception.code, 1008)
            self.capabilities.assert_not_awaited()
            for path, args in (
                (
                    "/v1/realtime",
                    {"headers": {"Authorization": "Bearer secret-token-123"}},
                ),
                ("/v1/realtime?token=secret-token-123", {}),
                (
                    "/v1/realtime",
                    {
                        "subprotocols": [
                            "realtime",
                            "openai-insecure-api-key.secret-token-123",
                        ]
                    },
                ),
            ):
                with self.client.websocket_connect(path, **args) as ws:
                    self.assertEqual(ws.receive_json()["type"], "session.created")
                    if "subprotocols" in args:
                        self.assertEqual(ws.accepted_subprotocol, "realtime")

    def test_backend_failure_emits_failed_and_releases_session(self):
        self.fail_next = True
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws)
            self.append(ws)
            self.assertEqual(ws.receive_json()["type"], FAILED)
            self.assertEqual(ws.receive_json()["type"], "error")
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()
        self.assertTrue(self.upstreams[0].closed)

    def test_capacity_rejection_is_a_protocol_error(self):
        self.next_error = StreamError("capacity_exceeded", "Busy", 503)
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.append(ws)
            self.assertEqual(ws.receive_json()["type"], FAILED)
            self.assertEqual(ws.receive_json()["error"]["code"], "capacity_exceeded")
        self.assertEqual(self.upstreams, [])

    def test_disconnect_during_commit_releases_waiting_native_stream(self):
        self.hold_end = True
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws)
            self.append(ws)
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.commit"})
            self.assertEqual(ws.receive_json()["type"], "input_audio_buffer.committed")
            self.assertEqual(ws.receive_json()["type"], "conversation.item.added")
        self.assertTrue(self.upstreams[0].closed)

    def test_large_append_is_rejected_without_allocating_backend(self):
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.append(ws, b"\0" * (MAX_APPEND_BYTES + 2))
            self.assertEqual(ws.receive_json()["error"]["code"], "invalid_audio")
        self.assertEqual(self.upstreams, [])

    def test_backend_duration_limit_rejects_audio_and_allows_commit(self):
        self.max_seconds = 1
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws)
            self.append(ws, b"\0\0" * 16000)
            ws.receive_json()
            self.append(ws, b"\0\0")
            self.assertEqual(ws.receive_json()["error"]["code"], "session_limit")
            self.commit(ws)
        self.assertEqual(len(self.upstreams[0].audio), 32000)
        self.assertTrue(self.upstreams[0].closed)

    def test_inconsistent_final_transcript_fails_and_releases_backend(self):
        self.bad_final = True
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws)
            self.append(ws)
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.commit"})
            events = [ws.receive_json() for _ in range(5)]
            self.assertEqual(events[-2]["type"], FAILED)
            self.assertEqual(events[-1]["error"]["code"], "upstream_error")
            self.assertNotIn(COMPLETED, [e["type"] for e in events])
        self.assertTrue(self.upstreams[0].closed)

    def test_legacy_schema_cannot_misreport_16khz_pcm_or_switch_mid_turn(self):
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            self.configure(ws)
            ws.send_json({"type": "transcription_session.update", "session": {}})
            self.assertEqual(ws.receive_json()["error"]["code"], "invalid_session")
            self.configure(ws, rate=24000)
            self.append(ws, b"\0\0" * 12000)
            ws.receive_json()
            ws.send_json({"type": "transcription_session.update", "session": {}})
            self.assertEqual(ws.receive_json()["error"]["code"], "invalid_session")
            self.commit(ws)


class RealtimeConfigTest(unittest.TestCase):
    def test_partial_updates_preserve_settings_and_combine_hints(self):
        config = TranscriptionSessionConfig().update(
            {
                "audio": {
                    "input": {
                        "transcription": {
                            "prompt": "Names",
                            "keywords": ["Ada", "中文"],
                            "language": "en",
                        }
                    }
                }
            }
        )
        updated = config.update(
            {"audio": {"input": {"transcription": {"prompt": "Meeting"}}}}
        )
        self.assertEqual(
            updated.audio.input.transcription.context(),
            "Languages: en\nMeeting\nAda, 中文",
        )
        self.assertEqual(config.audio.input.transcription.prompt, "Names")

    def test_resampler_bypass_and_empty_flush(self):
        pcm = np.arange(17, dtype="<i2").tobytes()
        stream = PCMResampler(16000)
        self.assertEqual(stream.push(pcm), pcm)
        self.assertEqual(stream.push(b"", final=True), b"")
        self.assertEqual(PCMResampler(24000).push(b"", final=True), b"")
