import asyncio
import io
import os
from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1 import api_router
from app.core.config import settings
from app.services.asr.engines import ASRFullResult, ASRSegmentResult, WordToken
from app.services.realtime.protocol import MAX_CONTEXT_CHARACTERS, MODEL_ID, StreamError
from deploy import entrypoint as launcher
from app.utils.speaker_diarizer import SpeakerSegment


class APIContractTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(api_router)
        self.app = app
        self.client = TestClient(app)
        self.result = ASRFullResult(
            "hello",
            [ASRSegmentResult("hello", 0, 1, "speaker1", [WordToken("hello", 0, 1)])],
            1,
        )

        async def start_transcription(**kwargs):
            async def transcribe():
                return self.result

            return asyncio.create_task(transcribe())

        self.service = SimpleNamespace(
            start_transcription=AsyncMock(side_effect=start_transcription),
        )

    def test_offline_all_response_formats(self):
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            for fmt in ("json", "verbose_json", "text", "srt", "vtt"):
                with self.subTest(format=fmt):
                    response = self.client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"fake", "audio/wav")},
                        data={
                            "model": MODEL_ID,
                            "response_format": fmt,
                            "word_timestamps": "true",
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIn("hello", response.text)
                    if fmt == "verbose_json":
                        self.assertEqual(
                            response.json()["segments"][0]["speaker"], "speaker1"
                        )
                        self.assertEqual(response.json()["words"][0]["word"], "hello")
                    if fmt == "vtt":
                        self.assertTrue(response.text.startswith("WEBVTT"))
        self.assertEqual(self.service.start_transcription.await_count, 5)

    def test_overlap_metadata_and_fractional_offsets_survive(self) -> None:
        segments = [
            ASRSegmentResult(
                "Mixed. ",
                3.25,
                4,
                word_tokens=[WordToken("Mixed", 0.125, 0.375)],
                speaker_candidates=["speaker1", "speaker2"],
            ),
            ASRSegmentResult(
                "Again.", 4.5, 5, "speaker1", [WordToken("Again", 0, 0.2)]
            ),
        ]
        spans = [
            SpeakerSegment(3, 4, "speaker1", 0.9),
            SpeakerSegment(3.5, 4.5, "speaker2", 0.8),
        ]
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            for words in (True, False):
                self.result = ASRFullResult(
                    "Mixed. Again.",
                    [
                        replace(
                            segment, word_tokens=segment.word_tokens if words else None
                        )
                        for segment in segments
                    ],
                    5,
                    speaker_segments=spans,
                )
                with self.subTest(words=words):
                    response = self.client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"fake", "audio/wav")},
                        data={
                            "response_format": "verbose_json",
                            "word_timestamps": str(words).lower(),
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    payload = response.json()
                    self.assertEqual(
                        payload["segments"][0]["speaker_candidates"],
                        ["speaker1", "speaker2"],
                    )
                    self.assertIsNone(payload["segments"][0]["speaker"])
                    self.assertEqual(payload["segments"][0]["start"], 3.25)
                    self.assertEqual(
                        payload["speaker_segments"],
                        [
                            {
                                "start": 3,
                                "end": 4,
                                "speaker": "speaker1",
                                "confidence": 0.9,
                            },
                            {
                                "start": 3.5,
                                "end": 4.5,
                                "speaker": "speaker2",
                                "confidence": 0.8,
                            },
                        ],
                    )
                    if words:
                        self.assertEqual(
                            payload["words"],
                            [
                                {"word": "Mixed", "start": 3.375, "end": 3.625},
                                {"word": "Again", "start": 4.5, "end": 4.7},
                            ],
                        )
                    else:
                        self.assertIsNone(payload.get("words"))
                    self.assertEqual(
                        self.service.start_transcription.call_args.kwargs[
                            "options"
                        ].word_timestamps,
                        words,
                    )

    def test_offline_default_model(self) -> None:
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", b"fake")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.service.start_transcription.assert_awaited_once()

    def test_r2t2_model_with_audio_url(self) -> None:
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": MODEL_ID,
                    "audio_address": "https://example.test/audio.wav",
                    "response_format": "verbose_json",
                    "word_timestamps": "true",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["words"][0]["word"], "hello")
        self.service.start_transcription.assert_awaited_once()
        options = self.service.start_transcription.call_args.kwargs
        self.assertIsNone(options["audio_data"])
        self.assertEqual(options["audio_address"], "https://example.test/audio.wav")

    def test_upload_and_url_context_reaches_transcription_options(self) -> None:
        cases = (
            ({}, ""),
            (
                {"hotwords": "  光谷水投,武汉新城\nAda & Alan  "},
                "光谷水投,武汉新城\nAda & Alan",
            ),
            ({"prompt": "  水务项目会议  "}, "水务项目会议"),
            (
                {"prompt": "水务项目会议", "hotwords": "光谷水投,武汉新城"},
                "水务项目会议\n光谷水投,武汉新城",
            ),
            ({"prompt": " \t", "hotwords": " "}, ""),
            (
                {"hotwords": "词" * MAX_CONTEXT_CHARACTERS},
                "词" * MAX_CONTEXT_CHARACTERS,
            ),
            (
                {"prompt": "话" * 1024, "hotwords": "词" * 1023},
                "话" * 1024 + "\n" + "词" * 1023,
            ),
        )
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ):
            for upload in (True, False):
                for fields, expected in cases:
                    with self.subTest(upload=upload, fields=tuple(fields)):
                        source = (
                            {"files": {"file": ("test.wav", b"audio")}}
                            if upload
                            else {}
                        )
                        data = {"response_format": "json", **fields}
                        if not upload:
                            data["audio_address"] = "https://example.test/audio.wav"
                        response = self.client.post(
                            "/v1/audio/transcriptions", data=data, **source
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        options = self.service.start_transcription.call_args.kwargs[
                            "options"
                        ]
                        self.assertEqual(options.hotwords, expected)

    def test_oversized_context_fails_before_starting_transcription(self) -> None:
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ) as get_service:
            for fields in (
                {"hotwords": "词" * (MAX_CONTEXT_CHARACTERS + 1)},
                {"prompt": "x" * (MAX_CONTEXT_CHARACTERS + 1)},
                {"prompt": "话" * 1024, "hotwords": "词" * 1024},
            ):
                with self.subTest(fields=tuple(fields)):
                    response = self.client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"audio")},
                        data=fields,
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()["error_code"], "INVALID_PARAMETER")
            get_service.assert_not_called()
            self.service.start_transcription.assert_not_awaited()

    def test_api_schema_exposes_supported_hotwords_and_prompt(self) -> None:
        schema = self.app.openapi()
        body = schema["paths"]["/v1/audio/transcriptions"]["post"]["requestBody"]
        reference = body["content"]["multipart/form-data"]["schema"]["$ref"]
        fields = schema["components"]["schemas"][reference.rsplit("/", 1)[1]][
            "properties"
        ]
        self.assertIn("hotwords", fields)
        self.assertNotIn("暂不支持", fields["prompt"]["description"])

    def test_any_model_value_uses_configured_transcription_service(self) -> None:
        names = (
            "",
            "arbitrary-model",
            "qwen3-asr",
            "qwen3-asr-0.6b",
            "qwen3-asr-1.7b",
            "Qwen/Qwen3-ASR-1.7B",
            "whisper-1",
            "paraformer-large",
            "custom-local-model",
            "netease-youdao/Confucius4-R2T2",
            "Confucius4-R2T2",
        )
        with patch(
            "app.api.v1.openai_compatible.get_offline_transcription_service",
            return_value=self.service,
        ) as get_service:
            for model in names:
                with self.subTest(model=model):
                    response = self.client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"fake")},
                        data={"model": model},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["text"], "hello")
        self.assertEqual(get_service.call_count, len(names))
        self.assertEqual(self.service.start_transcription.await_count, len(names))

    def test_models_do_not_probe_remote_availability(self) -> None:
        with patch(
            "app.services.realtime.client.get_capabilities",
            side_effect=StreamError("unavailable", "test", 503),
        ) as get_capabilities:
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([model["id"] for model in response.json()["data"]], [MODEL_ID])
        self.assertEqual(response.json()["data"][0]["owned_by"], "netease-youdao")
        get_capabilities.assert_not_called()

    def test_health_does_not_borrow_busy_offline_engine(self):
        runtime = SimpleNamespace(
            resolve_model_id=Mock(return_value=MODEL_ID),
            get_loaded_model_ids=Mock(return_value=[MODEL_ID]),
            get_memory_usage=Mock(return_value={}),
            acquire_engine=AsyncMock(
                side_effect=AssertionError("health borrowed engine")
            ),
        )
        with patch("app.api.v1.get_runtime_router", return_value=runtime):
            with patch.object(settings, "DEVICE", "cuda:0"):
                self.assertTrue(
                    self.client.get("/health").json()["model_loaded"]
                )
                runtime.get_loaded_model_ids.return_value = []
                self.assertFalse(
                    self.client.get("/health").json()["model_loaded"]
                )
        runtime.acquire_engine.assert_not_called()

    def test_startup_probe_authenticates_to_the_health_route(self) -> None:
        runtime = SimpleNamespace(
            resolve_model_id=Mock(return_value=MODEL_ID),
            get_loaded_model_ids=Mock(return_value=[MODEL_ID]),
            get_memory_usage=Mock(return_value={}),
        )

        def fetch(
            request: launcher.urllib.request.Request, timeout: float
        ) -> io.BytesIO:
            response = self.client.get(
                "/health", headers=dict(request.header_items())
            )
            body = io.BytesIO(response.content)
            body.status = response.status_code
            return body

        with (
            patch.dict(os.environ, {"API_KEY": "health-test-token"}),
            patch.object(settings, "API_KEY", "health-test-token"),
            patch("app.api.v1.get_runtime_router", return_value=runtime),
            patch.object(settings, "DEVICE", "cuda:0"),
            patch.object(launcher.urllib.request, "urlopen", side_effect=fetch),
        ):
            self.assertTrue(launcher.healthy(launcher.API_URL, "model_loaded"))

    def test_realtime_auth_and_unavailable_engine(self):
        with patch.object(settings, "API_KEY", "secret-token-123"):
            self.assertEqual(self.client.get("/v1/config").status_code, 401)
            # Browsers cannot set headers, so the stream also accepts ?token=.
            with self.assertRaises(WebSocketDisconnect):
                with self.client.websocket_connect("/v1/stream"):
                    pass
            with patch.object(settings, "R2T2_URL", ""):
                with self.client.websocket_connect(
                    "/v1/stream?token=secret-token-123"
                ) as ws:
                    ws.send_json({})
                    self.assertEqual(
                        ws.receive_json()["code"], "realtime_unavailable"
                    )
