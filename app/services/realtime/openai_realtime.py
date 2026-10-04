"""OpenAI Realtime API compatible WebSocket endpoint.

Implements the OpenAI Realtime transcription protocol at /v1/realtime.
Accepts 24 kHz PCM audio, resamples to 16 kHz for the R2T2 engine.
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services.asr.runtime import get_runtime_router

from .protocol import StreamError

logger = logging.getLogger(__name__)

# OpenAI Realtime API uses 24 kHz, R2T2 uses 16 kHz
OPENAI_SAMPLE_RATE = 24000
R2T2_SAMPLE_RATE = 16000


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another.

    Args:
        audio: Input audio array (float32, mono)
        from_rate: Input sample rate
        to_rate: Output sample rate

    Returns:
        Resampled audio array
    """
    if from_rate == to_rate:
        return audio

    try:
        import librosa

        return librosa.resample(audio, orig_sr=from_rate, target_sr=to_rate)
    except ImportError:
        # Fallback: simple linear resampling
        import scipy.signal

        ratio = to_rate / from_rate
        num_samples = int(len(audio) * ratio)
        return scipy.signal.resample(audio, num_samples)


class OpenAIRealtimeSession:
    """Manages a single OpenAI Realtime API session."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.session_id = str(uuid.uuid4())
        self.item_id = str(uuid.uuid4())
        self.audio_buffer = bytearray()
        self.config: dict[str, Any] = {}
        self.closed = False

    async def send_event(self, event: dict[str, Any]) -> None:
        """Send an event to the client."""
        if self.closed:
            return
        try:
            await self.websocket.send_text(json.dumps(event))
        except Exception:
            self.closed = True

    async def send_delta(self, delta: str) -> None:
        """Send a transcription delta event."""
        await self.send_event({
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": self.item_id,
            "content_index": 0,
            "delta": delta,
        })

    async def send_completed(self, transcript: str, language: str = "en") -> None:
        """Send a transcription completed event."""
        await self.send_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": self.item_id,
            "content_index": 0,
            "transcript": transcript,
            "language": language,
        })

    async def send_failed(self, code: str, message: str) -> None:
        """Send a transcription failed event."""
        await self.send_event({
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": self.item_id,
            "content_index": 0,
            "error": {
                "code": code,
                "message": message,
            },
        })

    async def handle_session_update(self, data: dict[str, Any]) -> None:
        """Handle a session.update event."""
        self.config = data.get("session", {})
        logger.info(f"Session {self.session_id} updated: {self.config}")

    async def handle_audio_append(self, data: dict[str, Any]) -> None:
        """Handle an input_audio_buffer.append event."""
        audio_b64 = data.get("audio", "")
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                self.audio_buffer.extend(audio_bytes)
            except Exception as e:
                logger.error(f"Failed to decode audio: {e}")

    async def handle_audio_commit(self) -> None:
        """Handle an input_audio_buffer.commit event."""
        if not self.audio_buffer:
            return

        try:
            # Convert bytes to float32 numpy array
            audio_int16 = np.frombuffer(self.audio_buffer, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Resample from 24 kHz to 16 kHz
            audio_resampled = resample_audio(audio_float, OPENAI_SAMPLE_RATE, R2T2_SAMPLE_RATE)

            # Get the runtime router and transcribe
            runtime_router = get_runtime_router()
            model_id = runtime_router.resolve_model_id(None)

            # Transcribe the audio
            result = await asyncio.to_thread(
                runtime_router.transcribe,
                model_id,
                audio_resampled,
                R2T2_SAMPLE_RATE,
            )

            # Send completed event
            await self.send_completed(result.text)

        except Exception as e:
            logger.error(f"Transcription failed for session {self.session_id}: {e}")
            await self.send_failed("transcription_error", str(e))
        finally:
            # Reset buffer for next utterance
            self.audio_buffer = bytearray()
            self.item_id = str(uuid.uuid4())


async def handle_openai_realtime(websocket: WebSocket) -> None:
    """Handle an OpenAI Realtime API WebSocket connection."""
    await websocket.accept()

    session = OpenAIRealtimeSession(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "session.update":
                await session.handle_session_update(event)
            elif event_type == "input_audio_buffer.append":
                await session.handle_audio_append(event)
            elif event_type == "input_audio_buffer.commit":
                await session.handle_audio_commit()
            elif event_type == "input_audio_buffer.clear":
                session.audio_buffer = bytearray()
            else:
                logger.debug(f"Unhandled event type: {event_type}")

    except WebSocketDisconnect:
        logger.info(f"Session {session.session_id} disconnected")
    except Exception as e:
        logger.error(f"Error in session {session.session_id}: {e}")
    finally:
        session.closed = True
