"""Validated configuration for the OpenAI transcription WebSocket subset."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .protocol import MAX_CONTEXT_CHARACTERS, MODEL_ID

OPENAI_SAMPLE_RATE = 24000
MAX_APPEND_BYTES = OPENAI_SAMPLE_RATE * 2 * 10
MAX_EVENT_BYTES = (MAX_APPEND_BYTES + 2) // 3 * 4 + 4096


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PCMFormat(StrictConfig):
    type: Literal["audio/pcm"] = "audio/pcm"
    rate: Literal[16000, 24000] = OPENAI_SAMPLE_RATE


class Transcription(StrictConfig):
    model: str = Field(default=MODEL_ID, min_length=1, max_length=128)
    prompt: str | None = Field(default="", max_length=MAX_CONTEXT_CHARACTERS)
    language: str | None = Field(default=None, min_length=2, max_length=16)
    languages: list[str] | None = Field(default=None, max_length=32)
    keywords: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_hints(self):
        if self.language and self.languages:
            raise ValueError("Use language or languages, not both")
        if self.languages and any(
            not word.strip() or len(word) > 16 for word in self.languages
        ):
            raise ValueError(
                "Language hints must be nonempty codes of at most 16 characters"
            )
        if any(
            not word.strip() or any(c in word for c in "<>\r\n")
            for word in self.keywords
        ):
            raise ValueError(
                "Keywords must be nonempty, single-line terms without < or >"
            )
        if len(self.context()) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(
                f"Combined transcription hints exceed {MAX_CONTEXT_CHARACTERS} characters"
            )
        return self

    def context(self) -> str:
        languages = self.languages or (
            [self.language] if self.language and self.language != "auto" else []
        )
        parts = [
            "Languages: " + ", ".join(languages) if languages else "",
            (self.prompt or "").strip(),
            ", ".join(self.keywords),
        ]
        return "\n".join(part for part in parts if part)


class AudioInput(StrictConfig):
    format: PCMFormat = Field(default_factory=PCMFormat)
    transcription: Transcription = Field(default_factory=Transcription)
    # R2T2 streams continuously; the client owns utterance boundaries.
    turn_detection: None = None
    noise_reduction: None = None


class AudioConfig(StrictConfig):
    input: AudioInput = Field(default_factory=AudioInput)


class TranscriptionSessionConfig(StrictConfig):
    type: Literal["transcription"] = "transcription"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    include: list[str] = Field(default_factory=list, max_length=0)

    def update(self, data: dict) -> TranscriptionSessionConfig:
        """Apply partial updates atomically; accept the older flat ASR schema."""
        data = dict(data)
        flat = {"input_audio_format", "input_audio_transcription", "turn_detection"}
        if flat.intersection(data):
            if "audio" in data:
                raise ValueError("Use nested audio or flat audio fields, not both")
            audio = {}
            if "input_audio_format" in data:
                if data.pop("input_audio_format") != "pcm16":
                    raise ValueError("Only pcm16 input is supported")
                audio["format"] = {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE}
            if "input_audio_transcription" in data:
                audio["transcription"] = data.pop("input_audio_transcription")
            if "turn_detection" in data:
                audio["turn_detection"] = data.pop("turn_detection")
            data["audio"] = {"input": audio}

        def merge(old: dict, new: dict) -> dict:
            result = dict(old)
            for key, value in new.items():
                result[key] = (
                    merge(result[key], value)
                    if isinstance(result.get(key), dict) and isinstance(value, dict)
                    else value
                )
            return result

        return self.model_validate(merge(self.model_dump(), data))

    def snapshot(self, session_id: str, *, legacy: bool = False) -> dict:
        data = self.model_dump()
        # Compatibility model names are accepted, but report the actual ASR.
        data["audio"]["input"]["transcription"]["model"] = MODEL_ID
        if legacy:
            return {
                "id": session_id,
                "object": "realtime.transcription_session",
                "input_audio_format": "pcm16",
                "input_audio_transcription": data["audio"]["input"]["transcription"],
                "turn_detection": None,
            }
        return {"id": session_id, "object": "realtime.transcription_session", **data}
