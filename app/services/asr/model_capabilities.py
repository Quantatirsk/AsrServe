# -*- coding: utf-8 -*-
"""Shared capability-to-model asset definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.config import settings
from app.services.realtime.protocol import MODEL_REPOSITORY, MODEL_REVISION
from .punctuation import MODEL_HASHES, MODEL_ID, MODEL_REVISION as PUNCTUATION_REVISION


@dataclass(frozen=True)
class ModelAsset:
    model_id: str
    description: str
    revision: Optional[str] = None
    required_patterns: tuple[str, ...] = ()
    alternative_required_patterns: tuple[tuple[str, ...], ...] = ()
    min_total_size_bytes: int = 0
    local_dir: str | None = None
    hub: str = "huggingface"
    file_hashes: tuple[tuple[str, str], ...] = ()


def get_huggingface_assets() -> list[ModelAsset]:
    """Return the diarizer, shared ASR checkpoint, and timestamp aligner."""
    return [
        ModelAsset(
            model_id="nvidia/Nemotron-3-Diarization",
            revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e",
            description="Nemotron Speaker Diarization",
            required_patterns=(
                "config.json",
                "processor_config.json",
                "model.safetensors",
            ),
            min_total_size_bytes=100_000_000,
            local_dir=settings.NEMOTRON_MODEL_PATH,
        ),
        ModelAsset(
            model_id=MODEL_REPOSITORY,
            description="Confucius4-R2T2",
            revision=MODEL_REVISION,
            required_patterns=(
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
            alternative_required_patterns=(
                ("model.safetensors",),
                ("model.safetensors.index.json", "model-*.safetensors"),
            ),
            min_total_size_bytes=500_000_000,
        ),
        ModelAsset(
            model_id="Qwen/Qwen3-ForcedAligner-0.6B",
            description="Forced Aligner",
            required_patterns=("config.json", "model.safetensors"),
            min_total_size_bytes=500_000_000,
        ),
    ]


def get_model_assets() -> list[ModelAsset]:
    return [
        *get_huggingface_assets(),
        ModelAsset(
            model_id=MODEL_ID,
            description="CT-Transformer Punctuation",
            revision=PUNCTUATION_REVISION,
            required_patterns=("config.yaml", "model.pt", "tokens.json"),
            min_total_size_bytes=290_000_000,
            local_dir=settings.PUNCTUATION_MODEL_PATH,
            hub="modelscope",
            file_hashes=MODEL_HASHES,
        ),
    ]
