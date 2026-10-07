"""Both alignment modes preserve timestamp and speaker response semantics."""

import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.api.v1.openai_compatible import ResponseFormat, build_transcription_payload
from app.core.config import Settings, settings
from app.services.asr.model_capabilities import get_huggingface_assets
from app.services.asr.long_audio import PreparedLongAudio
from app.services.asr.r2t2_engine import R2T2Engine
from app.services.asr.uniform_alignment import uniform_word_timestamps
from app.utils.audio_splitter import AudioSegment
from app.utils.speaker_diarizer import DiarizationResult, SpeakerSegment


def test_alignment_mode_validation_and_required_assets() -> None:
    with patch.dict(os.environ, {"ALIGNMENT_MODE": "invalid"}):
        with pytest.raises(ValueError, match="ALIGNMENT_MODE"):
            Settings()
    for mode, count in (("uniform", 2), ("forced", 3)):
        with patch.dict(os.environ, {"ALIGNMENT_MODE": mode}):
            assert Settings().ALIGNMENT_MODE == mode
        with patch.object(settings, "ALIGNMENT_MODE", mode):
            assert len(get_huggingface_assets()) == count


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("words", [False, True])
def test_uniform_pipeline_switches_and_serialization(labels: bool, words: bool) -> None:
    with (
        patch.object(settings, "DEVICE", "cpu"),
        patch.object(settings, "ALIGNMENT_MODE", "uniform"),
        patch("app.services.asr.r2t2_engine.ForcedAligner") as forced,
        patch("app.services.asr.r2t2_engine.RustForcedAligner") as rust,
    ):
        engine = R2T2Engine()
        assert len(get_huggingface_assets()) == 2
        forced.assert_not_called()
        rust.assert_not_called()
    activity = DiarizationResult(
        [SpeakerSegment(5, 7, "A", 0.9)],
        np.ones((700, 8)),
        0.01,
        7,
        ("A", None, None, None, None, None, None, None),
    )
    with tempfile.NamedTemporaryFile() as source:
        segment = AudioSegment(5000, 7000, temp_file=source.name)
        prepared = PreparedLongAudio([segment], 7, activity if labels else None)
        context = Mock(
            __enter__=Mock(return_value=prepared), __exit__=Mock(return_value=False)
        )
        with (
            patch(
                "app.services.asr.r2t2_engine.prepare_long_audio", return_value=context
            ),
            patch(
                "app.services.asr.r2t2_engine._load_audio", return_value=np.zeros(32000)
            ),
            patch(
                "app.services.asr.r2t2_engine.transcribe_segment",
                return_value="\u4f60\u597d\uff0cworld!",
            ),
            patch(
                "app.services.asr.r2t2_engine.restore_punctuation",
                side_effect=lambda texts: list(texts),
            ),
            patch(
                "app.services.asr.r2t2_engine.uniform_word_timestamps",
                wraps=uniform_word_timestamps,
            ) as align,
        ):
            result = engine.transcribe_long_audio(
                source.name,
                enable_speaker_diarization=labels,
                word_timestamps=words,
                timestamp_scale=2,
            )
        assert align.call_count == int(labels or words)
    assert result.text == "\u4f60\u597d\uff0cworld!"
    assert result.word_timestamp_method == (
        "uniform_fallback" if labels or words else None
    )
    assert (result.speaker_segments is not None) == labels
    assert bool(result.segments[0].word_tokens) == words
    payload, _, _ = build_transcription_payload(
        response_format=ResponseFormat.VERBOSE_JSON,
        asr_result=result,
        audio_duration=result.duration,
        language=None,
    )
    assert payload["word_timestamp_method"] == result.word_timestamp_method
    if words:
        assert payload["words"][0]["start"] == 10
        assert payload["words"][-1]["end"] == 14
    engine.close()
    assert uniform_word_timestamps("", 1) == []
    assert uniform_word_timestamps("hello", 0) == []
