"""Hardware-free smoke checks for the Ascend split environment and timing path."""

import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.core.config import Settings, settings
from app.core.device import detect_device
from app.services.asr.model_capabilities import get_huggingface_assets
from app.services.asr.r2t2_engine import R2T2Engine
from app.services.asr.uniform_alignment import uniform_word_timestamps
from app.services.asr.long_audio import PreparedLongAudio
from app.utils.audio_splitter import AudioSegment
from app.utils.speaker_diarizer import DiarizationResult, SpeakerSegment
from app.api.v1.openai_compatible import build_transcription_payload, ResponseFormat
from deploy.entrypoint import services


def test_ascend_environments_and_device_validation():
    with patch.dict(
        os.environ,
        {"DEVICE": "npu:0", "R2T2_PYTHON": "/npu/python", "API_PYTHON": "/cpu/python"},
    ):
        config = Settings()
        engine, api = services()
    assert config.ALIGNMENT_MODE == "uniform"
    assert config.SPEAKER_DIARIZATION_DEVICE == "cpu"
    assert engine.command[0] == "/npu/python"
    assert api.command[0] == "/cpu/python"
    assert "ascend_model" in engine.env["VLLM_PLUGINS"].split(",")
    assert engine.env["DEVICE"] == api.env["DEVICE"] == "npu:0"
    assert api.env["R2T2_URL"] == "http://127.0.0.1:8001"
    npu = SimpleNamespace(is_available=Mock(return_value=True), set_device=Mock())
    with (
        patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}),
        patch("torch.npu", npu, create=True),
    ):
        assert detect_device("npu:0") == "npu:0"
        npu.set_device.assert_called_once_with(0)
        npu.is_available.return_value = False
        with pytest.raises(RuntimeError, match="available Ascend"):
            detect_device("npu:0")
    with patch.dict(os.environ, {"DEVICE": "npu:0", "ALIGNMENT_MODE": "forced"}):
        with pytest.raises(ValueError, match="requires ALIGNMENT_MODE"):
            Settings()


def test_npu_engine_uses_bf16_eager_and_priority():
    from app.services.realtime.engine import Model

    args = Mock(side_effect=lambda **kwargs: kwargs)
    llm = SimpleNamespace(from_engine_args=Mock())
    modules = {
        "vllm": SimpleNamespace(SamplingParams=Mock(side_effect=lambda **kw: kw)),
        "vllm.engine.arg_utils": SimpleNamespace(AsyncEngineArgs=args),
        "vllm.transformers_utils.processors.qwen3_asr": SimpleNamespace(
            Qwen3ASRProcessor=SimpleNamespace(from_pretrained=Mock())
        ),
        "vllm.v1.engine.async_llm": SimpleNamespace(AsyncLLM=llm),
    }
    with (
        patch.dict(sys.modules, modules),
        patch.dict(os.environ, {}, clear=True),
        patch.object(settings, "DEVICE", "npu:0"),
        patch("app.core.device.detect_device", return_value="npu:0"),
    ):
        model = Model(4)
    options = llm.from_engine_args.call_args.args[0]
    assert options["dtype"] == "bfloat16"
    assert options["enforce_eager"] is True
    assert options["max_model_len"] == 16384
    assert options["scheduling_policy"] == "priority"
    assert model.chunk_samples == 10240


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("words", [False, True])
def test_uniform_pipeline_switches_and_serialization(labels, words):
    with (
        patch.object(settings, "DEVICE", "npu:0"),
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
                return_value="你好，world!",
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
    assert result.text == "你好，world!"
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


def test_nemotron_loads_on_cpu_even_when_asr_is_npu():
    from app.utils.speaker_diarizer import SpeakerDiarizer

    model = Mock(config=SimpleNamespace(head_config=SimpleNamespace(num_speakers=8)))
    model.to.return_value = model
    model.eval.return_value = model
    processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000, hop_length=160)
    )
    with (
        patch.object(settings, "DEVICE", "npu:0"),
        patch.object(settings, "SPEAKER_DIARIZATION_DEVICE", "cpu"),
        patch(
            "transformers.AutoModelForAudioFrameClassification.from_pretrained",
            return_value=model,
        ),
        patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
    ):
        diarizer = SpeakerDiarizer()
        diarizer.warmup()
    model.to.assert_called_once_with("cpu")
    diarizer.close()
