"""Independent ASR chunking respects inference limits without dropping tails."""

import tempfile
import unittest
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.core.config import settings
from app.utils.audio_splitter import AudioSplitter
from app.utils.speaker_diarizer import DiarizationResult, SpeakerSegment


class SegmentBoundsTest(unittest.TestCase):
    def test_long_vad_and_fixed_duration_preserve_all_intervals(self) -> None:
        with patch.object(settings, "MAX_SEGMENT_SEC", 60):
            splitter = AudioSplitter()
        for duration in (500, 60500, 120500, 125000):
            for segments in (
                splitter.merge_segments_greedy([(0, duration)], duration),
                splitter._split_by_fixed_duration(duration),
            ):
                self.assertEqual(segments[0][0], 0)
                self.assertEqual(segments[-1][1], duration)
                self.assertTrue(
                    all(0 < end - start <= 60000 for start, end in segments)
                )
                self.assertEqual(sum(end - start for start, end in segments), duration)
                self.assertTrue(all(a[1] == b[0] for a, b in pairwise(segments)))

    def test_diarization_intervals_replace_vad_without_losing_audio(self) -> None:
        overlapping = DiarizationResult(
            segments=[
                SpeakerSegment(1.0, 30.0, "说话人1", 0.9),
                SpeakerSegment(25.0, 70.0, "说话人2", 0.9),
                SpeakerSegment(90.0, 95.0, "说话人1", 0.9),
            ],
            probabilities=np.zeros((9500, 8), dtype=np.float32),
            frame_seconds=0.01,
            duration=95.0,
            speaker_ids=("说话人1", "说话人2") + (None,) * 6,
        )
        # Overlapping turns become one span; the silence between 70 s and 90 s is
        # kept as a gap so a cut can land there.
        self.assertEqual(
            overlapping.speech_intervals_ms(), [(1000, 70000), (90000, 95000)]
        )

        audio = np.zeros(95 * 16000, dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "MAX_SEGMENT_SEC", 60),
            patch(
                "app.utils.audio_splitter.librosa",
                SimpleNamespace(load=Mock(return_value=(audio, 16000))),
            ),
            patch.object(
                AudioSplitter,
                "get_vad_segments",
                side_effect=AssertionError("VAD must not run when diarization ran"),
            ),
            patch("app.utils.audio_splitter.sf.write"),
        ):
            segments = AudioSplitter().split_audio_file(
                "original.wav",
                directory,
                speech_segments=overlapping.speech_intervals_ms(),
            )
        self.assertTrue(all(0 < segment.duration_ms <= 60000 for segment in segments))
        self.assertEqual(segments[0].start_ms, 1000)
        self.assertEqual(segments[-1].end_ms, 95000)
        # Only the 70-90 s silence is dropped; no speech interval is truncated.
        self.assertEqual(
            sum(segment.duration_ms for segment in segments), 69000 + 5000
        )

    def test_one_sample_past_limit_is_split_and_retained(self) -> None:
        audio = np.zeros(60 * 16000 + 1, dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "MAX_SEGMENT_SEC", 60),
            patch(
                "app.utils.audio_splitter.librosa",
                SimpleNamespace(load=Mock(return_value=(audio, 16000))),
            ),
            patch.object(AudioSplitter, "get_vad_segments", return_value=[]),
            patch("app.utils.audio_splitter.sf.write"),
        ):
            segments = AudioSplitter().split_audio_file("original.wav", directory)
            self.assertGreater(len(segments), 1)
            self.assertTrue(
                all(len(item.audio_data) <= 60 * 16000 for item in segments)
            )
            self.assertEqual(sum(len(item.audio_data) for item in segments), len(audio))


if __name__ == "__main__":
    unittest.main()
