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
    def test_long_activity_and_fixed_duration_preserve_all_intervals(self) -> None:
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
            patch("app.utils.audio_splitter.sf.write"),
        ):
            segments = AudioSplitter().split_audio_file(
                "original.wav",
                directory,
                speech_segments=overlapping.speech_intervals_ms(),
            )
        self.assertTrue(all(0 < segment.duration_ms <= 60000 for segment in segments))
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[-1].end_ms, 95000)
        # Retain the full timeline, including gaps, so low activity cannot drop words.
        self.assertEqual(sum(segment.duration_ms for segment in segments), 95000)

    def test_pack_activity_until_budget_and_reuse_pause_boundaries(self) -> None:
        splitter = AudioSplitter()
        spans = [(i * 1000, i * 1000 + 500) for i in range(125)]
        chunks = splitter.merge_segments_greedy(spans, 125000)
        self.assertEqual(chunks, [(0, 60000), (60000, 120000), (120000, 125000)])
        self.assertEqual(sum(end - start for start, end in chunks), 125000)
        self.assertEqual(
            splitter.merge_segments_greedy([(0, 58000), (59000, 125000)], 125000),
            [(0, 58500), (58500, 118500), (118500, 125000)],
        )

    def test_empty_activity_retains_silence_and_unrecognized_signal_samples(
        self,
    ) -> None:
        size = 125 * 16000
        signals = {
            "silence": np.zeros(size, dtype=np.float32),
            "quiet_noise": np.random.default_rng(7)
            .normal(0, 0.0001, size)
            .astype(np.float32),
        }
        for name, audio in signals.items():
            with (
                self.subTest(signal=name),
                tempfile.TemporaryDirectory() as directory,
                patch.object(settings, "MAX_SEGMENT_SEC", 60),
                patch(
                    "app.utils.audio_splitter.librosa",
                    SimpleNamespace(load=Mock(return_value=(audio, 16000))),
                ),
                patch("app.utils.audio_splitter.sf.write"),
            ):
                segments = AudioSplitter().split_audio_file(
                    "original.wav", directory, speech_segments=[]
                )
                self.assertTrue(all(s.duration_ms <= 60000 for s in segments))
                np.testing.assert_array_equal(
                    np.concatenate([s.audio_data for s in segments]), audio
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
            patch("app.utils.audio_splitter.sf.write"),
        ):
            segments = AudioSplitter().split_audio_file(
                "original.wav", directory, speech_segments=[]
            )
            self.assertGreater(len(segments), 1)
            self.assertTrue(
                all(len(item.audio_data) <= 60 * 16000 for item in segments)
            )
            self.assertEqual(sum(len(item.audio_data) for item in segments), len(audio))


if __name__ == "__main__":
    unittest.main()
