import unittest

import numpy as np

from app.services.asr.engines.base import ASRSegmentResult, WordToken
from app.services.asr.qwen3_alignment import split_alignment_units
from app.services.asr.speaker_attribution import (
    assign_speakers,
    consolidate_speaker_turns,
)
from app.utils.speaker_diarizer import DiarizationResult, SpeakerSegment


def assign_one(result, activity):
    return assign_speakers([result], activity)


def diarization(
    spans: list[tuple[float, float, str]], duration: float = 5.0
) -> DiarizationResult:
    labels = list(dict.fromkeys(speaker for _, _, speaker in spans))
    probabilities = np.zeros((int(duration * 100), 8), dtype=np.float32)
    for start, end, speaker in spans:
        probabilities[int(start * 100) : int(end * 100), labels.index(speaker)] = 0.9
    return DiarizationResult(
        segments=[
            SpeakerSegment(start, end, speaker, 0.9) for start, end, speaker in spans
        ],
        probabilities=probabilities,
        frame_seconds=0.01,
        duration=duration,
        speaker_ids=tuple(labels + [None] * (8 - len(labels))),
    )


def transcript(
    text: str,
    timings: list[tuple[float, float]],
    start: float = 0.0,
    end: float = 5.0,
) -> ASRSegmentResult:
    units = split_alignment_units(text)
    assert len(units) == len(timings)
    return ASRSegmentResult(
        text=text,
        start_time=start,
        end_time=end,
        word_tokens=[
            WordToken(unit, lower, upper)
            for unit, (lower, upper) in zip(units, timings)
        ],
    )


class SpeakerAttributionTest(unittest.TestCase):
    def assert_preserved(
        self, original: ASRSegmentResult, output: list[ASRSegmentResult]
    ) -> None:
        self.assertEqual("".join(segment.text for segment in output), original.text)
        actual = [
            (
                word.text,
                segment.start_time + word.start_time,
                segment.start_time + word.end_time,
            )
            for segment in output
            for word in segment.word_tokens or []
        ]
        expected = [
            (
                word.text,
                original.start_time + word.start_time,
                original.start_time + word.end_time,
            )
            for word in original.word_tokens or []
        ]
        self.assertEqual([item[0] for item in actual], [item[0] for item in expected])
        np.testing.assert_allclose(
            [item[1:] for item in actual], [item[1:] for item in expected]
        )
        for segment in output:
            for word in segment.word_tokens or []:
                self.assertGreaterEqual(word.start_time, 0)
                self.assertLessEqual(
                    word.end_time, segment.end_time - segment.start_time + 1e-6
                )

    def test_single_speaker_preserves_punctuation_spaces_and_numbers(self) -> None:
        text = "  \u4eca\u5929\uff0c Qwen3 it's 12.5%: caf\u00e9!\n"
        units = split_alignment_units(text)
        original = transcript(
            text, [(i * 0.3, i * 0.3 + 0.2) for i in range(len(units))]
        )
        output = assign_one(original, diarization([(0, 5, "speaker-1")]))
        self.assert_preserved(original, output)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].speaker_id, "speaker-1")

    def test_short_interjection_survives_and_times_are_rebased(self) -> None:
        original = transcript(
            "Start. Yes! Continue.", [(0, 0.8), (1, 1.1), (1.2, 2)], 10, 13
        )
        output = assign_one(
            original,
            diarization(
                [
                    (10, 11, "speaker-1"),
                    (11, 11.1, "speaker-2"),
                    (11.1, 13, "speaker-1"),
                ],
                13,
            ),
        )
        self.assert_preserved(original, output)
        self.assertEqual(
            [segment.speaker_id for segment in output],
            ["speaker-1", "speaker-2", "speaker-1"],
        )
        self.assertEqual(
            [segment.text for segment in output], ["Start. ", "Yes! ", "Continue."]
        )
        self.assertAlmostEqual(output[1].word_tokens[0].start_time, 0)
        self.assertAlmostEqual(output[1].word_tokens[0].end_time, 0.1)

    def test_overlap_uses_stable_labels_without_duplicate_text(self) -> None:
        original = transcript("Mixed voices.", [(0, 1), (1, 2)])
        activity = diarization([(0, 2, "speaker-1"), (0, 2, "speaker-2")])
        output = assign_one(original, activity)
        self.assert_preserved(original, output)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].speaker_id, "speaker-1")
        self.assertIsNone(output[0].speaker_candidates)

    def test_switch_inside_word_uses_greatest_coverage(self) -> None:
        original = transcript("Across.", [(0, 1)])
        output = assign_one(
            original, diarization([(0, 0.6, "speaker-1"), (0.6, 1, "speaker-2")])
        )
        self.assertEqual(output[0].speaker_id, "speaker-1")
        self.assert_preserved(original, output)

    def test_missing_activity_keeps_previous_speaker(self) -> None:
        original = transcript("One. Two.", [(0, 1), (2, 3)])
        output = assign_one(original, diarization([(0, 0.3, "speaker-2")]))
        self.assert_preserved(original, output)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].speaker_id, "speaker-2")

    def test_no_diarization_retains_transcript(self) -> None:
        original = transcript("Still here!", [(0, 1), (1, 2)])
        output = assign_one(original, diarization([]))
        self.assert_preserved(original, output)
        self.assertEqual(output[0].speaker_id, "说话人1")

    def test_empty_and_unaligned_text(self) -> None:
        self.assertEqual(assign_one(ASRSegmentResult("", 0, 1), diarization([])), [])
        for text in ("! \n", "No alignment."):
            original = ASRSegmentResult(text, 0, 1)
            output = assign_one(original, diarization([(0, 1, "speaker-1")]))
            self.assertEqual(output[0].text, text)
            self.assertEqual(output[0].speaker_id, "speaker-1")

    def test_alignment_mismatch_keeps_text_and_assigns_whole_block(self) -> None:
        original = transcript("Keep every word.", [(0, 1), (1, 2), (2, 3)])
        original.word_tokens[1].text = "different"
        output = assign_one(original, diarization([(0, 5, "speaker-1")]))
        self.assert_preserved(original, output)
        self.assertEqual(output[0].speaker_id, "speaker-1")

    def test_tail_and_uncertainty_across_compute_chunks_do_not_split_paragraph(
        self,
    ) -> None:
        source = [
            transcript("定价", [(0, 1), (1, 60)], end=60),
            transcript("权，继续", [(0, 0.5), (0.5, 1), (1, 35)], start=60, end=95),
        ]
        activity = diarization([(0, 59.9, "speaker-2"), (61, 95, "speaker-2")], 95)
        output = consolidate_speaker_turns(assign_speakers(source, activity))
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].text, "定价权，继续")
        self.assertEqual(output[0].speaker_id, "speaker-2")
        self.assertEqual(output[0].end_time, 95)
        before = [
            (w.text, r.start_time + w.start_time, r.start_time + w.end_time)
            for r in source
            for w in r.word_tokens
        ]
        after = [
            (w.text, r.start_time + w.start_time, r.start_time + w.end_time)
            for r in output
            for w in r.word_tokens
        ]
        self.assertEqual(before, after)
        self.assertEqual(len(activity.segments), 2)

    def test_ambiguous_chunk_opening_carries_recording_speaker(self) -> None:
        source = [
            transcript("前", [(0, 1)], end=60),
            transcript("尾新", [(0, 1), (30, 31)], start=60, end=95),
        ]
        activity = diarization([(0, 1, "A"), (90, 95, "B")], 95)
        result = consolidate_speaker_turns(assign_speakers(source, activity))
        self.assertEqual(
            [(r.text, r.speaker_id) for r in result], [("前尾", "A"), ("新", "B")]
        )

    def test_uncovered_opening_uses_nearest_speaker(self) -> None:
        original = transcript("开始", [(0, 0.5), (0.5, 1)])
        output = assign_one(original, diarization([(2, 4, "speaker-2")]))
        self.assertEqual(output[0].speaker_id, "speaker-2")
        self.assert_preserved(original, output)

    def test_zero_duration_word_at_boundary_and_recording_end(self) -> None:
        original = transcript(
            "Before. After. End.", [(0.5, 0.5), (1, 1), (2, 2)], end=2
        )
        output = assign_one(
            original, diarization([(0, 1, "speaker-1"), (1, 2, "speaker-2")], 2)
        )
        self.assert_preserved(original, output)
        self.assertEqual(
            [segment.speaker_id for segment in output], ["speaker-1", "speaker-2"]
        )

    def test_recording_labels_are_preserved_across_asr_chunks(self) -> None:
        speakers = diarization(
            [(0, 1, "speaker-1"), (1, 2, "speaker-2"), (90, 91, "speaker-1")], 91
        )
        early = transcript("Early.", [(0, 1)], end=1)
        late = transcript("Returned.", [(0, 1)], start=90, end=91)
        self.assertEqual(assign_one(early, speakers)[0].speaker_id, "speaker-1")
        self.assertEqual(assign_one(late, speakers)[0].speaker_id, "speaker-1")

    def test_invalid_timestamps_raise(self) -> None:
        for lower, upper in ((0, 6), (-1, 1), (2, 1), (float("nan"), 1)):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                assign_one(transcript("Invalid.", [(lower, upper)]), diarization([]))


def turn(text: str, start: float, end: float, speaker: str | None) -> ASRSegmentResult:
    return ASRSegmentResult(
        text, start, end, speaker, [WordToken(text.strip(), 0, end - start)]
    )


class SpeakerTurnConsolidationTest(unittest.TestCase):
    def assert_preserved(
        self, source: list[ASRSegmentResult], output: list[ASRSegmentResult]
    ) -> None:
        self.assertEqual(
            "".join(s.text for s in source), "".join(s.text for s in output)
        )
        before = [
            (w.text, s.start_time + w.start_time, s.start_time + w.end_time)
            for s in source
            for w in s.word_tokens or []
        ]
        after = [
            (w.text, s.start_time + w.start_time, s.start_time + w.end_time)
            for s in output
            for w in s.word_tokens or []
        ]
        self.assertEqual([w[0] for w in before], [w[0] for w in after])
        np.testing.assert_allclose([w[1:] for w in before], [w[1:] for w in after])

    def test_one_second_interjection_with_pauses_merges_into_main_speaker(self) -> None:
        source = [
            turn("Before. ", 0, 3, "A"),
            turn("Yes. ", 3.1, 3.9, "B"),
            turn("Mixed. ", 4.79, 5.11, None),
            turn("Continue.", 5.27, 8.27, "A"),
        ]
        result = consolidate_speaker_turns(source)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speaker_id, "A")
        self.assert_preserved(source, result)

    def test_short_main_turns_still_absorb_brief_interjection(self) -> None:
        source = [
            turn("Before. ", 0, 1, "A"),
            turn("Yes. ", 1, 1.5, "B"),
            turn("Continue.", 1.5, 2.5, "A"),
        ]
        result = consolidate_speaker_turns(source)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speaker_id, "A")
        self.assert_preserved(source, result)

    def test_same_speaker_paragraph_has_no_duration_limit(self) -> None:
        source = [
            turn("First. ", 841, 845, "A"),
            turn("And. ", 846.2, 846.5, "A"),
            turn("Next. ", 847.6, 849, "A"),
            turn("Continue.", 851, 855, "A"),
            turn("New paragraph.", 870, 872, "A"),
        ]
        result = consolidate_speaker_turns(source)
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].start_time, result[0].end_time), (841, 872))
        self.assert_preserved(source, result)

    def test_adjacent_main_speaker_groups_merge_and_rebase_words(self) -> None:
        source = [turn("First. ", 10, 12, "A"), turn("Next!", 13, 15, "A")]
        result = consolidate_speaker_turns(source)
        self.assert_preserved(source, result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].word_tokens[1].start_time, 3)
        self.assertEqual(result[0].word_tokens[1].end_time, 5)
        self.assertEqual(source[0].text, "First. ")

    def test_brief_multiple_speaker_and_unknown_run_inherits_main_speaker(self) -> None:
        source = [
            turn("First. ", 10, 12, "A"),
            turn("Yes! ", 12.2, 12.7, "B"),
            turn("Mixed. ", 12.8, 13, None),
            turn("Right. ", 13.1, 13.3, "C"),
            turn("Continue.", 13.4, 15.4, "A"),
        ]
        source[2].speaker_candidates = ["A", "B"]
        result = consolidate_speaker_turns(source)
        self.assert_preserved(source, result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speaker_id, "A")
        self.assertIsNone(result[0].speaker_candidates)
        self.assertEqual(source[1].speaker_id, "B")
        self.assertEqual(source[2].speaker_candidates, ["A", "B"])

    def test_independent_sustained_or_distant_turns_remain_separate(self) -> None:
        a = turn("Main. ", 0, 3, "A")
        short = turn("Brief. ", 3, 3.5, "B")
        cases = [
            [short, turn("Return.", 3.5, 6.5, "A")],
            [a, short],
            [a, turn("Sustained. ", 3, 5, "B"), turn("Return.", 5, 8, "A")],
            [a, short, turn("Different.", 3.5, 6.5, "C")],
            [a, short, turn("Delayed.", 4.51, 7.51, "A")],
            [a, turn("Delayed. ", 4.01, 4.51, "B"), turn("Return.", 4.51, 7.51, "A")],
            [
                a,
                turn("First. ", 3, 4, "B"),
                turn("Second. ", 4, 5, None),
                turn("Return.", 5, 8, "A"),
            ],
        ]
        for source in cases:
            with self.subTest(
                timeline=[(s.start_time, s.end_time, s.speaker_id) for s in source]
            ):
                result = consolidate_speaker_turns(source)
                self.assertEqual(result, source)
                self.assert_preserved(source, result)

    def test_same_speaker_limit_does_not_block_short_interruption_suppression(
        self,
    ) -> None:
        plain = [turn("First. ", 0, 20, "A"), turn("Next.", 20, 40, "A")]
        self.assertEqual(len(consolidate_speaker_turns(plain)), 1)
        source = [
            turn("Long. ", 0, 39, "A"),
            turn("Brief. ", 39, 39.5, "B"),
            turn("Return.", 39.5, 42, "A"),
        ]
        result = consolidate_speaker_turns(source)
        self.assertEqual(len(result), 1)
        self.assert_preserved(source, result)

    def test_zero_duration_and_repeated_interruptions_preserve_all_words(self) -> None:
        source = [
            turn("Main. ", 0, 3, "A"),
            turn("Point. ", 3, 3, None),
            turn("Return. ", 3, 4, "A"),
            turn("Brief. ", 4, 4.5, "B"),
            turn("Continue.", 4.5, 5.5, "A"),
        ]
        result = consolidate_speaker_turns(source)
        self.assertEqual(len(result), 1)
        self.assert_preserved(source, result)

    def test_sparse_aligned_words_do_not_prevent_paragraph_consolidation(self) -> None:
        source = [
            turn("Sparse. ", 0, 10, "A"),
            turn("Brief. ", 10, 10.5, "B"),
            turn("Sparse.", 10.5, 20, "A"),
        ]
        source[0].word_tokens[0].end_time = 1
        source[2].word_tokens[0].end_time = 1
        result = consolidate_speaker_turns(source)
        self.assertEqual(len(result), 1)
        self.assert_preserved(source, result)
        self.assertEqual(consolidate_speaker_turns([]), [])


if __name__ == "__main__":
    unittest.main()
