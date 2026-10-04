"""Associate aligned words with independent diarization without duplicating text."""

from __future__ import annotations

import math
import unicodedata
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import replace

from app.utils.speaker_diarizer import DiarizationResult

from .engines.base import ASRSegmentResult, WordToken

MIN_SPEAKER_COVERAGE = 0.5
MATERIAL_SPEAKER_COVERAGE = 0.2
MAX_TURN_GAP_SECONDS = 1.0
MAX_INTERJECTION_SECONDS = 2.0
# Past this span a paragraph breaks at the next sentence end; at twice it, anywhere.
MAX_PARAGRAPH_SECONDS = 60.0
SENTENCE_ENDS = ("。", "！", "？", ".", "!", "?")


def _paragraph_full(
    first: ASRSegmentResult, last: ASRSegmentResult, following: ASRSegmentResult
) -> bool:
    span = following.end_time - first.start_time
    return span > 2 * MAX_PARAGRAPH_SECONDS or (
        span > MAX_PARAGRAPH_SECONDS and last.text.rstrip().endswith(SENTENCE_ENDS)
    )


def _join_turns(segments: Sequence[ASRSegmentResult]) -> ASRSegmentResult:
    first = segments[0]
    words = [
        replace(
            word,
            start_time=word.start_time + segment.start_time - first.start_time,
            end_time=word.end_time + segment.start_time - first.start_time,
        )
        for segment in segments
        for word in segment.word_tokens or []
    ]
    return replace(
        first,
        text="".join(segment.text for segment in segments),
        end_time=segments[-1].end_time,
        word_tokens=words or None,
    )


def _merge_same_speaker_turns(
    segments: Sequence[ASRSegmentResult],
) -> list[ASRSegmentResult]:
    turns = []
    group = []
    for segment in segments:
        if group and (
            segment.speaker_id != group[-1].speaker_id
            or _paragraph_full(group[0], group[-1], segment)
        ):
            turns.append(_join_turns(group))
            group = []
        group.append(segment)
    if group:
        turns.append(_join_turns(group))
    return turns


def consolidate_speaker_turns(
    segments: Sequence[ASRSegmentResult],
) -> list[ASRSegmentResult]:
    """Produce main-speaker paragraphs without changing raw diarization evidence.

    Brief interruptions inherit the surrounding main speaker for paragraph
    presentation only. Every text character and aligned word instant survives.
    Count interruption spans without pauses between them; short main turns
    qualify too. Long paragraphs break at sentence ends (see _paragraph_full). Thresholds use the final, sample-rate-scaled recording timeline.
    """
    turns = _merge_same_speaker_turns(segments)

    output: list[ASRSegmentResult] = []
    index = 0
    while index < len(turns):
        if output and output[-1].speaker_id is not None:
            main = output[-1]
            absorbed = False
            end = index
            previous = main
            interruption_seconds = 0.0
            while end < len(turns):
                candidate = turns[end]
                if (
                    candidate.start_time - previous.end_time > MAX_TURN_GAP_SECONDS
                    or _paragraph_full(main, previous, candidate)
                ):
                    break
                if candidate.speaker_id == main.speaker_id:
                    if end > index:
                        output[-1] = _join_turns([main, *turns[index : end + 1]])
                        index = end + 1
                        absorbed = True
                    break
                interruption_seconds += (
                    sum(word.end_time - word.start_time for word in candidate.word_tokens)
                    if candidate.word_tokens
                    else candidate.end_time - candidate.start_time
                )
                if interruption_seconds >= MAX_INTERJECTION_SECONDS:
                    break
                previous = candidate
                end += 1
            if absorbed:
                continue
        output.append(turns[index])
        index += 1
    return _merge_same_speaker_turns(output)


def _text_positions(text: str) -> list[int]:
    # Match the forced aligner's punctuation removal, including contractions.
    return [
        index
        for index, char in enumerate(text)
        if char == "'" or unicodedata.category(char).startswith(("L", "N"))
    ]


def assign_speakers(
    results: Sequence[ASRSegmentResult], diarization: DiarizationResult
) -> list[ASRSegmentResult]:
    """Assign one recording in order using one activity index and existing words.

    Clear changes switch speakers; ambiguity retains the current speaker.
    Without activity, use the nearest known speaker or a default display label.
    Raw activity remains separate: complete labels are estimates, not certainty.
    """
    intervals: dict[str, list[tuple[float, float]]] = {}
    ordered = sorted(diarization.segments, key=lambda item: item.start_sec)
    for segment in ordered:
        ranges = intervals.setdefault(segment.speaker_id, [])
        if ranges and segment.start_sec <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], segment.end_sec))
        else:
            ranges.append((segment.start_sec, segment.end_sec))
    interval_ends = {speaker: [end for _, end in spans] for speaker, spans in intervals.items()}
    current_speaker = None

    def attribute(start: float, end: float) -> str:
        nonlocal current_speaker
        if start == end:
            point = min(start, math.nextafter(diarization.duration, -math.inf))
            start, end = point, math.nextafter(point, math.inf)
        coverage = {}
        for speaker, spans in intervals.items():
            total = 0.0
            index = bisect_right(interval_ends[speaker], start)
            while index < len(spans) and spans[index][0] < end:
                lower, upper = spans[index]
                total += max(0.0, min(end, upper) - max(start, lower))
                index += 1
            if total > 0:
                coverage[speaker] = min(1.0, total / (end - start))
        candidates = [
            speaker
            for speaker, fraction in coverage.items()
            if fraction + 1e-9 >= MATERIAL_SPEAKER_COVERAGE
        ]
        if len(candidates) == 1 and coverage[candidates[0]] + 1e-9 >= MIN_SPEAKER_COVERAGE:
            current_speaker = candidates[0]
        elif current_speaker is None or (candidates and current_speaker not in candidates):
            if coverage:
                current_speaker = max(coverage, key=coverage.get)
            elif ordered:
                current_speaker = min(
                    ordered,
                    key=lambda span: max(span.start_sec - end, start - span.end_sec, 0),
                ).speaker_id
            else:
                current_speaker = "说话人1"
        return current_speaker

    output = []
    for result in results:
        if not result.text:
            continue
        words = result.word_tokens or []
        positions = _text_positions(result.text)
        normalized = "".join(result.text[index] for index in positions)
        units = ["".join(word.text[i] for i in _text_positions(word.text)) for word in words]
        if not words or not all(units) or "".join(units) != normalized:
            # Keep an unaligned ASR block intact rather than inventing text cuts.
            output.append(
                replace(
                    result,
                    speaker_id=attribute(result.start_time, result.end_time),
                    speaker_candidates=None,
                )
            )
            continue
        unit_offset = text_offset = 0
        groups = []
        for index, (word, unit) in enumerate(zip(words, units)):
            if not (
                math.isfinite(word.start_time)
                and math.isfinite(word.end_time)
                and 0
                <= word.start_time
                <= word.end_time
                <= result.end_time - result.start_time + 1e-6
            ):
                raise ValueError("Aligned word timestamps are outside their ASR segment")
            start = result.start_time + word.start_time
            end = result.start_time + word.end_time
            speaker = attribute(start, end)
            unit_offset += len(unit)
            text_end = positions[unit_offset] if index + 1 < len(words) else len(result.text)
            piece = result.text[text_offset:text_end]
            text_offset = text_end
            if groups and groups[-1].speaker_id == speaker:
                group = groups[-1]
                group.text += piece
                group.end_time = max(group.end_time, end)
            else:
                group = ASRSegmentResult(
                    text=piece,
                    start_time=start,
                    end_time=end,
                    speaker_id=speaker,
                    word_tokens=[],
                )
                groups.append(group)
            group.word_tokens.append(
                WordToken(word.text, start - group.start_time, end - group.start_time)
            )
        output.extend(groups)
    return output
