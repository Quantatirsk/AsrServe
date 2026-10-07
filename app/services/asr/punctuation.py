"""Restore sentence punctuation from token labels without rewriting ASR text."""

from __future__ import annotations

import re
import threading
import unicodedata
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate

from app.core.config import settings

MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
MODEL_REVISION = "v2.0.4"
MODEL_HASHES = (
    ("config.yaml", "a56ec10925b06fa976ad51af373396be2b13e1eb8dc62a5426b5adebaba7071d"),
    ("model.pt", "a5818bb9d933805a916eebe41eb41648f7f9caad30b4bd59d56f3ca135421916"),
    ("tokens.json", "c960ab87bccea4aa15cf49a59f71973c2c330b46668048cd8da253749ec71ee3"),
)
_LABELS = ("<unk>", "_", "，", "。", "？", "、")
_MARKS = frozenset(",.;:!?，。；：！？、…")
_TERMINALS = frozenset(".?!。？！…")
_ENGLISH = str.maketrans("，。？、", ",.?,")
_QUOTES = frozenset("\"'`‘’“”()[]{}「」『』【】")
_URL = re.compile(r"\b(?:[A-Za-z][A-Za-z\d+.-]*://|www\.)[^\s\x80-\uffff<>\"']+")
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_punctuation_model():
    import torch
    from funasr import AutoModel

    model = AutoModel(
        model=settings.PUNCTUATION_MODEL_PATH,
        device="cpu",
        ncpu=torch.get_num_threads(),
        trust_remote_code=False,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        local_files_only=True,
    )
    if tuple(model.model.punc_list) != _LABELS:
        raise RuntimeError("Unexpected CT-Transformer punctuation labels")
    return model


def close_punctuation_model() -> None:
    with _LOCK:
        get_punctuation_model.cache_clear()


@dataclass(frozen=True)
class _Unit:
    text: str
    start: int
    end: int
    segment: int


def _is_mark(text: str, index: int) -> bool:
    char = text[index]
    if char not in _MARKS:
        return False
    before = text[index - 1] if index else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    # Decimal points, digit grouping, times, versions and dotted names are text.
    if char in ",，:：" and before.isdigit() and after.isdigit():
        return False
    if char == "." and before.isalnum() and after.isalnum():
        return False
    return True


def _units(text: str, segment: int, offset: int) -> list[_Unit]:
    protected = set()
    for match in _URL.finditer(text):
        end = match.start() + len(match.group().rstrip(".,;!?"))
        protected.update(range(match.start(), end))
    units = []
    start = None

    def flush(end: int) -> None:
        nonlocal start
        if start is not None:
            word = text[start:end]
            if any(c.isalnum() for c in word):
                units.append(_Unit(word, offset + start, offset + end, segment))
            start = None

    for index, char in enumerate(text):
        contraction = (
            char == "'"
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isalpha()
            and text[index + 1].isalpha()
        )
        if char.isspace() or (
            index not in protected
            and (_is_mark(text, index) or (char in _QUOTES and not contraction))
        ):
            flush(index)
        elif char.isascii():
            if start is None:
                start = index
        else:
            flush(index)
            # Match FunASR split_words: non-ASCII characters are separate tokens.
            if unicodedata.category(char).startswith(("L", "N", "M")):
                units.append(_Unit(char, offset + index, offset + index + 1, segment))
    flush(len(text))
    return units


def _han(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff" or "\U00020000" <= char <= "\U000323af"


def _can_insert(text: str, index: int) -> bool:
    if index >= len(text):
        return True
    before, after = text[index - 1], text[index]
    if unicodedata.category(after).startswith("M"):
        return False
    # Do not split accented/non-Latin words or typographic contractions.
    if before.isalnum() and after.isalnum() and not (_han(before) or _han(after)):
        return False
    if after in "'’" and before.isalpha() and text[index + 1 : index + 2].isalpha():
        return False
    if before in "'’" and after.isalpha() and text[index - 2 : index - 1].isalpha():
        return False
    if (
        after in ".,，:："
        and before.isdigit()
        and text[index + 1 : index + 2].isdigit()
    ):
        return False
    return True


def restore_punctuation(texts: Sequence[str]) -> list[str]:
    """Predict one file with shared context, then scatter marks to original chunks.

    Existing terminal/structural marks survive. A predicted sentence ending can
    replace a comma; all other original characters, including whitespace, stay
    at their original positions. Never use FunASR's reformatted output text.
    """
    restored = list(texts)
    # A separator keeps periods at chunk edges distinct from dotted names.
    starts = list(accumulate([0, *(len(text) + 1 for text in texts)]))
    units = [
        unit for i, text in enumerate(texts) for unit in _units(text, i, starts[i])
    ]
    if not units:
        return restored
    with _LOCK:
        results = get_punctuation_model().generate(
            input=" ".join(unit.text for unit in units)
        )
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        raise RuntimeError("Punctuation model returned no token labels")
    labels = results[0].get("punc_array")
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    if (
        not isinstance(labels, list)
        or len(labels) != len(units)
        or any(
            type(label) is not int or not 0 <= label < len(_LABELS) for label in labels
        )
    ):
        raise RuntimeError("Punctuation labels do not match the input tokens")

    source = "\n".join(texts)
    edits: list[dict[int, str]] = [{} for _ in texts]
    for i, (unit, label) in enumerate(zip(units, labels, strict=True)):
        mark = _LABELS[label] if label > 1 else ""
        # FunASR's punc_array omits its text-only EOF period normalization.
        if i == len(units) - 1 and mark not in _TERMINALS:
            mark = "。"
        if not mark:
            continue
        following = units[i + 1] if i + 1 < len(units) else None
        end = following.start if following else len(source)
        existing = [p for p in range(unit.end, end) if _is_mark(source, p)]
        if any(source[p] in _TERMINALS or source[p] in ";；:：" for p in existing):
            continue
        if unicodedata.east_asian_width(unit.text[-1]) not in ("W", "F"):
            mark = mark.translate(_ENGLISH)
        if existing:
            if mark in _TERMINALS:
                pos = existing[0]
                owner = bisect_right(starts[1:], pos)
                edits[owner][pos - starts[owner]] = mark
            continue
        if (
            following
            and following.segment == unit.segment
            and not _can_insert(source, unit.end)
        ):
            continue
        pos = unit.end - starts[unit.segment]
        edits[unit.segment][pos] = mark

    for owner, changes in enumerate(edits):
        text = texts[owner]
        for pos, mark in sorted(changes.items(), reverse=True):
            # Positions inside original sentence punctuation replace one mark.
            replace = pos < len(text) and _is_mark(text, pos)
            text = text[:pos] + mark + text[pos + int(replace) :]
        restored[owner] = text
    return restored
