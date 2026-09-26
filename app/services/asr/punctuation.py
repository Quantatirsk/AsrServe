"""Restore only a missing final mark using FunASR's standard EOF policy."""

from functools import lru_cache

from funasr import AutoModel

from app.core.config import settings
from app.infrastructure import resolve_model_path

MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
_TERMINALS = ".?!\u3002\uff1f\uff01\u2026"
_CLOSING_QUOTES = "\"'\u2019\u201d\u300d\u300f\u3011)]}"
_CONTINUATIONS = ",;:\uff0c\uff1b\uff1a\u3001"
_ENGLISH_MARKS = str.maketrans("\u3002\uff1f\uff01", ".?!")


@lru_cache(maxsize=1)
def get_punctuation_model() -> AutoModel:
    return AutoModel(
        model=resolve_model_path(MODEL_ID),
        device="cpu",
        **settings.FUNASR_AUTOMODEL_KWARGS,
    )


def _ending(text: str) -> str:
    return text.rstrip().rstrip(_CLOSING_QUOTES).rstrip()


def restore_sentence_endings(texts: list[str]) -> list[str]:
    """One model call for every text missing a final mark: per-call overhead
    (about 0.2 s) dwarfs the per-text cost (about 0.01 s)."""
    missing = [
        i
        for i, text in enumerate(texts)
        if (e := _ending(text)) and e[-1] not in _TERMINALS
    ]
    restored = list(texts)
    if not missing:
        return restored
    results = get_punctuation_model().generate(input=[texts[i] for i in missing])
    if not isinstance(results, list) or len(results) != len(missing):
        raise RuntimeError("Punctuation model returned no valid text")
    for i, result in zip(missing, results):
        restored[i] = _copy_final_mark(texts[i], result)
    return restored


def _copy_final_mark(text: str, result: object) -> str:
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("text"), str)
        or not result["text"].strip()
    ):
        raise RuntimeError("Punctuation model returned no valid text")
    ending = _ending(text)
    restored = _ending(result["text"])
    if not restored or restored[-1] not in _TERMINALS:
        return text
    # FunASR includes EOF period restoration; never copy its interior rewrites.
    mark = restored[-1]
    if ending[-1].isascii() and ending[-1].isalnum():
        mark = mark.translate(_ENGLISH_MARKS)
    suffix = text[len(ending) :]
    if ending[-1] in _CONTINUATIONS:
        ending = ending[:-1]
    return ending + mark + suffix
