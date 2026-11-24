"""Automatic diacritization (tashkeel restoration) for undiacritized Arabic.

Wraps `mishkal`'s rule/statistics-based diacritizer. Mishkal was chosen over
CAMeL Tools for this project: CAMeL Tools' diacritization models are
downloaded separately via its `camel_data` fetcher (a dependency on an
external data host we can't guarantee is reachable from every environment
this project runs in), whereas mishkal ships as a self-contained pip
package with its data bundled in the wheel — lower operational risk for a
pipeline that needs to run unattended during training data prep and at
inference time. See PLAN.md Phase 1 for the full tooling trade-off.

Mishkal is MSA-oriented and rule/dictionary-based, so it will sometimes
mis-vocalize dialect-only words that aren't in its lexicon — this is a
known, documented limitation. The `levantine_rules` module's lexicon
override table is the escape hatch for words that come out wrong.
"""

from __future__ import annotations

import re

_TASHKEEL_RE = re.compile(r"[ً-ْٰ]")

_tashkeel_instance = None


def _get_tashkeel():
    global _tashkeel_instance
    if _tashkeel_instance is None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            from mishkal.tashkeel import TashkeelClass

        _tashkeel_instance = TashkeelClass()
    return _tashkeel_instance


def is_diacritized(text: str, threshold: float = 0.15) -> bool:
    """Heuristic: already-diacritized text has tashkeel marks on a decent
    fraction of its Arabic letters. Below `threshold`, treat as undiacritized."""
    letters = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    if letters == 0:
        return True
    marks = len(_TASHKEEL_RE.findall(text))
    return (marks / letters) >= threshold


def diacritize(text: str) -> str:
    """Restore tashkeel on undiacritized Arabic text. Pass-through if the
    text already looks diacritized (avoids mishkal re-vocalizing and
    potentially corrupting input that's already correct)."""
    if not text.strip():
        return text
    if is_diacritized(text):
        return text

    tashkeel = _get_tashkeel()
    try:
        return tashkeel.tashkeel_output_text(text)
    except Exception:
        # Diacritization is a best-effort enhancement, not a hard
        # dependency — fall back to the bare input rather than failing
        # the whole pipeline on a mishkal edge case.
        return text
