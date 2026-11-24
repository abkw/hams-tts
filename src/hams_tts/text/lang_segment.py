"""Split mixed Arabic/Latin-script text into per-language spans.

Code-switched Levantine/English text mixes scripts mid-sentence. Numbers,
punctuation, and symbols carry no script of their own, so they're folded
into whichever language span they're adjacent to rather than treated as a
third class — that decision is what the `normalize` module (number/date/
currency expansion) relies on to pick a language for spelling out a digit
run.

Known limitation: a single Latin-script loanword embedded inside an Arabic
sentence (e.g. a brand name) is classified as an "en" span like any other
English text, even when a fluent reader would treat it as a nativized
Arabic word with Arabic-influenced pronunciation. We don't attempt
loanword detection — see PLAN.md Phase 1 exit criteria / design doc
limitations.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_LATIN_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),
)

Lang = str  # "ar" | "en"


@dataclass(frozen=True)
class Span:
    text: str
    lang: Lang


def _char_class(ch: str) -> str:
    cp = ord(ch)
    for lo, hi in _ARABIC_RANGES:
        if lo <= cp <= hi:
            return "ar"
    for lo, hi in _LATIN_RANGES:
        if lo <= cp <= hi:
            return "en"
    if ch.isdigit() or unicodedata.category(ch) == "Nd":
        return "digit"
    if ch.isspace():
        return "space"
    return "neutral"  # punctuation, symbols, emoji, etc.


def _raw_runs(text: str) -> list[tuple[str, str]]:
    """Group consecutive characters of the same class, without merging yet."""
    runs: list[tuple[str, str]] = []
    for ch in text:
        cls = _char_class(ch)
        if runs and runs[-1][1] == cls:
            runs[-1] = (runs[-1][0] + ch, cls)
        else:
            runs.append((ch, cls))
    return runs


def segment_mixed_text(text: str, default_lang: Lang = "ar") -> list[Span]:
    """Turn raw mixed-script text into a list of (text, lang) spans.

    digit/space/neutral runs attach to the nearest preceding language run
    (or `default_lang` if they appear before any language run at all), so
    every returned span has lang in {"ar", "en"} and no span is purely
    punctuation/whitespace/digits with an undefined language.
    """
    if not text:
        return []

    runs = _raw_runs(text)

    spans: list[Span] = []
    for chunk, cls in runs:
        if cls in ("ar", "en"):
            spans.append(Span(chunk, cls))
        else:
            if spans:
                spans[-1] = Span(spans[-1].text + chunk, spans[-1].lang)
            else:
                spans.append(Span(chunk, default_lang))

    # Merge adjacent spans that ended up with the same language (can happen
    # after a neutral run was glued onto the previous span but the next raw
    # run is also the same language, e.g. "abc123def" if the digit/letter
    # boundary didn't change language).
    merged: list[Span] = []
    for span in spans:
        if merged and merged[-1].lang == span.lang:
            merged[-1] = Span(merged[-1].text + span.text, span.lang)
        else:
            merged.append(span)

    return merged


_WORD_RE = re.compile(r"\S+")


def dominant_language(text: str, default_lang: Lang = "ar") -> Lang:
    """Best-effort single language label for a whole string (e.g. a manifest row)."""
    spans = segment_mixed_text(text, default_lang=default_lang)
    if not spans:
        return default_lang
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.lang] = counts.get(span.lang, 0) + len(span.text.strip())
    return max(counts, key=counts.get)
