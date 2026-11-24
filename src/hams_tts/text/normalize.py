"""Number/date/currency/punctuation normalization, per language span.

Expands digit sequences into spoken words before phonemization — espeak-ng
can read bare digits itself, but currency and date patterns need an
explicit rule layer to come out right in both Arabic and English, and
mixed AR/EN punctuation needs to collapse to one canonical set so the
downstream phoneme-boundary logic (see `g2p.py`) has a single pause
vocabulary to key off instead of two.

Known limitation: date expansion only handles Gregorian DD/MM/YYYY and
YYYY-MM-DD style patterns. Hijri dates and spelled-out month names are not
normalized — see PLAN.md / design doc limitations.
"""

from __future__ import annotations

import re

from num2words import num2words

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_CURRENCY_SYMBOLS = {
    "$": {"ar": "دولار", "en": "dollars"},
    "USD": {"ar": "دولار", "en": "dollars"},
    "€": {"ar": "يورو", "en": "euros"},
    "EUR": {"ar": "يورو", "en": "euros"},
    "£": {"ar": "جنيه إسترليني", "en": "pounds"},
    "JD": {"ar": "دينار أردني", "en": "Jordanian dinars"},
    "SAR": {"ar": "ريال سعودي", "en": "Saudi riyals"},
    "SYP": {"ar": "ليرة سورية", "en": "Syrian pounds"},
    "ل.س": {"ar": "ليرة سورية", "en": "Syrian pounds"},
    "ل.ل": {"ar": "ليرة لبنانية", "en": "Lebanese pounds"},
    "د.أ": {"ar": "دينار أردني", "en": "Jordanian dinars"},
}

# Longest-symbol-first so "USD" doesn't get shadowed by a later partial match.
_CURRENCY_PATTERN = re.compile(
    r"(?P<amount>\d+(?:[.,]\d+)?)\s*(?P<symbol>"
    + "|".join(re.escape(sym) for sym in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True))
    + r")"
    r"|(?P<symbol2>"
    + "|".join(re.escape(sym) for sym in sorted(_CURRENCY_SYMBOLS, key=len, reverse=True))
    + r")\s*(?P<amount2>\d+(?:[.,]\d+)?)"
)

_DATE_PATTERN = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")

# Arabic vs. Latin punctuation normalized to one canonical, pause-bearing set.
_PUNCT_MAP = {
    "،": ",",
    "؛": ";",
    "؟": "?",
    "…": "...",
    "ـ": "",  # tatweel/kashida — no phonetic content
}

_PAUSE_CHARS = set(",.;:!?")


def normalize_punctuation(text: str) -> str:
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text


def _spell_number(value: str, lang: str) -> str:
    value = value.replace(",", "")
    try:
        if "." in value:
            return num2words(float(value), lang=lang)
        return num2words(int(value), lang=lang)
    except (ValueError, NotImplementedError):
        return value


def expand_currency(text: str, lang: str) -> str:
    def repl(match: re.Match) -> str:
        amount = match.group("amount") or match.group("amount2")
        symbol = match.group("symbol") or match.group("symbol2")
        words = _spell_number(amount, lang)
        currency_name = _CURRENCY_SYMBOLS[symbol][lang]
        return f"{words} {currency_name}"

    return _CURRENCY_PATTERN.sub(repl, text)


def expand_dates(text: str, lang: str) -> str:
    def repl(match: re.Match) -> str:
        d, m, y = match.groups()
        day = _spell_number(d, lang)
        month = _spell_number(m, lang)
        year = _spell_number(y, lang)
        return f"{day} {month} {year}"

    return _DATE_PATTERN.sub(repl, text)


def expand_numbers(text: str, lang: str) -> str:
    def repl(match: re.Match) -> str:
        return _spell_number(match.group(0), lang)

    return re.sub(r"\d+(?:[.,]\d+)?", repl, text)


def normalize_span(text: str, lang: str) -> str:
    """Full normalization pipeline for one (text, lang) span from `lang_segment`."""
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = normalize_punctuation(text)
    text = expand_dates(text, lang)
    text = expand_currency(text, lang)
    text = expand_numbers(text, lang)
    text = re.sub(r"\s+", " ", text).strip()
    return text
