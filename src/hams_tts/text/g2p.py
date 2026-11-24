"""Grapheme-to-phoneme: espeak-ng backend, word-level lexicon override for Arabic.

Why espeak-ng (+ a thin rule layer) instead of a custom/trained G2P model:
it already has mature Arabic and English voices, is fast enough to run
inline during both data prep and real-time inference (no model forward
pass needed just to get phonemes), and is the same tool the `finetune-hf-vits`
/ original VITS recipe already uses for English — reusing it keeps the
English path identical to a well-tested baseline and concentrates all of
our actual engineering effort on the Arabic/Levantine-specific layer
(diacritization + `levantine_rules`) where it's actually needed. A trained
custom G2P was rejected: it would need its own training data and
evaluation effort that buys nothing over espeak-ng for the sounds the
brief asks us to get right (ق, ج, vowel reduction, dialect lexicon) — all
of which are handled more directly and more auditably as explicit rules.
"""

from __future__ import annotations

import re

from .espeak_setup import ensure_espeak_configured
from .levantine_rules import apply_levantine_phonology, lookup_lexicon

_WORD_SPLIT_RE = re.compile(r"(\s+)")


def _phonemize_batch(words: list[str], language: str) -> list[str]:
    if not words:
        return []
    ensure_espeak_configured()
    from phonemizer import phonemize

    out = phonemize(
        words,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=False,
        njobs=1,
    )
    return out


def phonemize_arabic(text: str) -> str:
    """Word-level phonemization so `levantine_rules.LEXICON_OVERRIDES` can
    intercept individual dialect words before they ever reach espeak-ng."""
    if not text.strip():
        return ""

    tokens = _WORD_SPLIT_RE.split(text)
    words_to_phonemize: list[str] = []
    word_positions: list[int] = []
    for i, tok in enumerate(tokens):
        if tok and not tok.isspace():
            words_to_phonemize.append(tok)
            word_positions.append(i)

    overridden = {i: lookup_lexicon(w) for i, w in zip(word_positions, words_to_phonemize)}
    remaining = [w for i, w in zip(word_positions, words_to_phonemize) if overridden[i] is None]
    remaining_positions = [i for i in word_positions if overridden[i] is None]

    phonemized = _phonemize_batch(remaining, language="ar")
    phon_by_pos = dict(zip(remaining_positions, phonemized))

    out_tokens = list(tokens)
    for i in word_positions:
        if overridden[i] is not None:
            out_tokens[i] = overridden[i]
        else:
            out_tokens[i] = apply_levantine_phonology(phon_by_pos[i])

    return "".join(out_tokens)


def phonemize_english(text: str) -> str:
    if not text.strip():
        return ""
    out = _phonemize_batch([text], language="en-us")
    return out[0] if out else ""


def phonemize_span(text: str, lang: str) -> str:
    if lang == "ar":
        return phonemize_arabic(text)
    if lang == "en":
        return phonemize_english(text)
    raise ValueError(f"Unknown language tag: {lang!r}")
