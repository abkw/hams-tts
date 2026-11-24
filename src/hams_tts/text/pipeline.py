"""End-to-end text -> phoneme pipeline: the single entry point everything
else (training data prep, inference server, benchmark) should import.

    segment_mixed_text -> normalize_span -> diacritize (ar only)
        -> phonemize_span -> apply language-boundary markers -> join

Language conditioning: rather than adding a new embedding table to the
VITS architecture (a model-code change with real integration risk — see
PLAN.md's updated architecture notes), each span is prefixed with a
reserved single-codepoint marker from the Unicode Private Use Area before
phonemization:

    U+E000 -> start of an Arabic span
    U+E001 -> start of an English span

Because `VitsTokenizer` tokenizes one Unicode codepoint at a time (see
transformers' `tokenization_vits.py`), these markers become ordinary
tokens through the *same* embedding table as every phoneme symbol, with no
architecture change needed — the self-attention text encoder can learn to
use them as language context. A dedicated language-ID embedding table
(added at the model level, broadcast per-frame rather than as a token)
would likely condition more strongly; it's documented in PLAN.md /
design doc as a known limitation and future improvement, not implemented
here due to the integration risk of patching a pretrained checkpoint's
forward pass within this project's time budget.

No silence/pause token is inserted purely because the script changed —
only punctuation-driven pauses survive from `normalize_span`, per the
brief's "no abrupt voice or prosody breaks" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

from .diacritize import diacritize
from .g2p import phonemize_span
from .lang_segment import Span, segment_mixed_text
from .normalize import normalize_span

LANG_MARKERS = {"ar": "", "en": ""}


@dataclass(frozen=True)
class PhonemizedSpan:
    source_text: str
    lang: str
    phonemes: str


@dataclass(frozen=True)
class PhonemizationResult:
    raw_text: str
    spans: list[PhonemizedSpan]
    phoneme_string: str


def phonemize_text(text: str, default_lang: str = "ar") -> PhonemizationResult:
    spans: list[Span] = segment_mixed_text(text, default_lang=default_lang)

    phon_spans: list[PhonemizedSpan] = []
    for span in spans:
        normalized = normalize_span(span.text, span.lang)
        if not normalized:
            continue
        if span.lang == "ar":
            normalized = diacritize(normalized)
        phonemes = phonemize_span(normalized, span.lang)
        if not phonemes.strip():
            continue
        phon_spans.append(PhonemizedSpan(source_text=span.text, lang=span.lang, phonemes=phonemes))

    parts = [f"{LANG_MARKERS[s.lang]}{s.phonemes}" for s in phon_spans]
    phoneme_string = " ".join(parts)

    return PhonemizationResult(raw_text=text, spans=phon_spans, phoneme_string=phoneme_string)


def phonemize(text: str, default_lang: str = "ar") -> str:
    """Convenience wrapper returning just the final phoneme string."""
    return phonemize_text(text, default_lang=default_lang).phoneme_string
