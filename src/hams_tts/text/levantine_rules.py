"""Levantine phonological overrides layered on top of espeak-ng's MSA output.

espeak-ng's `ar` voice is trained/rule-based for Modern Standard Arabic. Two
systematic consonant realizations differ in (urban) Levantine speech and are
corrected here at the phoneme-string level, per the project brief:

  - ق (qaf)  : MSA /q/            -> Levantine glottal stop /ʔ/
  - ج (jeem) : MSA affricate /dʒ/ -> Levantine fricative /ʒ/

These are ordered, whole-symbol string substitutions applied to espeak-ng's
IPA output, not a phonological rule engine — simple, auditable, and matches
how the brief frames the task ("tooling is your choice ... document
methodology and known limitations").

Vowel reduction and dialect-specific lexicon are handled by a second,
growable mechanism: `LEXICON_OVERRIDES` maps known Levantine words (keyed by
their undiacritized surface form) directly to a hand-specified phoneme
string, bypassing espeak-ng entirely for words that differ from MSA
*lexically* (not just phonologically) — e.g. شو ("what", Levantine) has no
relation to MSA ماذا, so no phoneme substitution rule could ever produce it.

This table starts small and is expected to grow from Phase 4 evaluation
failures (see PLAN.md Phase 3 exit criteria) — append, don't rewrite.
"""

from __future__ import annotations

import re

# Longest-match-first isn't needed here since "dʒ" and "q" don't overlap,
# but keep substitutions as an ordered list in case a future addition does.
_PHONEME_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("dʒ", "ʒ"),  # ج: affricate -> fricative
    ("q", "ʔ"),  # ق: uvular stop -> glottal stop
]

# Surface form (tashkeel stripped) -> hand-specified espeak-IPA-alphabet
# phoneme string, post Levantine substitution. Pronunciations here are
# intentionally written using the SAME symbol inventory espeak-ng's `ar`
# voice produces (after the substitutions above), so they merge into one
# phoneme vocabulary with everything else instead of introducing a second,
# incompatible transcription convention.
LEXICON_OVERRIDES: dict[str, str] = {
    "هاد": "haːd",
    "هاي": "haːj",
    "هدول": "hadoːl",
    "شو": "ʃuː",
    "ليش": "leːʃ",
    "وين": "weːn",
    "كيفك": "keːfak",
    "كمان": "kamaːn",
    "هلق": "halaʔ",
    "مش": "muʃ",
    "بدي": "biddi",
    "فيني": "fiːni",
    "منيح": "mniːħ",
    "كتير": "ktiːr",
    "شوي": "ʃwaj",
    "هسا": "hassaʔ",
    "ع": "ʕa",
    "بس": "bass",
    # Added from round-2 listening feedback: mishkal (MSA-oriented) was
    # diacritizing each of these as a different, unrelated MSA word
    # entirely rather than the intended dialect word/usage, e.g. دialect
    # "عم" (progressive-aspect marker, "is doing") -> mishkal's "عَمٌّ"
    # ("uncle" + nunation). Confirmed via
    # `hams_tts.text.diacritize.diacritize()` producing the wrong word
    # before adding these entries — not a hypothesis.
    "عم": "ʕam",
    "روح": "ruːħ",  # dialect "go" (as in بدي روح = "I want to go") — NOT
    # the MSA noun "soul"; this table is exact-surface-form, so this
    # entry assumes the more common casual-speech reading, a known
    # simplification (see design_doc.md §5, no homograph disambiguation)
    "رايح": "raːjiħ",
    "جاي": "ʒaːj",
    "يلا": "jalla",
    "خلص": "χalas̪",  # ص uses espeak's dental-bridge diacritic (U+032A)
    # here, not the pharyngealization mark ˤ — espeak is inconsistent
    # across emphatic consonants (ض gets ˤ, ص/ط get ̪), confirmed by
    # testing espeak's own output for this word directly rather than
    # guessing, so this entry matches what the model actually learned
    # from real ص instances in training data instead of introducing an
    # untrained symbol combination.
    "لسا": "lissa",
    "عالسوق": "ʕassuːʔ",  # dialect contraction ع+الـ+سوق ("to the market")
}


def strip_tashkeel(text: str) -> str:
    return re.sub(r"[ً-ْٰ]", "", text)


def lookup_lexicon(word: str) -> str | None:
    """Return a hand-specified phoneme string for a known Levantine word, if any."""
    key = strip_tashkeel(word).strip()
    return LEXICON_OVERRIDES.get(key)


_CLEANUP_RE = re.compile(r"[.ˈˌ]")  # syllable-boundary dot, stress marks
_LENGTH_COLLAPSE_RE = re.compile(r"ː+")


def apply_levantine_phonology(phoneme_str: str) -> str:
    """Apply MSA->Levantine substitutions plus generic espeak-output cleanup."""
    for src, dst in _PHONEME_SUBSTITUTIONS:
        phoneme_str = phoneme_str.replace(src, dst)
    phoneme_str = _CLEANUP_RE.sub("", phoneme_str)
    phoneme_str = _LENGTH_COLLAPSE_RE.sub("ː", phoneme_str)
    return phoneme_str
