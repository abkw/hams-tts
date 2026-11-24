"""Targeted diagnostic sample set for human pronunciation review.

Separate from the FR4 held-out eval set (`data/manifests/held_out_eval.jsonl`,
used for the required MOS-style listening test on naturalness/code-switch
smoothness) — this set isolates *specific* phonological/lexical targets
individually, so a listener's "this word sounds wrong" feedback can be
traced to a specific fix:

  - Individual words testing the two fixed phoneme substitutions
    (ق → /ʔ/, ج → /ʒ/) in isolation from everything else in a sentence.
  - Individual dialect-lexicon words from `levantine_rules.py`'s override
    table, to check whether each lexicon entry actually sounds like the
    intended word.
  - Full sentences spanning pure Arabic (short/long), pure English,
    numbers/dates, and code-switched — the same four categories as the
    FR4 set, but held separate so this set can be regenerated/extended
    quickly as specific problem words get reported, without touching the
    fixed FR4 benchmark set.

A word rated badly here points at `levantine_rules.py` (lexicon entry
missing or wrong) or `diacritize.py` (mishkal mis-vocalized it) as the
fix — no retraining needed either way, since phonemization happens at
inference time, not training time. A full sentence rated badly despite
its individual words rating fine points at something else (prosody,
duration modeling, data scale) that *would* need more/different training
data.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_DIR = REPO_ROOT / "docs" / "eval_results" / "diagnostic_samples"

# (id, category, lang_hint, text, what_this_tests)
ITEMS: list[tuple[str, str, str, str, str]] = [
    # Individual words — ق → /ʔ/
    ("w01_qalb", "word_phoneme", "ar", "قلب", "ق → /ʔ/ (qalb = heart)"),
    ("w02_qamar", "word_phoneme", "ar", "قمر", "ق → /ʔ/ (qamar = moon)"),
    ("w03_waraqa", "word_phoneme", "ar", "ورقة", "ق → /ʔ/ mid-word (waraqa = paper)"),
    # Individual words — ج → /ʒ/
    ("w04_jamil", "word_phoneme", "ar", "جميل", "ج → /ʒ/ (jamil = beautiful)"),
    ("w05_jar", "word_phoneme", "ar", "جار", "ج → /ʒ/ (jar = neighbor)"),
    ("w06_jadid", "word_phoneme", "ar", "جديد", "ج → /ʒ/ (jadid = new)"),
    # Dialect lexicon words (levantine_rules.py overrides)
    ("w07_shu", "word_lexicon", "ar", "شو", "lexicon override (shu = what)"),
    ("w08_halla", "word_lexicon", "ar", "هلق", "lexicon override (halla' = now)"),
    ("w09_biddi", "word_lexicon", "ar", "بدي", "lexicon override (biddi = I want)"),
    ("w10_ktir", "word_lexicon", "ar", "كتير", "lexicon override (ktir = a lot)"),
    ("w11_mnih", "word_lexicon", "ar", "منيح", "lexicon override (mnih = good)"),
    ("w12_mush", "word_lexicon", "ar", "مش", "lexicon override (mush = not)"),
    # Added after round-2 listening feedback surfaced these as
    # mis-diacritized by mishkal (diacritized as a different, unrelated
    # MSA word) — see the comments next to each entry in
    # levantine_rules.py's LEXICON_OVERRIDES for the diagnosis.
    ("w13_aam", "word_lexicon", "ar", "عم", "lexicon override (ʕam = progressive-aspect marker)"),
    ("w14_rouh", "word_lexicon", "ar", "روح", "lexicon override (ruːħ = dialect 'go', not MSA 'soul')"),
    ("w15_rayih", "word_lexicon", "ar", "رايح", "lexicon override (raːjiħ = going)"),
    ("w16_jay", "word_lexicon", "ar", "جاي", "lexicon override (ʒaːj = coming)"),
    ("w17_yalla", "word_lexicon", "ar", "يلا", "lexicon override (jalla = let's go/come on)"),
    ("w18_khalas", "word_lexicon", "ar", "خلص", "lexicon override (χalas̪ = done/enough)"),
    ("w19_lissa", "word_lexicon", "ar", "لسا", "lexicon override (lissa = still/not yet)"),
    ("w20_aalsouq", "word_lexicon", "ar", "عالسوق", "lexicon override (ʕassuːʔ = to the market, contraction)"),
    # Pure Arabic sentences
    ("s01_greeting", "sentence_ar", "ar", "مرحبا، كيف حالك اليوم؟", "simple greeting"),
    ("s02_dialect", "sentence_ar", "ar", "شو عم تعمل هلق؟", "dialect-heavy short sentence"),
    ("s03_longer", "sentence_ar", "ar", "بدي روح عالسوق وأشتري خضرة وفواكه", "longer sentence, multiple clauses"),
    ("s04_descriptive", "sentence_ar", "ar", "الجو كتير حلو اليوم، في شمس وما في غيم", "descriptive sentence, dialect lexicon in context"),
    ("s05_currency", "sentence_ar", "ar", "عندي 25 دولار بس", "number/currency normalization"),
    # Pure English sentences
    ("s06_en_simple", "sentence_en", "en", "Good morning, I hope you have a wonderful day.", "simple English sentence"),
    ("s07_en_request", "sentence_en", "en", "Can you please send me that report by tomorrow afternoon?", "longer English sentence"),
    # Code-switched sentences
    ("s08_cs_schedule", "code_switch", "ar", "بدي أعمل schedule لـ meeting بكرا", "code-switch, tech nouns"),
    ("s09_cs_laptop", "code_switch", "ar", "ال laptop تبعي خربان، في مشكلة بالـ password", "code-switch, two English nouns"),
    ("s10_cs_email", "code_switch", "ar", "شفت ال email من يومين بس ما كان عندي وقت", "code-switch, mid-sentence switch"),
    ("s11_cs_presentation", "code_switch", "ar", "عندي presentation بكرا، لازم أحضرها اليوم", "code-switch, English noun early"),
    ("s12_cs_date", "code_switch", "ar", "الاجتماع يوم 14 مارس الساعة 3", "code-switch + date/number normalization"),
]


def main() -> None:
    from hams_tts.inference.engine import Engine

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[diagnostic] loading engine...")
    engine = Engine(model_dir="data/processed/checkpoint", device="cuda")

    rows = []
    for item_id, category, lang_hint, text, tests in ITEMS:
        audio, sr = engine.synthesize(text, default_lang=lang_hint)
        wav_path = OUT_DIR / f"{item_id}.wav"
        sf.write(str(wav_path), audio, sr, subtype="PCM_16")
        rows.append(
            {
                "id": item_id,
                "category": category,
                "text": text,
                "tests": tests,
                "audio_path": str(wav_path.relative_to(REPO_ROOT)),
                "rating_1_10": "",
                "notes": "",
            }
        )
        print(f"[diagnostic] {item_id} ({category}): {text!r} -> {wav_path.name}")

    csv_path = OUT_DIR / "rating_sheet.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[diagnostic] wrote {len(rows)} audio files + {csv_path}")


if __name__ == "__main__":
    main()
