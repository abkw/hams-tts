"""Download the base datasets for Phase 2 (PLAN.md) and cache them locally
under data/raw/ so later steps (manifest building, phonemization) don't
re-hit the Hub every run.

Dataset choices and why (see PLAN.md Phase 2 / design doc for full writeup):

- MBZUAI/ClArTTS: official ClArTTS mirror (parquet, no legacy loading
  script). Diacritized MSA transcripts already included — this is our
  cleanest Arabic text-audio alignment signal.
- tunis-ai/arabic_speech_corpus: parquet mirror of Nawar Halabi's Arabic
  Speech Corpus (Syrian-accented MSA, phonetically annotated) — closest
  public match to Levantine realization. Transcripts are Buckwalter
  transliteration, not Arabic script; `prepare_manifests.py` converts them.
- bookbot/ljspeech_phonemes: parquet mirror of LJSpeech (single-speaker
  English baseline). We use its `normalized_text` column and run audio
  through our own phonemization pipeline rather than its bundled phonemes,
  so English and Arabic go through one consistent pipeline.

Legacy script-based dataset repos (keithito/lj_speech, halabi2016/...) were
deliberately avoided — `datasets` is deprecating arbitrary loading-script
execution, and the parquet mirrors above give the same content without that
dependency.
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    "clartts": "MBZUAI/ClArTTS",
    "asc": "tunis-ai/arabic_speech_corpus",
    "ljspeech": "bookbot/ljspeech_phonemes",
}


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for local_name, hub_id in DATASETS.items():
        out_dir = DATA_RAW / local_name
        if out_dir.exists():
            print(f"[skip] {local_name} already present at {out_dir}")
            continue
        print(f"[download] {hub_id} -> {out_dir}")
        ds = load_dataset(hub_id)
        ds.save_to_disk(str(out_dir))
        print(f"[done] {local_name}: {ds}")


if __name__ == "__main__":
    main()
