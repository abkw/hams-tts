"""Build the custom phoneme vocabulary from the processed training corpus.

Must run after `prepare_manifests.py` (which writes
data/processed/phoneme_corpus.jsonl). Resizing the model's token/speaker
embeddings to match this vocab happens inside `run_vits_finetuning.py`
itself (`override_vocabulary_embeddings` / `override_speaker_embeddings`
in the training config) — this script only needs to produce the
vocab.json + tokenizer_config.json that `tokenizer_name` will point at.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hams_tts.text.vocab import build_vocab, load_tokenizer, save_vocab  # noqa: E402

PHONEME_CORPUS = REPO_ROOT / "data" / "processed" / "phoneme_corpus.jsonl"
TOKENIZER_DIR = REPO_ROOT / "data" / "processed" / "tokenizer"


def main() -> None:
    texts = []
    with PHONEME_CORPUS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])

    print(f"[vocab] building from {len(texts)} phoneme strings")
    vocab = build_vocab(texts)
    print(f"[vocab] vocab size: {len(vocab)}")
    save_vocab(vocab, TOKENIZER_DIR)

    tokenizer = load_tokenizer(TOKENIZER_DIR)
    sample = texts[0]
    ids = tokenizer(sample, return_tensors="pt")["input_ids"][0]
    assert tokenizer.decode(ids) == sample, "tokenizer round-trip failed on real corpus sample"
    print(f"[vocab] round-trip OK, saved to {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
