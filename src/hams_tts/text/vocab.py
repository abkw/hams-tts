"""Builds a `VitsTokenizer`-compatible vocab.json from our own phoneme inventory.

MMS/VITS checkpoints ship a `vocab.json` mapping single Unicode codepoints
to ids, plus a `tokenizer_config.json` with `phonemize=False, normalize=True`
for MMS (raw-character input) or `phonemize=True` for the original English
VITS (espeak-ng called inside the tokenizer). Neither fits what we need:
our *own* `hams_tts.text.pipeline` already produced fully-phonemized,
Levantine-corrected, language-marked text upstream, so the tokenizer's job
here is just "one id per codepoint," with its own normalize/phonemize
hooks both switched off so it doesn't try to re-touch our output.

Id 0 is reserved as the blank/pad token VITS's `add_blank` interleaves
between every real symbol (see transformers' `VitsTokenizer._tokenize`) —
it carries no phonetic meaning, it's a structural placeholder the model
learns to treat as a transition/silence slot, same convention the
upstream MMS checkpoints use for whatever symbol happens to land at id 0.

Building the vocab empirically (scan actual pipeline output) rather than
hand-enumerating IPA symbols is deliberate: it guarantees every symbol the
pipeline can actually produce — including espeak-ng quirks we didn't
anticipate — has a slot, instead of silently mapping unknowns to `<unk>`
and losing signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocab(phoneme_strings: Iterable[str]) -> dict[str, int]:
    chars: set[str] = set()
    for s in phoneme_strings:
        chars.update(s)

    chars.discard(PAD_TOKEN)
    chars.discard(UNK_TOKEN)

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, ch in enumerate(sorted(chars), start=2):
        vocab[ch] = i
    return vocab


def save_vocab(vocab: dict[str, int], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    tokenizer_config = {
        "tokenizer_class": "VitsTokenizer",
        "add_blank": True,
        "normalize": False,
        "phonemize": False,
        "is_uroman": False,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "language": "lev-ar-en",
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")
    return vocab_path


def load_tokenizer(vocab_dir: str | Path):
    from transformers import VitsTokenizer

    return VitsTokenizer.from_pretrained(str(vocab_dir))
