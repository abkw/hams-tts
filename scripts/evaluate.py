"""FR4 evaluation: held-out set -> ASR round-trip CER/WER + reference-free
quality metric (DNSMOS). Complements `benchmark.py` (which covers FR5's
TTFA/RTF/peak-VRAM instead) — together they cover every *objective* metric
FR4/FR5 ask for. The MOS-style listening test FR4 also asks for needs real
human listeners (n >= 5) and can't be run by this script; `main()` instead
writes a listener-scoring CSV template (utterance id/text/audio path, blank
columns for naturalness/code-switch-smoothness ratings + free-text notes)
so that test is one step — handing the template + the sample WAVs to
listeners — away from being run, not a missing deliverable.

ASR round-trip: faster-whisper (large-v3) transcribes each synthesized
utterance; the transcript is compared against the *input* text (not a
separate reference recording — there is no ground-truth audio for
code-switched utterances, see PLAN.md Phase 2). CER is used for Arabic-
majority utterances, WER for English-majority ones, per FR4's framing
("CER/WER" is written as a pair precisely because Arabic is usually scored
by character and English by word).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

HELD_OUT_EVAL = REPO_ROOT / "data" / "manifests" / "held_out_eval.jsonl"


@dataclass
class EvalResult:
    utterance_id: str
    category: str
    input_text: str
    asr_transcript: str
    error_rate: float
    error_rate_type: str  # "CER" or "WER"
    dnsmos_ovrl: float
    audio_path: str


def load_eval_utterances() -> list[dict]:
    rows = []
    with HELD_OUT_EVAL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_ARABIC_DIACRITICS_RE = re.compile(r"[ً-ْٰۖ-ۭ]")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_for_comparison(text: str) -> str:
    """Strip diacritics/punctuation and lowercase so ASR transcription
    noise (casing, punctuation choices) doesn't inflate error rates for
    things that aren't actually mispronunciations."""
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def compute_error_rate(reference: str, hypothesis: str, category: str) -> tuple[float, str]:
    import jiwer

    ref = _normalize_for_comparison(reference)
    hyp = _normalize_for_comparison(hypothesis)

    if category == "en":
        return jiwer.wer(ref, hyp), "WER"
    # Arabic and code-switched utterances are scored by character — a
    # code-switched utterance's English span would be unfairly penalized
    # at the word level by Whisper's own Arabic-biased transcription of
    # mixed-script audio; character-level is more forgiving of that.
    return jiwer.cer(ref, hyp), "CER"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="data/processed/checkpoint")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-device", default="cuda")
    parser.add_argument("--whisper-compute-type", default="float16")
    parser.add_argument("--out-dir", default="docs/eval_results")
    args = parser.parse_args()

    from faster_whisper import WhisperModel
    from speechmos import dnsmos

    from hams_tts.inference.engine import Engine

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("[evaluate] loading TTS engine...")
    engine = Engine(model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir, device=args.device)

    print(f"[evaluate] loading faster-whisper {args.whisper_model}...")
    asr = WhisperModel(args.whisper_model, device=args.whisper_device, compute_type=args.whisper_compute_type)

    results: list[EvalResult] = []
    for row in load_eval_utterances():
        lang_hint = "en" if row["category"] == "en" else "ar"
        audio, sr = engine.synthesize(row["text"], default_lang=lang_hint)

        wav_path = audio_dir / f"{row['id']}.wav"
        sf.write(str(wav_path), audio, sr, subtype="PCM_16")

        asr_language = "en" if row["category"] == "en" else "ar"
        # Pass the decoded array directly rather than the wav path — the
        # installed faster-whisper's file-path code path goes through
        # PyAV's `av.open(..., metadata_errors=...)`, which the installed
        # PyAV version doesn't accept (another transformers/PyAV-style
        # version-skew mismatch); the array path skips that entirely and
        # avoids a redundant decode of audio we already have in memory.
        segments, _ = asr.transcribe(audio.astype(np.float32), language=asr_language)
        transcript = " ".join(seg.text.strip() for seg in segments)

        error_rate, error_type = compute_error_rate(row["text"], transcript, row["category"])

        mos = dnsmos.run(audio.astype(np.float64), sr=sr)

        results.append(
            EvalResult(
                utterance_id=row["id"],
                category=row["category"],
                input_text=row["text"],
                asr_transcript=transcript,
                error_rate=error_rate,
                error_rate_type=error_type,
                dnsmos_ovrl=float(mos["ovrl_mos"]),
                audio_path=str(wav_path.relative_to(REPO_ROOT)),
            )
        )
        print(f"[evaluate] {row['id']} ({row['category']}): {error_type}={error_rate:.3f} DNSMOS={mos['ovrl_mos']:.2f}")

    _write_markdown_summary(results, out_dir / "eval_results.md")
    _write_listener_template(results, out_dir / "listener_scoring_template.csv")
    print(f"\n[evaluate] wrote {out_dir / 'eval_results.md'} and listener_scoring_template.csv")


def _write_markdown_summary(results: list[EvalResult], out_path: Path) -> None:
    import statistics

    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    lines = ["# FR4 Evaluation Results", "", "| Category | n | Mean error rate | Mean DNSMOS (OVRL) |", "|---|---|---|---|"]
    for cat, rows in by_category.items():
        err_type = rows[0].error_rate_type
        mean_err = statistics.mean(r.error_rate for r in rows)
        mean_mos = statistics.mean(r.dnsmos_ovrl for r in rows)
        lines.append(f"| {cat} | {len(rows)} | {mean_err:.3f} {err_type} | {mean_mos:.2f} |")

    lines += ["", "| id | category | error rate | DNSMOS | transcript |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.utterance_id} | {r.category} | {r.error_rate:.3f} {r.error_rate_type} | "
            f"{r.dnsmos_ovrl:.2f} | {r.asr_transcript[:60]} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_listener_template(results: list[EvalResult], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["utterance_id", "category", "audio_path", "naturalness_1_5", "code_switch_smoothness_1_5", "pronunciation_notes"]
        )
        for r in results:
            writer.writerow([r.utterance_id, r.category, r.audio_path, "", "", ""])


if __name__ == "__main__":
    main()
