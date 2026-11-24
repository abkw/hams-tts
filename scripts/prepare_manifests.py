"""Phase 2 (PLAN.md): turn the three raw datasets into one combined,
phonemized, 16kHz training manifest ready for `run_vits_finetuning.py`.

For each source dataset:
  - ClArTTS: already-diacritized MSA text, used as-is.
  - ASC (Arabic Speech Corpus): Buckwalter transcripts -> Arabic script
    (`data_prep.buckwalter`), which happens to also recover full tashkeel.
  - LJSpeech: `normalized_text` column, English.

Every utterance's text is run through OUR phonemization pipeline
(`hams_tts.text.pipeline.phonemize_text`) rather than any phonemes bundled
with the source dataset — this is what makes Arabic and English share one
phoneme vocabulary with Levantine corrections and language-marker tokens
applied consistently, regardless of which corpus a sentence came from.

Each source corpus is effectively single-speaker, so we assign one
`speaker_id` per dataset (0=ClArTTS, 1=ASC, 2=LJSpeech) rather than trying
to cluster speakers within a corpus.

Sample caps (`MAX_PER_DATASET`) bound preprocessing/training wall-clock
time for this project; the full corpora remain on disk under
data/raw/ for a larger follow-up run — see PRD §8 / design doc limitations.

Memory incident (2026-11-23): an earlier version of this script held every
resampled utterance's audio as an in-memory numpy array in a big Python
list, then handed that whole list straight to `datasets.Dataset.from_dict`
with embedded arrays under an "audio" dict key. That got the process
OOM-killed twice by the kernel (confirmed via `journalctl -k`:
`anon-rss:~24GB`), and since the killed process's memory cgroup was shared
with the VSCode snap, it took the editor down with it both times.
Root-caused with the bounded, watchdog-guarded diagnostics in
`scripts/diagnose_memory_leak*.py`: `Dataset.from_dict` converting a list
of {"array": ndarray, "sampling_rate": int} dicts into an Arrow table uses
something like 9x the raw audio data size while doing that conversion
(measured: 1500 rows / ~510MB raw float32 -> 5.2GB mid-construction), on
top of the input list itself still being held — the classic "two full
copies alive during a conversion, and the library's conversion path isn't
memory-efficient for this particular shape of input" problem.

Fix applied below: every utterance's resampled audio is written straight
to a small WAV file on disk as soon as it's produced (`_write_wav`), and
only the *file path* (a string) is kept in memory per row — audio never
exists as an in-memory array for more than one iteration of the loop.
`Dataset.from_dict` then builds the final dataset from path strings, and
`.cast_column("audio", Audio(...))` makes `datasets` read bytes from disk
lazily per-row at `.to_parquet()` time instead of ever materializing all
rows in RAM simultaneously.

A periodic RSS logger (`_log_progress`) and a hard watchdog (`_rss_guard`)
are kept in even though the fix should make them unnecessary — cheap
insurance, and exactly the kind of visibility that was missing last time
(there was no in-process memory logging at all before this incident, so
the only evidence was a kernel log dug up after the fact).
"""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hams_tts.text.pipeline import phonemize_text  # noqa: E402
from hams_tts.training.data_prep.buckwalter import buckwalter_to_arabic  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
AUDIO_CACHE = DATA_PROCESSED / "audio_cache"
TARGET_SR = 16000

MAX_PER_DATASET = {"clartts": 100_000, "asc": 10_000, "ljspeech": 1700}
# clartts/asc caps are both "effectively all" sentinels (9,500 and 1,913
# available respectively) — ClArTTS was capped at 1,500 in the first
# training round to bound wall-clock; raised to use the full corpus once
# pronunciation quality (not wall-clock) became the limiting factor. The
# vocab resize (§ design_doc.md "Custom vocabulary") means the text
# encoder's embedding table was reinitialized from scratch, so it needs
# more Arabic phoneme exposure than a small subset gives it. LJSpeech
# stays capped — English pronunciation was already near-perfect (§4 WER),
# so there's no quality reason to add training time there.
EVAL_HOLDOUT_PER_DATASET = 40  # carved out for the training script's own periodic eval

SPEAKER_IDS = {"clartts": 0, "asc": 1, "ljspeech": 2}

RSS_LOG_EVERY = 100
# Watchdog on *anonymous* RSS specifically, not total RSS. Processing the
# full ClArTTS corpus sequentially touches nearly the whole ~13GB
# memory-mapped arrow file, which legitimately grows total RSS past 6GB —
# but that growth is clean, reclaimable file-backed page cache (confirmed
# via /proc/self/status's RssFile), not the unreclaimable heap growth that
# caused the original incident (`anon-rss:~24GB, file-rss:8kB` in the
# kernel OOM log — see the module docstring). The kernel reclaims RssFile
# under pressure before ever reaching the OOM killer, so it isn't the
# dangerous kind of memory growth; RssAnon is. Gating on total RSS here
# would make processing the full corpus impossible without actually fixing
# anything real.
ANON_RSS_ABORT_MB = 6000.0


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _anon_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024
    return 0.0  # /proc unavailable (non-Linux) — guard becomes a no-op


def _rss_guard(context: str) -> None:
    anon_rss = _anon_rss_mb()
    if anon_rss > ANON_RSS_ABORT_MB:
        raise MemoryError(
            f"[prepare] ABORT during {context}: anonymous RSS={anon_rss:.0f} MB exceeded the "
            f"{ANON_RSS_ABORT_MB:.0f} MB safety watchdog. Stopping cleanly instead of risking "
            "another OOM kill — see the module docstring's 'Memory incident' note for what this "
            "is guarding against. (Total RSS, including reclaimable file-backed pages from the "
            "large mmap'd dataset files, is logged separately and is expected to run higher.)"
        )


def _log_progress(context: str, i: int) -> None:
    if (i + 1) % RSS_LOG_EVERY == 0:
        print(
            f"[prepare] {context}: {i + 1} rows done, RSS={_rss_mb():.0f} MB "
            f"(anon={_anon_rss_mb():.0f} MB)",
            flush=True,
        )
        _rss_guard(context)


def _resample(array: np.ndarray, orig_sr: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if orig_sr == TARGET_SR:
        return array
    return librosa.resample(array, orig_sr=orig_sr, target_sr=TARGET_SR)


def _write_wav(source: str, idx: int, audio: np.ndarray) -> str:
    out_dir = AUDIO_CACHE / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{idx:06d}.wav"
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")
    return str(path)


def _process_clartts(limit: int) -> list[dict]:
    ds = load_from_disk(str(DATA_RAW / "clartts"))["train"]
    ds = ds.select(range(min(limit, len(ds))))
    ds.set_format("numpy", columns=["audio", "sampling_rate"], output_all_columns=True)
    rows = []
    for i in range(len(ds)):
        ex = ds[i]
        audio = _resample(ex["audio"], int(ex["sampling_rate"]))
        phon = phonemize_text(str(ex["text"]), default_lang="ar").phoneme_string
        if phon.strip():
            path = _write_wav("clartts", i, audio)
            rows.append({"audio_path": path, "text": phon, "speaker_id": SPEAKER_IDS["clartts"], "source": "clartts"})
        del ex, audio
        _log_progress("clartts", i)
    return rows


def _process_asc(limit: int) -> list[dict]:
    rows = []
    global_i = 0
    for split in ("train", "test"):
        ds = load_from_disk(str(DATA_RAW / "asc"))[split]
        for i in range(len(ds)):
            if len(rows) >= limit:
                break
            ex = ds[i]
            arabic_text = buckwalter_to_arabic(ex["text"])
            audio_arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
            audio = _resample(audio_arr, ex["audio"]["sampling_rate"])
            phon = phonemize_text(arabic_text, default_lang="ar").phoneme_string
            if phon.strip():
                path = _write_wav("asc", global_i, audio)
                rows.append({"audio_path": path, "text": phon, "speaker_id": SPEAKER_IDS["asc"], "source": "asc"})
            del ex, audio_arr, audio
            global_i += 1
            _log_progress(f"asc/{split}", global_i)
    return rows


def _process_ljspeech(limit: int) -> list[dict]:
    ds = load_from_disk(str(DATA_RAW / "ljspeech"))["train"]
    ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for i in range(len(ds)):
        ex = ds[i]
        audio_arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
        audio = _resample(audio_arr, ex["audio"]["sampling_rate"])
        phon = phonemize_text(ex["normalized_text"], default_lang="en").phoneme_string
        if phon.strip():
            path = _write_wav("ljspeech", i, audio)
            rows.append({"audio_path": path, "text": phon, "speaker_id": SPEAKER_IDS["ljspeech"], "source": "ljspeech"})
        del ex, audio_arr, audio
        _log_progress("ljspeech", i)
    return rows


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] start RSS={_rss_mb():.0f} MB")

    print("[prepare] processing ClArTTS...")
    clartts_rows = _process_clartts(MAX_PER_DATASET["clartts"] + EVAL_HOLDOUT_PER_DATASET)
    print(f"  -> {len(clartts_rows)} rows, RSS={_rss_mb():.0f} MB")

    print("[prepare] processing ASC...")
    asc_rows = _process_asc(MAX_PER_DATASET["asc"] + EVAL_HOLDOUT_PER_DATASET)
    print(f"  -> {len(asc_rows)} rows, RSS={_rss_mb():.0f} MB")

    print("[prepare] processing LJSpeech...")
    ljspeech_rows = _process_ljspeech(MAX_PER_DATASET["ljspeech"] + EVAL_HOLDOUT_PER_DATASET)
    print(f"  -> {len(ljspeech_rows)} rows, RSS={_rss_mb():.0f} MB")

    def split_train_eval(rows: list[dict]) -> tuple[list[dict], list[dict]]:
        if len(rows) <= EVAL_HOLDOUT_PER_DATASET:
            return rows, []
        return rows[:-EVAL_HOLDOUT_PER_DATASET], rows[-EVAL_HOLDOUT_PER_DATASET:]

    clartts_train, clartts_eval = split_train_eval(clartts_rows)
    asc_train, asc_eval = split_train_eval(asc_rows)
    ljspeech_train, ljspeech_eval = split_train_eval(ljspeech_rows)

    train_rows = clartts_train + asc_train + ljspeech_train
    eval_rows = clartts_eval + asc_eval + ljspeech_eval

    print(f"[prepare] total train rows: {len(train_rows)}, eval rows: {len(eval_rows)}")

    # Exported as plain parquet with canonical split-keyword filenames
    # (train / validation) rather than `save_to_disk`'s arrow layout: the
    # vendored `run_vits_finetuning.py` calls `datasets.load_dataset(
    # dataset_name, dataset_config_name, split=...)` with no `data_files`
    # kwarg exposed, so the only way to point it at local data without
    # further modifying that script is to hand it a directory that
    # `load_dataset`'s generic parquet loader can auto-split by filename.
    # "validation" (not "eval") is used for the second split because it's
    # one of the three canonical split keywords the auto-inference
    # recognizes — `eval_split_name` in the training config below is set
    # to "validation" to match.
    combined_dir = DATA_PROCESSED / "hams_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    split_file_names = {"train": "train-00000-of-00001.parquet", "eval": "validation-00000-of-00001.parquet"}

    for split_key, rows in (("train", train_rows), ("eval", eval_rows)):
        if not rows:
            continue
        # NOTE: "audio" here is a list of file PATH STRINGS, not embedded
        # arrays — see module docstring. Dataset.from_dict on path strings
        # is cheap; cast_column(Audio) + to_parquet() then stream bytes
        # from disk per row instead of needing everything in RAM at once.
        ds = Dataset.from_dict(
            {
                "audio": [r["audio_path"] for r in rows],
                "text": [r["text"] for r in rows],
                "speaker_id": [r["speaker_id"] for r in rows],
                "source": [r["source"] for r in rows],
            }
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        _rss_guard(f"cast_column before writing {split_key}")
        out_path = combined_dir / split_file_names[split_key]
        ds.to_parquet(str(out_path))
        print(f"[prepare] saved {split_key} -> {out_path} ({len(ds)} rows), RSS={_rss_mb():.0f} MB")

    vocab_source_path = DATA_PROCESSED / "phoneme_corpus.jsonl"
    with vocab_source_path.open("w", encoding="utf-8") as f:
        for r in train_rows + eval_rows:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"[prepare] wrote phoneme corpus for vocab building -> {vocab_source_path}")
    print(f"[prepare] FINAL RSS={_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
