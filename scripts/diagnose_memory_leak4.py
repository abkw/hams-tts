"""Bounded diagnostic #4: isolate whether librosa.resample (vs. dataset
row access) is the leak source, by running three variants in one process
and logging RSS after each phase.
"""

from __future__ import annotations

import gc
import resource
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(n_rows: int = 800, log_every: int = 100) -> None:
    ds = load_from_disk(str(DATA_RAW / "clartts"))["train"]
    ds = ds.select(range(min(n_rows, len(ds))))
    ds.set_format("numpy", columns=["audio", "sampling_rate"], output_all_columns=False)

    print(f"[phase=read_only] start RSS: {rss_mb():.1f} MB")
    for i in range(len(ds)):
        row = ds[i]
        arr = row["audio"].astype(np.float32, copy=True)
        del arr, row
        if (i + 1) % log_every == 0:
            gc.collect()
            print(f"[phase=read_only] after {i + 1} rows: RSS={rss_mb():.1f} MB")

    gc.collect()
    print(f"[phase=read_only] FINAL RSS: {rss_mb():.1f} MB\n")

    import librosa

    print(f"[phase=with_resample] start RSS: {rss_mb():.1f} MB")
    for i in range(len(ds)):
        row = ds[i]
        arr = row["audio"].astype(np.float32, copy=False)
        sr = int(row["sampling_rate"])
        resampled = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        del arr, resampled, row
        if (i + 1) % log_every == 0:
            gc.collect()
            print(f"[phase=with_resample] after {i + 1} rows: RSS={rss_mb():.1f} MB")
    gc.collect()
    print(f"[phase=with_resample] FINAL RSS: {rss_mb():.1f} MB")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    main(n_rows=n)
