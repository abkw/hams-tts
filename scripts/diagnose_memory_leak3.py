"""Bounded diagnostic #3: does `ds.set_format("numpy")` fix the leak from
diagnose_memory_leak2.py? (reading Arrow buffers directly instead of
materializing a Python list of boxed floats per row via `for ex in ds`)
"""

from __future__ import annotations

import gc
import resource
import sys
from pathlib import Path

import librosa
import numpy as np
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"

SELF_ABORT_RSS_MB = 3000.0


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(n_rows: int = 1500, log_every: int = 100) -> None:
    print(f"[diagnose3] start RSS: {rss_mb():.1f} MB")
    ds = load_from_disk(str(DATA_RAW / "clartts"))["train"]
    ds = ds.select(range(min(n_rows, len(ds))))
    ds.set_format("numpy", columns=["audio", "sampling_rate"], output_all_columns=False)
    print(f"[diagnose3] after select+set_format: RSS={rss_mb():.1f} MB")

    total_bytes = 0
    for i in range(len(ds)):
        row = ds[i]
        arr = row["audio"].astype(np.float32, copy=False)
        sr = int(row["sampling_rate"])
        resampled = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        total_bytes += resampled.nbytes
        del arr, resampled, row
        if (i + 1) % log_every == 0:
            gc.collect()
            current_rss = rss_mb()
            print(f"[diagnose3] after {i + 1} rows: RSS={current_rss:.1f} MB, cumulative audio={total_bytes / 1e6:.1f} MB")
            if current_rss > SELF_ABORT_RSS_MB:
                print(f"[diagnose3] ABORT: RSS exceeded {SELF_ABORT_RSS_MB} MB watchdog threshold")
                return


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    main(n_rows=n)
