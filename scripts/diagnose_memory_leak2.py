"""Bounded diagnostic #2: does iterating + resampling the ClArTTS dataset leak?

Run under a shell-level `ulimit -v` — see diagnose_memory_leak.py.
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


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


SELF_ABORT_RSS_MB = 3000.0  # self-monitoring watchdog, not an OS ulimit (mmap reservations defeat ulimit -v here)


def main(n_rows: int = 1500, log_every: int = 100) -> None:
    print(f"[diagnose2] start RSS: {rss_mb():.1f} MB")
    ds = load_from_disk(str(DATA_RAW / "clartts"))["train"]
    print(f"[diagnose2] after load_from_disk: RSS={rss_mb():.1f} MB, len={len(ds)}")
    ds = ds.select(range(min(n_rows, len(ds))))
    print(f"[diagnose2] after select: RSS={rss_mb():.1f} MB")

    total_bytes = 0
    for i, ex in enumerate(ds):
        arr = np.asarray(ex["audio"], dtype=np.float32)
        resampled = librosa.resample(arr, orig_sr=ex["sampling_rate"], target_sr=16000)
        total_bytes += resampled.nbytes
        del arr, resampled, ex
        if (i + 1) % log_every == 0:
            gc.collect()
            current_rss = rss_mb()
            print(f"[diagnose2] after {i + 1} rows: RSS={current_rss:.1f} MB, cumulative audio={total_bytes / 1e6:.1f} MB")
            if current_rss > SELF_ABORT_RSS_MB:
                print(f"[diagnose2] ABORT: RSS exceeded {SELF_ABORT_RSS_MB} MB watchdog threshold")
                return


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    main(n_rows=n)
