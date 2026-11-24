"""Bounded diagnostic #5: does building the final Dataset.from_dict(...)
+ cast_column("audio", Audio(...)) over ~1500 rows of realistic-sized
float32 arrays spike anonymous memory? This is the step
`prepare_manifests.py` does once, over its *entire* accumulated `rows`
list, after processing all three source datasets.
"""

from __future__ import annotations

import gc
import resource

import numpy as np
from datasets import Audio, Dataset

SELF_ABORT_RSS_MB = 3000.0


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(n_rows: int = 1500, avg_seconds: float = 5.3, sr: int = 16000) -> None:
    print(f"[diagnose5] start RSS: {rss_mb():.1f} MB")
    rows = [
        {"array": np.random.randn(int(avg_seconds * sr)).astype(np.float32), "sampling_rate": sr}
        for _ in range(n_rows)
    ]
    gc.collect()
    print(f"[diagnose5] after building {n_rows} synthetic rows: RSS={rss_mb():.1f} MB")

    ds = Dataset.from_dict({"audio": rows, "text": [f"t{i}" for i in range(n_rows)]})
    gc.collect()
    print(f"[diagnose5] after Dataset.from_dict: RSS={rss_mb():.1f} MB")
    if rss_mb() > SELF_ABORT_RSS_MB:
        print("[diagnose5] ABORT before cast_column")
        return

    ds = ds.cast_column("audio", Audio(sampling_rate=sr))
    gc.collect()
    print(f"[diagnose5] after cast_column(Audio): RSS={rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
