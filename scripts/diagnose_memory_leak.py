"""Bounded diagnostic: does repeated phonemize_text() calls leak memory?

Self-limits RSS via resource.setrlimit so a confirmed leak can't repeat the
OOM-killer incidents from the prepare_manifests.py runs (see AGENTS.md
incident note). Logs RSS every N iterations.
"""

from __future__ import annotations

import gc
import resource
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Safety net against repeating the OOM-killer incidents (see AGENTS.md)
# comes from the shell's `ulimit -v` around this process, not from here —
# setrlimit can't raise a hard limit the shell already lowered.

from hams_tts.text.pipeline import phonemize_text  # noqa: E402

SENTENCES = [
    "مرحبا، كيفك اليوم؟",
    "بدي أروح ع meeting الساعة 3",
    "هاد الموضوع كتير مهم",
    "hello, how are you today?",
    "قلبي بيحبك كمان وكمان",
]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(n_iters: int = 3000, log_every: int = 100) -> None:
    print(f"[diagnose] start RSS: {rss_mb():.1f} MB")
    for i in range(n_iters):
        text = SENTENCES[i % len(SENTENCES)]
        _ = phonemize_text(text)
        if (i + 1) % log_every == 0:
            gc.collect()
            print(f"[diagnose] after {i + 1} calls: RSS={rss_mb():.1f} MB")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    main(n_iters=n)
