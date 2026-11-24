"""Generate the round 1/2/3 CER-WER/DNSMOS comparison chart.

Numbers are copied from docs/design_doc.md §4 (FR4 held-out-set eval
table) and §1.1 (training data per round) — update both places together
if these ever change.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "eval_results" / "round_comparison.png"

ROUNDS = ["Round 1\n(5,073 rows, 50 epochs)", "Round 2\n(13,033 rows, 20 epochs)", "Round 3\n(13,033 rows, 30 epochs)"]

ERROR_RATE = {
    "Pure Arabic (CER)": [0.363, 0.346, 0.384],
    "Pure English (WER)": [0.048, 0.079, 0.021],
    "Code-switched (CER)": [0.593, 0.589, 0.587],
}

DNSMOS = {
    "Pure Arabic": [3.06, 3.19, 3.20],
    "Pure English": [3.03, 2.96, 3.10],
    "Code-switched": [3.13, 3.30, 3.27],
}

COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def _grouped_bars(ax, data: dict, ylabel: str, title: str) -> None:
    categories = list(data.keys())
    n_rounds = 3
    x = np.arange(len(categories))
    width = 0.25
    handles = []

    for i in range(n_rounds):
        values = [data[c][i] for c in categories]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=ROUNDS[i].split("\n")[0], color=COLORS[i])
        handles.append(bars)
        for bar, v in zip(bars, values):
            ax.annotate(
                f"{v:.3f}" if v < 1 else f"{v:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles


def main() -> None:
    fig, (ax_err, ax_mos) = plt.subplots(1, 2, figsize=(11, 4.8))

    handles = _grouped_bars(ax_err, ERROR_RATE, "Error rate (CER / WER)", "ASR round-trip error rate — lower is better")
    ax_err.set_ylim(0, 0.72)

    _grouped_bars(ax_mos, DNSMOS, "DNSMOS (1-5 scale)", "Reference-free audio quality — higher is better")
    ax_mos.set_ylim(2.5, 3.6)

    fig.suptitle(
        "Round 1 → 2 → 3: training data/epoch changes vs. measured quality",
        fontsize=12,
        fontweight="bold",
    )
    fig.legend(
        handles,
        [r.split("\n")[0] for r in ROUNDS],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        frameon=False,
    )
    fig.text(
        0.5,
        0.885,
        "Round 1: 5,073 rows, 50 epochs   |   Round 2: 13,033 rows, 20 epochs   |   Round 3 (deployed): 13,033 rows, 30 epochs",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
