"""Reproducible TTFA / RTF / peak-VRAM benchmark (PRD §6, PLAN.md Phase 5).

Two modes:

  --mode local   Calls the inference `Engine` in-process. Useful for fast
                 iteration during development (no server needed) — skips
                 network/serialization overhead, so it understates real
                 latency; not what goes in the design doc's results table.

  --mode server  Drives an already-running `hams_tts.server` instance over
                 its WebSocket endpoint end-to-end (network + serialization
                 included), which is what the brief actually asks for:
                 "a reproducible benchmarking script that measures TTFA,
                 RTF, and peak VRAM against the target hardware. We should
                 be able to run it." This is the number that belongs in
                 the design doc's results table.

  The brief names an L4-class GPU as the deployment target; this project
  has no L4 access (no cloud credentials/budget), so the RTX 4070 Ti dev
  box is the final cited hardware instead — a deliberate, documented
  decision (see PRD.md §8 / design_doc.md §4 for the reasoning), not a
  placeholder awaiting an L4 run.

Peak VRAM is sampled externally via `nvidia-smi` polling on a background
thread rather than `torch.cuda.max_memory_allocated()` in both modes
uniformly — in `--mode local` that in-process call would also work, but
`--mode server` has no access to the server process's CUDA context, and a
single external-sampling implementation that works identically in both
modes is less code to get wrong than two different measurement paths that
are then hard to compare apples-to-apples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

HELD_OUT_EVAL = REPO_ROOT / "data" / "manifests" / "held_out_eval.jsonl"


@dataclass
class RunResult:
    utterance_id: str
    category: str
    ttfa_seconds: float
    total_gen_seconds: float
    audio_seconds: float
    rtf: float


@dataclass
class VramSampler:
    device_index: int = 0
    _peak_mib: float = field(default=0.0, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                mib = float(out.stdout.strip().splitlines()[0])
                self._peak_mib = max(self._peak_mib, mib)
            except Exception:
                pass
            time.sleep(0.05)

    def __enter__(self) -> "VramSampler":
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_mib(self) -> float:
        return self._peak_mib


def load_eval_utterances() -> list[dict]:
    rows = []
    with HELD_OUT_EVAL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_local(model_dir: str, tokenizer_dir: str | None, device: str | None, dtype: str = "float32") -> list[RunResult]:
    import torch

    from hams_tts.inference.engine import Engine

    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    engine = Engine(model_dir=model_dir, tokenizer_dir=tokenizer_dir, device=device, dtype=torch_dtype)
    results = []
    for row in load_eval_utterances():
        lang_hint = "en" if row["category"] == "en" else "ar"
        _, stats = engine.synthesize_streaming_with_stats(row["text"], default_lang=lang_hint)
        results.append(
            RunResult(
                utterance_id=row["id"],
                category=row["category"],
                ttfa_seconds=stats.ttfa_seconds,
                total_gen_seconds=stats.total_gen_seconds,
                audio_seconds=stats.audio_seconds,
                rtf=stats.rtf,
            )
        )
    return results


async def _run_server_one(uri: str, text: str, lang_hint: str) -> tuple[float, float, float]:
    import websockets

    start = time.perf_counter()
    ttfa = None
    total_audio_bytes = 0
    sampling_rate = 16000

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"text": text, "lang_hint": lang_hint}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                if ttfa is None:
                    ttfa = time.perf_counter() - start
                total_audio_bytes += len(msg)
                continue
            header = json.loads(msg)
            if "sampling_rate" in header:
                sampling_rate = header["sampling_rate"]
            if header.get("error"):
                raise RuntimeError(f"server error: {header['error']}")
            # `is_final` (on a chunk header) still has that chunk's binary
            # frame to come next — only `stream_end` (sent *after* the last
            # chunk's bytes) means there's nothing left to receive. Breaking
            # on `is_final` instead would drop the last chunk's audio
            # entirely for any single-chunk utterance.
            if header.get("type") == "stream_end":
                break

    total_gen = time.perf_counter() - start
    audio_seconds = (total_audio_bytes / 2) / sampling_rate  # int16 PCM mono
    return ttfa or total_gen, total_gen, audio_seconds


def run_server(host: str, port: int, path: str = "/tts/stream") -> list[RunResult]:
    uri = f"ws://{host}:{port}{path}"
    results = []
    for row in load_eval_utterances():
        lang_hint = "en" if row["category"] == "en" else "ar"
        ttfa, total_gen, audio_seconds = asyncio.run(_run_server_one(uri, row["text"], lang_hint))
        rtf = (total_gen / audio_seconds) if audio_seconds > 0 else float("nan")
        results.append(
            RunResult(
                utterance_id=row["id"],
                category=row["category"],
                ttfa_seconds=ttfa,
                total_gen_seconds=total_gen,
                audio_seconds=audio_seconds,
                rtf=rtf,
            )
        )
    return results


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(results: list[RunResult], peak_vram_mib: float | None) -> str:
    ttfas = [r.ttfa_seconds for r in results]
    rtfs = [r.rtf for r in results if r.rtf == r.rtf]  # filter NaN

    lines = [
        "| Metric | Target | Measured |",
        "|---|---|---|",
        f"| TTFA p50 | < 300 ms | {percentile(ttfas, 0.5) * 1000:.1f} ms |",
        f"| TTFA p95 | < 300 ms | {percentile(ttfas, 0.95) * 1000:.1f} ms |",
        f"| RTF (mean) | < 0.3 | {statistics.mean(rtfs):.3f} |",
        f"| RTF (p95) | < 0.3 | {percentile(rtfs, 0.95):.3f} |",
        f"| Peak VRAM | <= 3 GB | {f'{peak_vram_mib / 1024:.2f} GB' if peak_vram_mib else 'n/a (no CUDA device sampled)'} |",
        f"| Utterances benchmarked | 10-20 | {len(results)} |",
        "",
        "| id | category | TTFA (ms) | RTF | audio (s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.utterance_id} | {r.category} | {r.ttfa_seconds * 1000:.1f} | {r.rtf:.3f} | {r.audio_seconds:.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local", "server"], required=True)
    parser.add_argument("--model-dir", default="data/processed/checkpoint")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--out", default="docs/benchmark_results.md")
    args = parser.parse_args()

    with VramSampler(device_index=args.gpu_index) as sampler:
        if args.mode == "local":
            results = run_local(args.model_dir, args.tokenizer_dir, args.device, args.dtype)
        else:
            results = run_server(args.host, args.port)
        time.sleep(0.1)  # let the sampler catch the tail end
        peak_mib = sampler.peak_mib

    table = summarize(results, peak_mib)
    print(table)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Benchmark results ({args.mode} mode)\n\n{table}\n",
        encoding="utf-8",
    )
    print(f"\n[benchmark] wrote {out_path}")


if __name__ == "__main__":
    main()
