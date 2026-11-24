# AGENTS.md

Guidance for anyone (human or AI-assisted tooling) working in this repository.

## What this project is

A Hams.AI project: a low-latency, streaming TTS system
that speaks Levantine Arabic, English, and natural code-switching between
them, optimized to run on an NVIDIA L4-class GPU and integrated as a Pipecat
`TTSService` plugin.

Read these two files before making any non-trivial change — they are the
source of truth, not this file:

- `PRD.md` — requirements, acceptance criteria (the KPI table), deliverables,
  explicit non-goals.
- `PLAN.md` — the phased build sequence and the architecture decision
  (VITS-family / Piper / MMS-TTS-ara, chosen over StyleTTS2/XTTS-v2/Kokoro)
  with the reasoning behind it.

The original project brief (kept outside this repo) is the binding spec
if `PRD.md`/`PLAN.md` ever seem to drift from it.

## Hard constraints — do not casually relax these

These are hard acceptance criteria, not soft targets:

- Peak inference VRAM ≤ 3 GB on an L4 / RTX-3090-class GPU.
- Time-to-First-Audio (TTFA) < 300 ms.
- Real-Time Factor (RTF) < 0.3.
- Streaming (chunked) audio output — whole-utterance-only synthesis does not
  satisfy the requirement even if fast.
- Benchmark numbers that go in `docs/design_doc.md` come from the local
  RTX 4070 Ti — **this project has no L4 access** (no cloud credentials,
  no budget approved for renting one), and the user has decided the RTX
  4070 Ti's numbers are the final, cited result rather than blocking on
  L4 access. This is a deliberate, documented substitution, not an
  oversight — see `PRD.md` §8 and `docs/design_doc.md` §4 for the
  reasoning (RTX 4070 Ti is a comparable-or-stronger consumer GPU in the
  same general class the brief names as acceptable, and the model is
  small enough — 46.7M params — that the two GPUs' architectural
  differences are unlikely to flip any of the four KPIs from pass to
  fail). Don't revert to "needs L4 validation" language without the user
  re-opening that question.

If a change makes one of these worse, say so explicitly rather than quietly
shipping it — sound engineering judgment matters here, and silently
regressing a named KPI is the kind of thing that undermines trust in the
numbers.

## Architecture choice (don't re-litigate without cause)

Base model: VITS/MMS-TTS (`facebook/mms-tts-ara`, 46.7M params incl.
discriminator), **fully fine-tuned** (not LoRA — see below), extended with
a custom phoneme vocabulary covering both languages plus two reserved
Private-Use-Area marker codepoints (U+E000 Arabic-span-start, U+E001
English-span-start) for language conditioning. Full reasoning and the
rejected alternatives (StyleTTS2, XTTS-v2, Kokoro-82M) are in `PLAN.md` §0.

Two decisions from the original plan changed once real vendor code/API
constraints were in hand — both documented in `PLAN.md`, don't silently
re-revert either without re-reading why:

- **LoRA → full fine-tune.** The model is only ~47M params; the vendored
  training recipe (below) already demonstrates full-finetune VITS/MMS
  convergence on 80-150 samples in ~20 minutes. No mature LoRA tooling
  exists for VITS's flow/conv layers, so full fine-tune is both simpler
  and lower-risk than building adapter injection from scratch, with no
  real overfitting concern given our dataset is far larger than 150
  samples.
- **Separate language-ID embedding table → PUA marker tokens.** Adding a
  new embedding table to a pretrained VITS encoder's forward pass is a
  real architecture change with real integration risk. Since
  `VitsTokenizer._tokenize` maps text one Unicode codepoint at a time
  (confirmed by reading transformers' `tokenization_vits.py`), reserving
  two codepoints as language markers gets the same conditioning signal
  through the *existing* embedding table with zero model-code changes.
  Documented as a known limitation (weaker signal than a dedicated,
  per-frame embedding) in the design doc, not silently presented as
  equivalent.

Training is **not** a from-scratch VITS GAN loop we wrote — that would
have reimplemented well-trodden commodity code for no benefit. We vendored
`ylacombe/finetune-hf-vits` (MIT license) into
`src/hams_tts/training/vendor/`, patched it for transformers 5.x
compatibility (two removed/changed APIs — `_prepare_4d_attention_mask` and
`VitsConfig`'s `pad_token_id` default; see `# HAMS:` comments in that
directory and `VENDORED_FROM.md`), and point it at our own phonemized
data + custom tokenizer via its existing `tokenizer_name` /
`override_vocabulary_embeddings` / `override_speaker_embeddings` config
flags — no fork-specific logic needed for that part.

Don't swap architectures mid-project without updating `PLAN.md`'s decision
table and explaining why the original reasoning (latency risk of
autoregressive/diffusion approaches vs. the hard KPI table) no longer holds.

## Repo layout

Scaffolded and in active use:

```
src/hams_tts/
  text/                  # normalize, diacritize (mishkal), g2p (espeak-ng),
                         # levantine_rules, pipeline.py (entry point), vocab.py
  training/
    vendor/              # ylacombe/finetune-hf-vits, vendored + patched (see above)
    data_prep/           # buckwalter.py (ASC transliteration), codeswitch_templates.py
  inference/
    engine.py            # Engine class: the one place that loads the model —
                         # server, benchmark script, and Pipecat plugin all use it
    chunking.py          # clause-boundary text chunking for streaming
  server/                # FastAPI + WebSocket streaming server
  pipecat_plugin/        # Pipecat TTSService subclass
configs/finetune_hams_levantine.json   # training config (points at vendor/run_vits_finetuning.py)
scripts/                 # download_datasets.py, prepare_manifests.py, build_vocab.py, benchmark.py
data/
  raw/{clartts,asc,ljspeech,checkpoints}/   # downloaded datasets + converted base checkpoint
  processed/{hams_combined,tokenizer,checkpoint}/   # phonemized training data, custom vocab, trained model
  manifests/held_out_eval.jsonl            # FR4's 10-20 utterance eval set (hand-curated)
```

## Environment

- Local GPU: NVIDIA RTX 4070 Ti, 12 GB VRAM, driver supports CUDA 13.0.
  Good enough for dev iteration and even full fine-tuning at the model
  sizes involved (VITS-family models are small, tens of millions of
  params) — it is **not** the target deployment GPU.
- Python 3.12, managed with **uv** (`uv sync`, `uv run ...`). `pyproject.toml`
  pins torch to the `pytorch-cu124` index (CUDA 13.0 driver is backward
  compatible with cu124 wheels).
- **espeak-ng is vendored, not system-installed** — the dev box has no
  passwordless sudo. It was extracted from the Ubuntu `.deb` packages via
  `apt-get download espeak-ng espeak-ng-data libespeak-ng1` +
  `dpkg-deb -x` into `.local-tools/` at the repo root (no root required).
  `src/hams_tts/text/espeak_setup.py` points `phonemizer` at that vendored
  `libespeak-ng.so.1` — always import/call `ensure_espeak_configured()`
  before any `phonemizer` call elsewhere in new code.
- No GPU cloud credentials are configured — renting an L4 instance was
  considered and explicitly declined by the user (2026-11-23); the RTX
  4070 Ti benchmark numbers in `docs/design_doc.md` are final, not a
  placeholder awaiting L4 validation. Don't rent an L4 or re-raise this
  as an open task without being asked.
- **The dev box's Python processes share a memory cgroup with VSCode**
  (`snap.code.code-*.scope`) — a Python process that runs away on RAM can
  get OOM-killed by the kernel and take the editor down with it, not just
  itself. See the incident note below before writing any script that
  loads/holds more than a few hundred MB of data.

### Incident: `prepare_manifests.py` OOM-killed VSCode twice (2026-11-23)

`journalctl -k` confirmed two kernel OOM kills of a `python3` process
(`anon-rss` ~24GB each) inside the VSCode snap's cgroup, both from running
`scripts/prepare_manifests.py`. Root-caused with bounded, watchdog-guarded
diagnostic scripts (`scripts/diagnose_memory_leak*.py` — kept in the repo
as a worked example of this debugging technique, not routinely run):

- Per-row text phonemization and per-row audio resampling were both ruled
  out — flat memory in isolation.
- The actual cause: `datasets.Dataset.from_dict(...)` given a list of
  `{"array": numpy_array, "sampling_rate": int}` dicts (meant for the
  `Audio` feature) uses roughly **9x** the raw audio data's size while
  converting it to an Arrow table (measured: 1500 rows / ~510MB raw
  float32 -> 5.2GB mid-construction). The old script held *all* ~5000
  training utterances' decoded audio in a Python list and then handed
  that whole list to `from_dict` in one call — the input list and the
  ~9x-inflated conversion were briefly alive at the same time.

Fix (now in `prepare_manifests.py`): every utterance's audio is written
straight to a small WAV file on disk as soon as it's produced; only the
*file path* is kept in memory per row. `Dataset.from_dict` then builds
from path strings (cheap), and `.cast_column("audio", Audio(...))` reads
bytes from disk lazily per row instead of needing everything in RAM at
once. Confirmed fix: peak RSS during the full real run dropped from the
24GB that got OOM-killed to **2.3GB**.

Two things to carry forward into any new data-prep code in this repo:
1. **Never build a `datasets.Dataset`/`Dataset.from_dict` from a column of
   embedded numpy/audio arrays for anything but a tiny (~dozens of rows)
   dataset.** Write to disk and reference by path instead.
2. **Add periodic RSS logging and a hard watchdog threshold** to any
   script that loops over more than a few hundred dataset rows (see
   `_rss_mb()` / `_rss_guard()` / `_log_progress()` in
   `prepare_manifests.py`) — there was no in-process memory visibility at
   all before this incident, so the only evidence was a kernel log dug up
   after the fact. A script that can self-report "RSS=X MB after N rows"
   and abort cleanly before a kernel OOM kill is much easier to debug and
   far less disruptive than one that doesn't.

### Incident: GPU VRAM contention with Ollama crashed the training run (2026-11-23)

The first real `accelerate launch run_vits_finetuning.py` attempt died
silently — no Python traceback, the log just stops mid-`preprocess train
dataset` — right around when the training script would have moved the
model onto CUDA. `nvidia-smi` at the time showed **9.6-10 GB of the
12 GB GPU already in use** by an unrelated `ollama runner` process serving
`deepseek-r1:14b` (left loaded from earlier, separate work — not anything
this project started). `journalctl` around the same time showed the
desktop's own Chromium-based GPU process repeatedly failing to launch
(`GPU process launch failed: error_code=1002`) — consistent with this
being a genuinely *shared* GPU: it drives the desktop display (gnome-shell/
Xorg) as well as CUDA compute, so a process that can't get the VRAM it
needs doesn't just fail cleanly with `torch.cuda.OutOfMemoryError` — it
can destabilize the graphics stack other apps (including the editor)
depend on. This is the same cgroup-sharing hazard as the RAM/VSCode
incident above, just on the GPU instead of system RAM.

Fix: freed the GPU (`ollama stop <model>`, confirmed via
`nvidia-smi --query-compute-apps` showing zero processes and
`memory.used` back down to baseline ~560 MiB) before relaunching.

**Before starting any GPU training/inference run in this repo, check
`nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`
first.** This machine runs other GPU workloads (at minimum: `ollama`,
which the user drives independently of this project) that silently eat
into the 12 GB RTX 4070 Ti's budget — don't assume the whole GPU is free
just because this project isn't the one that allocated the memory.

## Working conventions

- Text pipeline modules (`src/hams_tts/text/`) must stay independently
  testable from the model — phonemization correctness should be checkable
  without loading a checkpoint. Keep `espeak-ng` / CAMeL Tools calls behind
  thin wrappers so they can be unit-tested with fixed input/output pairs.
- Levantine phonological overrides (ق→ʔ, ج→ʒ, vowel reduction, dialect
  lexicon) live in `levantine_rules.py` as an explicit, inspectable rule/
  lexicon table — not buried as magic constants inside the G2P call site.
  This table is expected to grow over time as eval failures surface specific
  mispronounced tokens; keep it easy to append to.
- Any new training data source gets recorded in `PLAN.md` Phase 2 /
  `docs/design_doc.md`'s data section with its license situation noted, even
  informally — several candidate corpora mentioned in the brief require
  registration/LDC agreements, and substitutions must be traceable.
- Benchmark scripts (`eval/benchmark.py`) must run against the **deployed
  server**, not just the in-process model — TTFA measured in-process
  understates real latency (serialization, network, websocket framing) and
  would misrepresent the KPI.
- The Pipecat plugin should mirror the structure of an existing in-tree
  Pipecat TTS service rather than inventing new conventions — check
  Pipecat's own TTS service implementations for the expected shape of
  `run_tts` / frame emission / interruption handling before writing it.

## Commands

All commands run through `uv run` (don't `pip install` into system Python).
espeak-ng env vars are set automatically by `espeak_setup.py` — no need to
export `LD_LIBRARY_PATH`/`ESPEAK_DATA_PATH` manually outside one-off shell
debugging.

```sh
uv sync                                   # install/update deps

uv run pytest tests/ -q                   # text pipeline (+ server/plugin) tests

uv run python3 scripts/download_datasets.py   # Phase 2: cache ClArTTS/ASC/LJSpeech to data/raw/
uv run python3 scripts/prepare_manifests.py   # phonemize everything, write data/processed/hams_combined/
uv run python3 scripts/build_vocab.py         # build custom vocab from the phonemized corpus

cd src/hams_tts/training/vendor && uv run --project /home/mohammed/Desktop/hams \
  accelerate launch run_vits_finetuning.py ../../../../configs/finetune_hams_levantine.json
  # must run with cwd = vendor/ — it does `from monotonic_align import ...` /
  # `from utils import ...` as top-level imports, not a package-relative import

uv run python3 scripts/benchmark.py --mode local --model-dir data/processed/checkpoint
  # dev-box numbers only — see PRD §8, not what goes in the design doc

uv run uvicorn hams_tts.server.main:app --host 0.0.0.0 --port 8000   # (confirm exact app path once server/main.py lands)
```

Re-running `convert_original_discriminator_checkpoint.py` for Arabic is a
one-time step already done — the output lives at
`data/raw/checkpoints/mms-tts-ara-train/`. Don't re-run it unless that
directory is deleted; it re-downloads a large file from
`facebook/mms-tts`'s `full_models/ara/` subfolder each time.

## Current status

Phases 0-5 complete, three fine-tuning rounds done:
- **Round 1**: 50 epochs / 15,750 steps on 5,073 rows (ClArTTS capped at 1,500).
- **Round 2**: 20 epochs / 16,240 steps on 13,033 rows (full ClArTTS, all 9,500 rows) — after round-1 Arabic pronunciation wasn't good enough on informal listening.
- **Round 3**: 30 epochs / 24,360 steps on the same 13,033 rows as round 2 — after real listening feedback on round 2 (see `docs/eval_results/diagnostic_samples/`) found specific mis-diacritized dialect words (fixed via 8 new `levantine_rules.py` lexicon entries, no retraining needed) plus a genuine model-level ح/ʕ confound (documented, not fixed), and round 2's higher final KL loss suggested under-convergence. **Round 3 is the currently deployed checkpoint.**

`data/processed/checkpoint_round2_backup/` keeps round 2 on disk in case
round 3 needs to be rolled back. Don't assume any single round's numbers
describe what's currently running — `docs/design_doc.md` §1.1/§3/§4
carry all three rounds' numbers side by side on purpose; re-read those
before citing a number from memory, don't rely on this summary alone.
All four hard KPIs pass in every round, on the RTX 4070 Ti, which is the
**final** cited benchmark hardware (see the "Hard constraints" section
above — L4 access was explicitly declined by the user, not a gap). 3
required sample WAVs at `samples/` are from round 3. The lexicon fixes
in `levantine_rules.py` apply regardless of which round is deployed,
since phonemization happens at inference time, not training time.

Published:
- Code: https://github.com/abkw/hams-tts (public; pushed 2026-11-23)
- Model: https://huggingface.co/mohammed/hams-tts-levantine (public;
  pushed 2026-11-23 — `config.json`, `model.safetensors`,
  `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`,
  `README.md` from `docs/model_card.md`)

Still outstanding (the user has taken ownership of these, not blocked on
agent time): recording the ≤3-minute demo video
(`docs/demo_video_script.md` has the shot list and exact commands);
running the human MOS listening test
(`docs/eval_results/listener_scoring_template.csv` + the WAVs in
`docs/eval_results/audio/` are ready to hand to ≥5 listeners).
