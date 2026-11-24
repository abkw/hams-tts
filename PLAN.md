# Implementation Plan — Low-Latency Code-Switching TTS (Levantine Arabic)

Companion to `PRD.md` (requirements/acceptance criteria) and `AGENTS.md`
(repo conventions). This document is the technical how: architecture
decision with reasoning, repo layout, and a phased build sequence.

## 0. Architecture Decision (made up front, drives everything else)

**Decision: fine-tune a VITS-family, non-autoregressive model — using
Piper's VITS recipe and/or Meta MMS-TTS (`facebook/mms-tts-ara`) as the
pretrained starting point — rather than a diffusion/style-based model
(StyleTTS2) or an autoregressive multilingual model (XTTS-v2, Tortoise-style).**

### Why this and not the alternatives

| Candidate | Naturalness | TTFA/RTF risk | VRAM | AR/Levantine support | Verdict |
|---|---|---|---|---|---|
| **VITS / Piper / MMS-TTS-ara** | Good (not SOTA) | Low — single non-autoregressive forward pass, latency ~flat w.r.t. text length | Small (~30–90M params), fits ≤3GB easily even unquantized | MMS has an Arabic checkpoint; Piper ecosystem already ships Arabic + English voices with espeak-ng phonemization | **Chosen** |
| StyleTTS2 | Excellent | Medium-high — diffusion-based style sampling adds latency and makes TTFA variance harder to bound | Medium | No off-the-shelf Arabic checkpoint; would need training from scratch | Rejected: latency risk against hard TTFA/RTF targets outweighs quality gain |
| XTTS-v2 (Coqui) | Very good, zero-shot cloning | High — GPT-style autoregressive token loop, latency scales with output length | ~1.8GB+ model alone before KV-cache/vocoder headroom | Multilingual incl. Arabic (MSA-leaning) | Rejected: autoregressive decoding makes RTF<0.3 and consistent TTFA<300ms across utterance lengths materially harder to guarantee |
| Kokoro-82M | Excellent for English | Low (small, fast) | Very small | No native Arabic — would need a new language essentially from scratch | Rejected: would cost more data/training effort than it saves, for a language we don't have confidence it'll fit well |

The brief's KPI table (VRAM ≤3GB, TTFA<300ms, RTF<0.3, streaming required)
is the primary filter — it explicitly rewards justified optimization of an
open-source base over raw naturalness. VITS's single-pass, parallel
mel+waveform generation is the architecture class best aligned with that,
and Piper already solves the "productionize a VITS model with ONNX +
streaming" half of the problem, which de-risks the deployment requirement.

**Fallback if eval naturalness is unacceptable**: keep StyleTTS2 as a
documented plan-B, explored only if time remains after the VITS path hits
its KPIs — call this out in the design doc's "known limitations" rather than
building both.

### Conditioning for code-switching — revised after reading the actual vendor code/API

The two items below were the original plan. Both changed once implementation
started and the real constraints (not assumptions about them) were in hand —
see `AGENTS.md`'s "Architecture choice" section for the short version; full
reasoning here.

- ~~Add a **language-ID embedding** (AR / EN flag) concatenated/added at the
  phoneme-encoder input~~ → **Implemented as two reserved Private-Use-Area
  marker codepoints** (U+E000 = Arabic span start, U+E001 = English span
  start) inserted into the phoneme string by `hams_tts.text.pipeline` at
  each language span boundary. `transformers`' `VitsTokenizer._tokenize`
  tokenizes one Unicode codepoint at a time (per `tokenization_vits.py`),
  so these markers become ordinary tokens through the model's existing
  embedding table — the
  self-attention text encoder can learn to use them as context with zero
  changes to pretrained model code. A dedicated embedding table added at
  the model level (the original plan) would likely condition more
  strongly, since it could be broadcast per-frame rather than competing as
  one token among many — that's documented as a known limitation / future
  improvement, not implemented, because patching a pretrained checkpoint's
  forward pass carries real integration risk for a benefit that's hard to
  measure in advance. The unified IPA-ish phoneme inventory (shared across
  languages) is unchanged from the original plan.

- ~~Fine-tune with **LoRA adapters** on the text encoder + duration
  predictor~~ → **Full fine-tune.** Once we had the actual training
  recipe in hand (`ylacombe/finetune-hf-vits`, vendored — see Phase 3
  below), it was clear the model is only ~47M params total (generator +
  discriminator) and that recipe's own documented results (full-finetune
  convergence in ~20 minutes on 80-150 samples) directly contradict the
  overfitting concern that originally motivated LoRA. There's also no
  mature, well-tested LoRA implementation for VITS's normalizing-flow and
  conv-based duration predictor layers the way there is for standard
  transformer attention blocks — building one from scratch would have
  added engineering risk for a problem (overfitting on scarce data) that
  doesn't actually apply at our dataset size. Full fine-tune is simpler
  and lower-risk.

## 1. Repo Layout (target end-state)

```
hams/
├── AGENTS.md
├── PRD.md
├── PLAN.md
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/                  # untouched downloads (gitignored)
│   ├── processed/            # aligned/cleaned clips + transcripts
│   └── manifests/            # train/val/held-out-eval JSONL manifests
├── src/hams_tts/
│   ├── text/
│   │   ├── normalize.py      # numbers/dates/currency/punctuation
│   │   ├── diacritize.py     # CAMeL Tools / Shakkala wrapper
│   │   ├── lang_segment.py   # script-based AR/EN span splitting
│   │   ├── g2p.py            # espeak-ng wrapper + phoneme unification
│   │   └── levantine_rules.py# ق→ʔ, ج→ʒ, vowel reduction, lexicon overrides
│   ├── model/
│   │   ├── vits/              # model definition (fork of base arch)
│   │   ├── lang_speaker_embed.py
│   │   └── config.py
│   ├── training/
│   │   ├── data_module.py
│   │   ├── train.py
│   │   └── finetune_lora.py
│   ├── inference/
│   │   ├── engine.py          # loads checkpoint, runs chunked inference
│   │   ├── onnx_export.py
│   │   └── tensorrt_build.py  # stretch goal
│   ├── server/
│   │   ├── main.py            # FastAPI + WebSocket streaming endpoint
│   │   └── schemas.py
│   ├── pipecat_plugin/
│   │   └── hams_tts_service.py
│   └── eval/
│       ├── benchmark.py       # TTFA / RTF / VRAM, runnable standalone
│       ├── objective_metrics.py # CER/WER via Whisper, UTMOS/NISQA/DNSMOS
│       └── mos_survey.py      # listening-test harness + aggregation
├── scripts/
│   ├── prepare_data.sh
│   ├── run_training.sh
│   ├── run_benchmark.sh
│   └── export_samples.sh
├── tests/
├── samples/                   # final 3 deliverable WAVs
└── docs/
    └── design_doc.md          # the 3–6 page final write-up deliverable
```

This isn't created yet — Phase 0 below scaffolds it.

## 2. Phased Build Sequence

### Phase 0 — Environment & Scaffolding
- Set up `pyproject.toml` / `uv` or `poetry` env; pin PyTorch + CUDA matching
  local driver (CUDA 13.0 per `nvidia-smi`).
- Scaffold the directory tree above with empty modules + `__init__.py`.
- Set up experiment tracking (Weights & Biases or plain CSV logging — keep
  lightweight given project scope).
- Create Hugging Face repo (model) and decide public vs private (open
  question in PRD §10).
- **Exit criteria**: `python -c "import torch; print(torch.cuda.is_available())"`
  returns True; repo skeleton committed.

### Phase 1 — Text Pipeline (FR1)
1. Implement `lang_segment.py`: Unicode-block-based splitter producing
   `(text_span, lang_tag)` sequences from raw mixed text; handle embedded
   Latin loanwords inside Arabic (e.g. brand names) as a known edge case
   documented, not silently misclassified.
2. Implement `normalize.py`: expand numbers, dates (Gregorian + common Hijri
   forms), currency (USD/SAR/SYP/JOD symbols → spoken forms) per language
   span, normalize code-mixed punctuation (Arabic vs Latin comma/question
   mark).
3. Implement `diacritize.py`: wrap CAMeL Tools' diacritization model (or
   Mishkal/Shakkala as fallback) for undiacritized Arabic spans; pass through
   already-diacritized input unchanged.
4. Implement `g2p.py`: run `espeak-ng` with `ar` voice on diacritized Arabic
   spans and `en-us` on English spans, normalize both outputs into one IPA
   symbol set.
5. Implement `levantine_rules.py`: post-process espeak-ng's MSA-oriented
   Arabic phonemes with the documented Levantine substitution rules (ق→ʔ,
   ج→ʒ, vowel reduction patterns) and a manually curated dialect lexicon
   override table (grows from eval failures in Phase 3).
6. Concatenate spans with language-ID tags, no inserted silence at boundaries
   purely from script switching (punctuation-driven pauses only).

   **Implementation note:** diacritization uses `mishkal`, not CAMeL Tools —
   CAMeL Tools' diacritization models download separately via its
   `camel_data` fetcher from a non-HF host we couldn't guarantee is always
   reachable; `mishkal` ships its data in the pip wheel itself, which is
   lower operational risk for a pipeline that needs to run unattended. See
   `hams_tts/text/diacritize.py` docstring. espeak-ng itself isn't
   system-installed (no root on the dev box) — see `AGENTS.md`
   "Environment" for how it was vendored.
7. **Exit criteria**: pipeline turns 20 hand-picked mixed sentences
   (including numbers/dates/currency/code-switch) into phoneme sequences that
   a native/fluent reviewer confirms look correct; limitations written down
   (OOV dialect words, ambiguous loanword classification, etc.) — this
   becomes the "Data & phonemization strategy" section of the design doc.

### Phase 2 — Dataset Assembly
1. Arabic monolingual: ClArTTS (clean, diacritized, good alignment signal)
   + Arabic Speech Corpus (ASC — Syrian-accented MSA, phonetically
   annotated, closest public match to Levantine realization).
2. English monolingual: LJSpeech (single-speaker baseline) + LibriTTS
   (multi-speaker diversity, feeds the speaker-embedding space).
3. Levantine supplementary audio: select the cleanest Levantine-heavy subset
   from available Arabic ASR corpora (evaluate QASR / Casablanca / MGB
   variants for license + audio quality; substitute if access friction is
   high, per PRD §8), force-align with Whisper-based alignment to get
   clean utterance-level clips + transcripts.
4. Code-switched augmentation (no authentic public corpus exists):
   - Text-level: template + rule-based insertion of common Levantine
     code-switch patterns (tech/brand nouns, discourse markers, numbers) into
     monolingual sentences.
   - Audio-level bootstrap: synthesize candidates with the in-progress
     fine-tuned model, filter by ASR round-trip CER + reference-free quality
     score, feed high-confidence outputs back into fine-tuning data
     (self-training loop) — run this *after* Phase 3's first fine-tuned
     checkpoint exists, not before.
   - Reserve a small set of real or semi-authentic code-switched clips
     (if obtainable) strictly for the held-out eval set, never for training,
     to keep evaluation honest.
5. Build `data/manifests/{train,val,held_out_eval}.jsonl` with
   `{audio_path, text, lang_tags, speaker_id, duration}`.
6. **Exit criteria**: manifests exist with target composition roughly
   40% AR / 40% EN / 20% code-switched by duration; held-out eval set has
   10–20 utterances spanning all three categories per FR4.

   **Implementation note — what was actually used:** HF-hosted parquet
   mirrors rather than the Hub's original script-based dataset repos
   (`datasets` is deprecating arbitrary loading-script execution, and
   script-based repos like `keithito/lj_speech` / `halabi2016/arabic_speech_corpus`
   would have added that dependency for no benefit over a parquet mirror
   with the same content):
   - `MBZUAI/ClArTTS` (official parquet mirror; 9500 train / 205 test,
     already diacritized) — capped at 1500 training utterances.
   - `tunis-ai/arabic_speech_corpus` (parquet mirror of Nawar Halabi's
     ASC; 1813 train / 100 test) — all of it used. Its transcripts are
     Buckwalter transliteration, not Arabic script; converted via
     `hams_tts/training/data_prep/buckwalter.py` (a standard reversible
     mapping table) before going through the normal pipeline. Bonus:
     Buckwalter is a lossless transliteration of fully-vocalized text, so
     this conversion recovers complete tashkeel for free — cleaner input
     than running `mishkal` on it.
   - `bookbot/ljspeech_phonemes` (parquet mirror of LJSpeech; 13100 rows)
     — capped at 1700 training utterances, using its `normalized_text`
     column (not its bundled `phonemes` column — we run our own
     phonemization pipeline on it instead, so English and Arabic go
     through one consistent code path).

   Caps bound preprocessing/training wall-clock for this project; full
   corpora remain on disk under `data/raw/` for a larger follow-up run —
   see design doc limitations. This produces an AR-heavier split (~3300 AR
   / ~1700 EN by utterance count) than the original 40/40/20 target, and
   **no code-switched utterances made it into the training set at all** —
   see the code-switch note below, this is a real, not cosmetic, gap.

   Real audio-text pairs for code-switched *training* data don't exist
   anywhere (that's the whole premise of PRD's risk table) and the
   self-training bootstrap loop needs a first checkpoint to exist before
   it can run — so round 1 training is monolingual AR + monolingual EN
   only, and code-switching capability has to come from the model
   generalizing across a shared phoneme vocabulary + language-marker
   tokens it learned from monolingual data alone, not from ever having
   seen matched code-switched audio. The held-out eval set
   (`data/manifests/held_out_eval.jsonl`, 18 hand-written utterances: 6
   pure AR / 6 pure EN / 6 code-switched) is text-only and doesn't need
   matching reference audio — FR4's metrics (ASR round-trip CER/WER,
   reference-free quality, MOS) all evaluate generated audio against the
   input text or human judgment, not against a ground-truth recording, so
   this is a legitimate design point, not a shortcut.

### Phase 3 — Fine-Tuning (FR3)
1. Load pretrained VITS/MMS-TTS-ara + LJSpeech/LibriTTS VITS checkpoints as
   initialization (two base checkpoints merged at the encoder/embedding
   level, or sequential fine-tune — decide based on what's actually
   compatible once checkpoints are in hand).
2. Add language-ID embedding + speaker embedding table sized to the
   training speaker set.
3. Attach LoRA adapters to text encoder + duration predictor; freeze base
   weights initially, fine-tune adapters + new embeddings on the assembled
   corpus.
4. Document: data composition %, batch size, learning rate schedule, LoRA
   rank, number of steps, loss curves, any vocabulary/BPE changes.
5. **Exit criteria**: checkpoint produces intelligible pure-AR, pure-EN, and
   code-switched speech on a few spot-check sentences (informal listen before
   moving to formal Phase 4 evaluation).

### Phase 4 — Evaluation (FR4)
1. Run held-out eval set through the checkpoint (in-process, not yet the
   server) to get a first objective metrics pass: RTF, peak VRAM, CER/WER via
   Whisper large-v3, UTMOS/NISQA/DNSMOS.
2. Fix obvious regressions (e.g., specific Levantine tokens mispronounced →
   extend `levantine_rules.py` lexicon override table and re-run just those
   cases).
3. Run the MOS-style listening test: n ≥ 5 listeners rate naturalness and
   code-switch smoothness on the held-out set; capture free-text
   pronunciation notes per utterance.
4. **Exit criteria**: full metrics table produced; this is the raw data for
   the design doc's benchmark results section (actual TTFA/p95 numbers come
   from Phase 5's server-level benchmark, not this in-process pass).

### Phase 5 — Optimization & Deployment (FR2 justification + FR5)
1. Export checkpoint to ONNX (dynamic axes for variable phoneme-sequence
   length); verify numerical parity against the PyTorch model on a few
   samples.
2. Convert to FP16; evaluate INT8 dynamic quantization as a stretch step,
   A/B against UTMOS/CER before adopting (quantization that measurably hurts
   intelligibility is not worth the VRAM savings here since the budget is
   already generous for a ~30-90M param model).
3. (Stretch) Build a TensorRT engine via `trtexec` for the ONNX graph on the
   target GPU; fall back to ONNX Runtime CUDA execution provider if this
   doesn't fit the time budget — either satisfies FR2's optimization
   justification requirement.
4. Implement `server/main.py`: FastAPI app, WebSocket `/tts/stream` endpoint,
   clause-boundary chunking (short first chunk to minimize TTFA, larger
   subsequent chunks for throughput), PCM16 output with Opus as an optional
   encode path, input validation (length caps, script allow-list), structured
   error responses, `/health` endpoint.
5. Implement `eval/benchmark.py`: drives the *running server* (not just the
   in-process model) end-to-end, measures wall-clock TTFA (p50/p95 over many
   runs), RTF, and samples peak VRAM via CUDA memory stats during the run;
   outputs a markdown table shaped exactly like the PRD §6 KPI table.
6. ~~Run the benchmark on both the local RTX 4070 Ti (dev numbers) and a
   rented L4 instance (official numbers)~~ — **revised 2026-11-23**: no L4
   access exists (no cloud credentials/budget), and the user explicitly
   decided to cite the RTX 4070 Ti server-mode benchmark as final rather
   than block on renting one. Judged acceptable given the small model
   size (46.7M params) and the wide margins measured (RTF ~23x under
   target) — see `docs/design_doc.md` §4 and `PRD.md` §8 for the full
   reasoning. Don't silently re-add an L4 requirement without the user
   re-opening this.
7. Implement the Pipecat plugin: subclass `TTSService`, implement
   `async run_tts(text) -> AsyncGenerator[Frame, None]` calling the inference
   engine in-process (bypassing the HTTP server to avoid double latency),
   emit `TTSAudioRawFrame` per chunk with correct sample-rate metadata,
   respect Pipecat's interruption/start-stop frame conventions — mirror the
   structure of an existing in-tree Pipecat TTS service rather than
   inventing a new shape.
8. **Exit criteria**: `scripts/benchmark.py --mode server` run against the
   deployed server (RTX 4070 Ti — final hardware, see step 6 above)
   produces a KPI table; a minimal Pipecat pipeline using the plugin runs
   end to end locally. Both done — see `docs/design_doc.md` §4.

### Phase 6 — Final Packaging (Deliverables)
1. Push final checkpoint + ONNX export to Hugging Face Hub with a model card
   (architecture, training data summary, intended use, known limitations).
2. Generate and save the 3 required sample WAVs (pure AR, pure EN,
   code-switched) from the final deployed checkpoint into `samples/`.
3. Record the ≤3-minute demo video: live text input → streamed audio via the
   running server, including one code-switched example, plus a quick cut
   showing the Pipecat pipeline working.
4. Write `docs/design_doc.md` (3–6 pages): data & phonemization strategy
   (from Phase 1 notes), model & optimization justification (from §0 of this
   plan + Phase 5 results), training/fine-tuning approach (Phase 3 notes),
   benchmark results table vs. targets (Phase 5 RTX 4070 Ti numbers —
   final, per the revised step 6 above, not L4), known
   limitations and what more time would buy (data scarcity, dialect
   coverage, TensorRT stretch goal status, etc.).
5. Finalize `README.md` with setup instructions covering: environment setup,
   data download/prep, running training, running the server, running the
   benchmark, running the Pipecat example.
6. **Exit criteria**: every item in PRD §7 "Deliverables" has a concrete
   artifact and a path/link recorded in the README.

## 3. Sequencing Notes

- Phases 1 and 2 can proceed in parallel (text pipeline work doesn't block on
  dataset downloads, and vice versa).
- Phase 3 cannot start meaningfully until Phase 2 produces manifests and
  Phase 1 produces a phonemizer to generate training targets from — they
  converge before fine-tuning begins.
- The self-training code-switch augmentation sub-step inside Phase 2 is
  intentionally sequenced *after* a first Phase 3 checkpoint exists (it needs
  a model to bootstrap from) — expect to loop back into Phase 2 once during
  Phase 3.
- Phase 5's optimization work can start as soon as *any* checkpoint exists
  (even before fine-tuning is "done") since ONNX export / streaming server
  plumbing doesn't depend on final model quality — parallelize this with
  Phase 3/4 to save calendar time if needed.

## 4. Next Action

Phase 0: scaffold the repo structure in §1 and pin the Python/CUDA
toolchain. Say the word and this can be scaffolded immediately.
