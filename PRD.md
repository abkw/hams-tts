# PRD — Low-Latency Code-Switching TTS for Levantine Arabic (Hams.AI)

Status: Draft v1
Owner: Mohammed
Source: Hams.AI project brief (kept outside this repo)
Last updated: 2026-11-22

## 1. Background & Problem Statement

Hams.AI needs a Text-to-Speech engine for real-time conversational AI agents that
can speak natural Levantine Arabic, natural English, and — critically — switch
fluidly between the two mid-utterance the way bilingual Levantine speakers
actually talk (tech terms, numbers, discourse markers in English dropped into
Arabic sentences and vice versa).

Two things make this hard simultaneously:

1. **Linguistic**: Arabic TTS tooling is built for Modern Standard Arabic (MSA).
   Levantine pronunciation diverges from MSA in specific, well-known ways (ق → /ʔ/,
   ج → /ʒ/, vowel reduction, dialectal lexicon), and almost no public corpus pairs
   Levantine dialect with natural English code-switching.
2. **Systems**: the model must run as a real-time conversational backend, which
   imposes hard latency and memory ceilings, not just "make it fast later."

This PRD scopes the project brief into a buildable project with clear
acceptance criteria, so implementation work (tracked in `PLAN.md`) has an
unambiguous definition of done.

## 2. Goals

- G1: A single TTS model that produces natural-sounding **Levantine Arabic**,
  **English**, and **code-switched** speech from raw mixed-script text.
- G2: Meets strict production inference KPIs on an NVIDIA L4-class GPU.
- G3: Ships as a **streaming** service, integrable into a real conversational
  agent via a **Pipecat** `TTSService` plugin — not just a notebook demo.
- G4: Every design decision (dataset, phonemizer, architecture, optimization)
  is justified in writing, not just implemented — engineering judgment
  matters here, not only code.

## 3. Non-Goals

- Not building a general-purpose multi-dialect Arabic TTS (MSA, Gulf, Egyptian,
  Maghrebi) — Levantine is the sole dialect target.
- Not building multi-speaker voice cloning / arbitrary zero-shot voice transfer.
  A small fixed set of speaker identities (from training data) is sufficient.
- Not targeting on-device / CPU-only / mobile inference. Target is a single
  GPU server process (L4-class).
- Not building a production auth/billing/multi-tenant API layer around the
  server — the FastAPI/WebSocket service is a reference implementation, not a
  hardened public-facing product.
- Not sourcing or licensing large proprietary Levantine speech corpora beyond
  what's reasonably obtainable (open corpora + ASR-repurposed + synthetic
  augmentation). Data scarcity is treated as a documented limitation, not a
  blocker.

## 4. Users & Use Case

Primary consumer: a **conversational AI agent** (Pipecat pipeline) that needs
to speak a response turn as soon as the first words of text are available,
with the ability to naturally code-switch when the underlying LLM response
does (e.g. mixing in English product names, numbers, or technical terms into
an Arabic reply).

## 5. Requirements

Carried over from the project brief, organized as functional requirements:

### FR1 — Text Processing Pipeline
- Normalize and phonemize mixed Arabic/Latin-script text: numbers, dates,
  currency, code-mixed punctuation.
- Automatic diacritization (tashkeel restoration) for undiacritized Arabic
  input.
- Produce a unified phoneme representation across Arabic and English with
  language-boundary tagging, with smooth prosody across switch points (no
  inserted silence/voice break purely due to script change).
- Apply Levantine-specific phonological rules on top of MSA-oriented tooling
  (ق→ʔ, ج→ʒ, vowel reduction, dialectal lexicon overrides).

### FR2 — Model
- A TTS architecture selected and justified against the KPI table (§6), with
  an explicit account of any quantization/pruning/ONNX/TensorRT optimization
  applied to hit the targets.
- Supports streaming (chunked) audio generation, not only whole-utterance
  synthesis.

### FR3 — Fine-tuning
- Adapts a base/pretrained model to Levantine pronunciation and code-switch
  smoothness using a documented technique (LoRA, language-ID embeddings,
  speaker conditioning, vocabulary changes, etc.), with training setup and
  data composition recorded.

### FR4 — Evaluation
- A held-out set of 10–20 utterances spanning pure Arabic, pure English, and
  code-switched content.
- Required objective metrics: RTF, TTFA (p50, p95), peak VRAM, ASR round-trip
  CER/WER (Whisper large-v3), one reference-free quality metric (UTMOS,
  NISQA, or DNSMOS).
- Required subjective metrics: small MOS-style listening test (n ≥ 5
  listeners where feasible) on naturalness and code-switch smoothness, plus
  qualitative notes on Levantine-specific pronunciation accuracy.

### FR5 — Deployment
- Streaming server (FastAPI / gRPC / WebSocket) accepting streaming/chunked
  text and returning low-TTFA audio chunks (PCM or Opus), with input
  validation and error handling.
- A reproducible, runnable benchmark script measuring TTFA/RTF/peak VRAM
  against target hardware.
- A Pipecat `TTSService` subclass following existing plugin conventions
  (`run_tts`, `TTSAudioRawFrame` emission, etc.).
- A ≤3-minute demo video of live text → streamed audio, including at least
  one code-switched example.

## 6. Success Metrics (Acceptance Criteria)

| Metric | Target | Measured by |
|---|---|---|
| Peak inference VRAM | ≤ 3 GB on L4 / RTX-3090-class GPU | `benchmark.py`, `nvidia-smi` / CUDA memory stats |
| Time-to-First-Audio (TTFA) | < 300 ms (p50 and p95 reported) | `benchmark.py` against running server |
| Real-Time Factor (RTF) | < 0.3 | `benchmark.py` |
| Streaming output | Required, chunked | Server integration test |
| Intelligibility | CER/WER reported via Whisper large-v3 round-trip, no fixed pass/fail threshold but must be competitive with reference TTS baselines | `eval/objective_metrics.py` |
| Reference-free quality | UTMOS or NISQA or DNSMOS reported | `eval/objective_metrics.py` |
| Subjective naturalness / code-switch smoothness | MOS-style rating, n ≥ 5 listeners, reported with distribution not just mean | `eval/mos_survey.py` + manual listening session |
| Pipecat integration | Plugin runs in a minimal Pipecat pipeline end-to-end | manual + scripted smoke test |

A model that misses one numeric KPI but documents *why* and *what quantified
gap remains* is considered an acceptable outcome — the brief explicitly asks
for justified trade-offs, not a black box that merely claims compliance.

## 7. Deliverables (from brief, unchanged)

1. Model weights on Hugging Face (link).
2. Code repository: training/fine-tuning code, inference code, streaming
   server, benchmark script, Pipecat plugin, README with setup instructions.
3. Three sample WAVs: pure Levantine Arabic, pure English, code-switched.
4. Demo video (≤ 3 min): streaming server + Pipecat integration in action.
5. Design doc (~3–6 pages): data & phonemization strategy; model &
   optimization choices with justification; training/fine-tuning approach;
   benchmark results vs. targets (table); known limitations and what more
   time would buy.

## 8. Constraints & Assumptions

- **Compute**: Dev/training **and final benchmark** both on the local RTX
  4070 Ti (12 GB VRAM). The brief names an L4-class GPU as the target;
  this project has no L4 access (no cloud credentials, no approved
  budget to rent one), and the user explicitly decided (2026-11-23) to
  cite the RTX 4070 Ti numbers as final rather than block on L4 access.
  This is judged acceptable because: (a) the model is small (46.7M
  params) relative to either GPU's capacity, making the two GPUs'
  differing memory-bandwidth/compute profiles unlikely to flip any KPI
  from pass to fail; (b) RTX 4070 Ti is a comparable-or-stronger consumer
  GPU in the same general class the brief's own KPI table names as
  acceptable ("L4 / RTX-3090-class GPU", see §6); and (c) the margins
  measured are wide (RTF ~23x under target, not a near-miss that would
  be sensitive to hardware choice). See `docs/design_doc.md` §4 for the
  actual numbers and this same reasoning restated there.
- **Timeline**: no hard deadline was stated in the brief or by the user as of
  this writing. The plan is sequenced by dependency, not calendar date;
  revisit and compress phases if a deadline is given.
- **Data licensing**: some referenced corpora (e.g., LDC-hosted Arabic ASR
  sets) require registration/agreements. Where access friction is high within
  the available time, substitute the closest permissively-licensed
  alternative and document the substitution — do not block the project on
  acquiring a specific named corpus.
- **Authentic Levantine code-switched speech is scarce.** The plan treats
  synthetic/augmented code-switched training data as the primary lever, with
  a small curated real/semi-real set reserved for evaluation only.

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| No public corpus of real Levantine EN/AR code-switched speech | Model's switch transitions may sound synthetic | Synthetic text-level augmentation + self-training bootstrap loop (generate → filter by ASR/quality metrics → add back); real eval-only mini-set |
| MSA-tuned diacritizers mishandle Levantine lexicon | Mispronounced dialect words | Maintain a manual exception/pronunciation lexicon for known Levantine tokens, expand iteratively from eval failures |
| TTFA/RTF/VRAM targets conflict with naturalness | Must pick an architecture biased toward speed | Default to a non-autoregressive (VITS-family) architecture; treat higher-naturalness autoregressive models as a documented rejected alternative |
| Local GPU (4070 Ti) isn't the brief's named target GPU (L4) | Benchmark numbers might not transfer to L4 | **Resolved, not open**: no L4 access available; user decided (2026-11-23) to cite RTX 4070 Ti as final given the wide KPI margins and small model size (see §8) — not treated as a risk requiring further mitigation |
| Pipecat plugin conventions drift from what's expected | Integration deliverable marked incomplete | Base the subclass directly on an existing in-tree Pipecat TTS plugin's structure, not a from-scratch guess |

## 10. Open Questions for the User

1. Is there a hard deadline for this delivery?
2. Should model weights/code repo be public on Hugging Face/GitHub, or
   private with an invite (brief allows either)?
3. Any preference on Opus vs PCM for the streaming audio wire format, or
   defer to whatever's simplest to demo in Pipecat?

These don't block starting the plan — defaults are assumed in `PLAN.md` and
can be revised if answered.
