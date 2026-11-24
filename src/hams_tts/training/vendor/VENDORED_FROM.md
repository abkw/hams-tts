# Vendored from ylacombe/finetune-hf-vits

Source: https://github.com/ylacombe/finetune-hf-vits (MIT License, see LICENSE in this directory)
Vendored commit: see /tmp clone date 2026-11-22 (shallow clone, HEAD at time of vendoring)

This directory contains an unmodified-or-lightly-adapted copy of the VITS/MMS
GAN fine-tuning recipe by Yoach Lacombe. We did not reimplement VITS's
training objective (discriminator loss, flow-based NLL, stochastic duration
predictor loss, feature-matching loss) from scratch — that loop is commodity
code once the architecture decision (PLAN.md §0: VITS/MMS family) is made,
and reimplementing it from the paper would have bought nothing over reusing
a well-tested community recipe. Our actual contribution sits around this:
the Levantine phonemization front-end (`hams_tts.text`), the custom phoneme
vocabulary + language-marker tokens fed into it (`hams_tts.text.vocab`),
dataset assembly, evaluation harness, streaming server, and Pipecat plugin.

Local adaptations on top of the upstream recipe are noted inline with
`# HAMS:` comments where they diverge from upstream behavior.
