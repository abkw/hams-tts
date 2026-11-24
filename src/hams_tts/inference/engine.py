"""Loads a fine-tuned HamsTTS checkpoint and runs synthesis.

This is the one place that owns the model/tokenizer lifecycle — the
FastAPI server, the benchmark script, and the Pipecat plugin all call into
an `Engine` instance rather than each loading the model themselves, so
there's exactly one inference code path to keep fast and correct.

Streaming note: see `chunking.py` docstring — VITS has no incremental
decode mode, so "streaming" here means synthesizing short text chunks
back-to-back and yielding each one's audio as soon as it's ready.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from hams_tts.inference.chunking import split_into_chunks
from hams_tts.text.pipeline import phonemize_text


@dataclass
class AudioChunk:
    audio: np.ndarray  # float32, mono, at `sampling_rate`
    sampling_rate: int
    chunk_index: int
    is_final: bool
    gen_seconds: float  # wall-clock time this chunk took to synthesize


@dataclass
class SynthesisStats:
    ttfa_seconds: float
    total_gen_seconds: float
    audio_seconds: float
    rtf: float
    peak_vram_bytes: int | None


class Engine:
    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_dir: str | Path | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        speaker_id: int | None = None,
    ) -> None:
        from transformers import VitsModel

        from hams_tts.text.vocab import load_tokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.speaker_id = speaker_id

        tokenizer_dir = tokenizer_dir or model_dir
        self.tokenizer = load_tokenizer(tokenizer_dir)

        self.model = VitsModel.from_pretrained(str(model_dir), dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

        self.sampling_rate = self.model.config.sampling_rate

    @torch.inference_mode()
    def _synthesize_chunk(self, phoneme_string: str) -> np.ndarray:
        inputs = self.tokenizer(phoneme_string, return_tensors="pt").to(self.device)
        kwargs = {}
        if self.speaker_id is not None and getattr(self.model.config, "num_speakers", 1) > 1:
            kwargs["speaker_id"] = torch.tensor([self.speaker_id], device=self.device)
        output = self.model(**inputs, **kwargs)
        waveform = output.waveform[0].detach().to(torch.float32).cpu().numpy()
        return waveform

    def synthesize(self, text: str, default_lang: str = "ar") -> tuple[np.ndarray, int]:
        """Non-streaming: full utterance in one call. Used for sample WAVs
        and as the ground truth RTF measurement in the benchmark script."""
        phoneme_string = phonemize_text(text, default_lang=default_lang).phoneme_string
        audio = self._synthesize_chunk(phoneme_string)
        return audio, self.sampling_rate

    def synthesize_streaming(
        self,
        text: str,
        default_lang: str = "ar",
    ) -> Iterator[AudioChunk]:
        """Chunked synthesis, yielding audio as soon as each chunk is ready.
        The caller (server/benchmark/Pipecat plugin) can measure TTFA as the
        wall-clock time to the first yielded `AudioChunk`."""
        text_chunks = split_into_chunks(text)
        if not text_chunks:
            return

        for i, chunk_text in enumerate(text_chunks):
            start = time.perf_counter()
            phoneme_string = phonemize_text(chunk_text, default_lang=default_lang).phoneme_string
            if not phoneme_string.strip():
                continue
            audio = self._synthesize_chunk(phoneme_string)
            gen_seconds = time.perf_counter() - start
            yield AudioChunk(
                audio=audio,
                sampling_rate=self.sampling_rate,
                chunk_index=i,
                is_final=(i == len(text_chunks) - 1),
                gen_seconds=gen_seconds,
            )

    def synthesize_streaming_with_stats(
        self, text: str, default_lang: str = "ar"
    ) -> tuple[list[AudioChunk], SynthesisStats]:
        """Convenience wrapper for the benchmark script: drains the stream
        and returns TTFA/RTF/VRAM in one call."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        chunks: list[AudioChunk] = []
        start = time.perf_counter()
        ttfa = None
        for chunk in self.synthesize_streaming(text, default_lang=default_lang):
            if ttfa is None:
                ttfa = time.perf_counter() - start
            chunks.append(chunk)
        total_gen = time.perf_counter() - start

        total_audio_samples = sum(len(c.audio) for c in chunks)
        audio_seconds = total_audio_samples / self.sampling_rate if chunks else 0.0
        rtf = (total_gen / audio_seconds) if audio_seconds > 0 else float("nan")
        peak_vram = torch.cuda.max_memory_allocated(self.device) if self.device == "cuda" else None

        stats = SynthesisStats(
            ttfa_seconds=ttfa or 0.0,
            total_gen_seconds=total_gen,
            audio_seconds=audio_seconds,
            rtf=rtf,
            peak_vram_bytes=peak_vram,
        )
        return chunks, stats
