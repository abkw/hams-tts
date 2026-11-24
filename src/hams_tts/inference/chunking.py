"""Split raw input text into clause-sized chunks for streaming synthesis.

VITS (and MMS-TTS) has no native incremental/causal decoding mode — a
single forward pass turns one full input sequence into one full waveform.
"Streaming" a VITS-family model therefore means splitting text into small
pieces *before* synthesis and emitting each piece's audio as soon as it's
ready, not a single model producing audio incrementally mid-sequence. This
is the same approach Piper and most VITS-based streaming TTS servers use.

Consequence worth documenting (design doc limitations): prosody is not
continuous across a chunk boundary the way it would be if a single
sequence were synthesized whole — each chunk is an independent forward
pass. Keeping the first chunk short directly trades off against this: a
short first clause minimizes Time-to-First-Audio but is also the chunk
most likely to sound clipped/independent from what follows.
"""

from __future__ import annotations

import re

_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[.!?؟!,،;؛:])\s+")

_DEFAULT_MAX_CHARS = 120
_FIRST_CHUNK_MAX_CHARS = 40


def split_into_chunks(
    text: str,
    first_chunk_max_chars: int = _FIRST_CHUNK_MAX_CHARS,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[str]:
    """Greedy clause-boundary chunking with a short first chunk (TTFA) and
    larger subsequent chunks (throughput), per PLAN.md Phase 5."""
    text = text.strip()
    if not text:
        return []

    clauses = [c.strip() for c in _CLAUSE_BOUNDARY_RE.split(text) if c.strip()]
    if not clauses:
        return [text]

    chunks: list[str] = []
    current = ""
    limit = first_chunk_max_chars

    for clause in clauses:
        candidate = f"{current} {clause}".strip() if current else clause
        if current and len(candidate) > limit:
            chunks.append(current)
            current = clause
            limit = max_chars
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
