"""Pydantic models shared by `hams_tts.server.main`.

See `main.py`'s module docstring for the full WebSocket wire protocol. This
module only defines the message shapes; `main.py` owns validation logic
(it needs custom rules beyond type-checking, e.g. max length, that are
easiest to apply uniformly across the HTTP and WebSocket entry points).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_TEXT_LENGTH = 2000


class TTSRequest(BaseModel):
    """Request body for POST /tts and the JSON message sent over /tts/stream."""

    text: str = Field(..., description="Text to synthesize; mixed AR/EN allowed.")
    lang_hint: Literal["ar", "en"] = Field(
        "ar",
        description="Default language for text spans without an explicit language tag.",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["ok"] = "ok"
    model_loaded: bool
    device: str | None = None
    sampling_rate: int | None = None
    load_error: str | None = None


# --- WebSocket server -> client message shapes (for documentation / tests) ---
# These are not used to validate outgoing messages (we build dicts directly
# in main.py to avoid an extra serialization round-trip per chunk), but they
# document the exact contract a client should expect.


class StreamStart(BaseModel):
    type: Literal["stream_start"] = "stream_start"
    sampling_rate: int
    channels: int = 1


class ChunkMetadata(BaseModel):
    type: Literal["chunk"] = "chunk"
    chunk_index: int
    is_final: bool
    sampling_rate: int
    channels: int = 1
    num_samples: int
    gen_seconds: float


class StreamEnd(BaseModel):
    type: Literal["stream_end"] = "stream_end"


class StreamError(BaseModel):
    type: Literal["error"] = "error"
    error: str
