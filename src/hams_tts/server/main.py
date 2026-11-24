"""FastAPI streaming TTS server wrapping `hams_tts.inference.engine.Engine`.

Per PRD FR5 / PLAN.md Phase 5 step 4.

Model path configuration
-------------------------
The fine-tuned checkpoint is produced by a training run that may still be in
progress, so the model directory is environment-driven rather than hardcoded:

    HAMS_MODEL_DIR      Directory passed to `Engine(model_dir=...)`.
                         Default: "data/processed/checkpoint".
    HAMS_TOKENIZER_DIR  Tokenizer directory. Default: same as HAMS_MODEL_DIR.
    HAMS_DEVICE         "cuda" or "cpu". Default: auto-detected by Engine.
    HAMS_SPEAKER_ID     Optional integer speaker id for multi-speaker checkpoints.

If `HAMS_MODEL_DIR` doesn't exist (or loading otherwise fails), the server
still starts and serves `/health` — it does NOT crash the process. Every
endpoint that needs the model responds with a clear 503 (HTTP) or a
`{"type": "error", ...}` message (WebSocket) referencing the load error
until a checkpoint becomes available, at which point the next request
retries the load without needing a server restart.

WebSocket contract (`/tts/stream`)
-----------------------------------
This endpoint accepts exactly one message shape — a JSON text frame:

    {"text": "<string, 1-2000 chars>", "lang_hint": "ar" | "en"}

`lang_hint` is optional (default "ar") and is passed as `default_lang` to
`Engine.synthesize_streaming` for any text spans that aren't otherwise
language-tagged. Binary frames or non-JSON text from the client are
rejected with a structured error (see below) — this server deliberately
does not also accept a "plain text frame" mode, to avoid an ambiguous
dual-mode contract.

For each accepted message the server streams back, interleaving one JSON
"header" message per audio chunk with one binary frame carrying that
chunk's raw audio, so a client can always pair the frame it just received
with the metadata that describes it:

    -> {"type": "stream_start", "sampling_rate": 16000, "channels": 1}
    -> {"type": "chunk", "chunk_index": 0, "is_final": false,
        "sampling_rate": 16000, "channels": 1, "num_samples": 7040,
        "gen_seconds": 0.031}
    -> <binary frame: raw PCM16, little-endian, mono audio for that chunk>
    -> {"type": "chunk", "chunk_index": 1, "is_final": true, ...}
    -> <binary frame>
    -> {"type": "stream_end"}

On any problem, instead of (or in addition to) the above:

    -> {"type": "error", "error": "<human readable message>"}

Input-validation errors (empty text, text over 2000 chars, malformed JSON,
non-string `text`, bad `lang_hint`) send an error message and leave the
connection open so the client can retry with a corrected message on the
same socket. An exception raised *during* synthesis sends an error message
and then closes the connection (code 1011) — a partially streamed
utterance can't be resumed mid-stream — but this never affects other
concurrent connections or the server process.

One socket may be reused for multiple utterances sequentially: send another
`{"text": ...}` message any time after a `stream_end`.

A plain, non-streaming `POST /tts` (JSON body, same shape as above) is also
provided, returning a complete WAV file — handy for `curl`/manual testing
without a WebSocket client. It is secondary to the streaming endpoint.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from hams_tts.inference.engine import AudioChunk, Engine
from hams_tts.server.schemas import MAX_TEXT_LENGTH, HealthResponse, TTSRequest

logger = logging.getLogger("hams_tts.server")

DEFAULT_MODEL_DIR = "data/processed/checkpoint"

_SENTINEL = object()


def _env_model_dir() -> Path:
    return Path(os.environ.get("HAMS_MODEL_DIR", DEFAULT_MODEL_DIR))


def _env_tokenizer_dir() -> Path | None:
    value = os.environ.get("HAMS_TOKENIZER_DIR")
    return Path(value) if value else None


def _env_speaker_id() -> int | None:
    value = os.environ.get("HAMS_SPEAKER_ID")
    return int(value) if value else None


def load_engine(app: FastAPI) -> None:
    """Attempt to construct the Engine and store it (or a clear error) on
    `app.state`. Never raises — a missing/broken checkpoint must not take
    down the server process; it should just mean "/health reports degraded
    and synthesis endpoints 503 until it's fixed"."""
    model_dir = _env_model_dir()

    if not model_dir.exists():
        msg = (
            f"HAMS_MODEL_DIR='{model_dir}' does not exist. The server will keep "
            "running and /health will report model_loaded=false until a trained "
            "checkpoint is present there (training may still be in progress) — "
            "no restart is required, the next request will retry the load."
        )
        logger.warning(msg)
        app.state.engine = None
        app.state.load_error = msg
        return

    try:
        engine = Engine(
            model_dir=model_dir,
            tokenizer_dir=_env_tokenizer_dir(),
            device=os.environ.get("HAMS_DEVICE"),
            speaker_id=_env_speaker_id(),
        )
    except Exception as exc:  # noqa: BLE001 - startup must never crash the process
        msg = f"Failed to load Engine from '{model_dir}': {exc}"
        logger.exception(msg)
        app.state.engine = None
        app.state.load_error = msg
        return

    app.state.engine = engine
    app.state.load_error = None
    logger.info(
        "Engine loaded from %s (device=%s, sampling_rate=%d)",
        model_dir,
        engine.device,
        engine.sampling_rate,
    )


def get_engine(app: FastAPI) -> Engine | None:
    """Return the loaded engine, retrying the load if it wasn't ready before.

    Cheap when the model dir still doesn't exist (just a stat call), so this
    is safe to call on every request without hammering disk/GPU init.
    """
    if app.state.engine is None:
        load_engine(app)
    return app.state.engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = None
    app.state.load_error = None
    logger.info("hams_tts server starting; attempting to load Engine...")
    load_engine(app)
    yield
    app.state.engine = None


app = FastAPI(title="Hams.AI TTS Server", lifespan=lifespan)

# Permissive by design: this is a demo/dev server meant to be driven from a
# locally-opened HTML test client (scripts/websocket_test_client.html,
# served as a file:// page) as well as arbitrary Pipecat/other deployments.
# A production deployment in front of a real browser app should restrict
# this to known origins instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_text_payload(text: object, lang_hint: object) -> TTSRequest:
    """Shared validation for both the HTTP and WebSocket entry points.

    Raises ValueError with a human-readable message on anything invalid.
    """
    if not isinstance(text, str):
        raise ValueError("`text` must be a string")
    if not text.strip():
        raise ValueError("`text` must not be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"`text` exceeds the max length of {MAX_TEXT_LENGTH} characters")

    if lang_hint is None:
        lang_hint = "ar"
    if lang_hint not in ("ar", "en"):
        raise ValueError("`lang_hint` must be 'ar' or 'en'")

    return TTSRequest(text=text, lang_hint=lang_hint)


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    """float32 mono [-1, 1] -> little-endian int16 PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    return pcm16.tobytes()


async def _iter_chunks_async(
    engine: Engine, text: str, lang: str
) -> AsyncIterator[AudioChunk]:
    """Drive the Engine's blocking `synthesize_streaming` generator one
    `next()` call at a time via `asyncio.to_thread`, so each (potentially
    slow) chunk synthesis doesn't block the whole event loop and other
    concurrent WebSocket connections stay responsive in between chunks."""
    gen = engine.synthesize_streaming(text, default_lang=lang)

    def _next() -> AudioChunk | object:
        try:
            return next(gen)
        except StopIteration:
            return _SENTINEL

    while True:
        item = await asyncio.to_thread(_next)
        if item is _SENTINEL:
            return
        yield item  # type: ignore[misc]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Reports whether the model is actually loaded, not just that the
    process is up."""
    engine: Engine | None = app.state.engine
    return HealthResponse(
        model_loaded=engine is not None,
        device=engine.device if engine else None,
        sampling_rate=engine.sampling_rate if engine else None,
        load_error=app.state.load_error,
    )


@app.post("/tts")
async def tts_http(request: TTSRequest) -> StreamingResponse:
    """Non-streaming helper: synthesizes the full utterance and returns a WAV
    file. Convenient for `curl`/manual testing; the WebSocket endpoint above
    is the primary low-TTFA path this service exists for."""
    try:
        req = _validate_text_payload(request.text, request.lang_hint)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    engine = get_engine(app)
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {app.state.load_error}")

    try:
        audio, sampling_rate = await asyncio.to_thread(engine.synthesize, req.text, req.lang_hint)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Synthesis failed for /tts request")
        raise HTTPException(status_code=500, detail="Synthesis failed") from exc

    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


async def _send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "error": message})


async def _stream_utterance(websocket: WebSocket, engine: Engine, req: TTSRequest) -> None:
    await websocket.send_json(
        {"type": "stream_start", "sampling_rate": engine.sampling_rate, "channels": 1}
    )
    async for chunk in _iter_chunks_async(engine, req.text, req.lang_hint):
        pcm = _pcm16_bytes(chunk.audio)
        await websocket.send_json(
            {
                "type": "chunk",
                "chunk_index": chunk.chunk_index,
                "is_final": chunk.is_final,
                "sampling_rate": chunk.sampling_rate,
                "channels": 1,
                "num_samples": int(len(chunk.audio)),
                "gen_seconds": chunk.gen_seconds,
            }
        )
        await websocket.send_bytes(pcm)
    await websocket.send_json({"type": "stream_end"})


@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                return
            except Exception:  # noqa: BLE001 - malformed JSON / wrong frame type
                await _send_error(
                    websocket,
                    'Expected a JSON text message of the form '
                    '{"text": "...", "lang_hint": "ar"|"en"}',
                )
                continue

            if not isinstance(message, dict):
                await _send_error(websocket, "Top-level JSON message must be an object")
                continue

            try:
                req = _validate_text_payload(message.get("text"), message.get("lang_hint"))
            except ValueError as exc:
                await _send_error(websocket, str(exc))
                continue

            engine = get_engine(app)
            if engine is None:
                await _send_error(websocket, f"Model not loaded: {app.state.load_error}")
                continue

            try:
                await _stream_utterance(websocket, engine, req)
            except WebSocketDisconnect:
                return
            except Exception as exc:  # noqa: BLE001 - keep the server/other conns alive
                logger.exception("Synthesis failed mid-stream")
                try:
                    await _send_error(websocket, f"Synthesis failed: {exc}")
                finally:
                    try:
                        await websocket.close(code=1011)
                    except Exception:  # noqa: BLE001
                        pass
                return
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - never let one bad connection kill the process
        logger.exception("Unexpected error in /tts/stream handler")
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass
