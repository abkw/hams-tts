"""Pipecat `TTSService` plugin wrapping `hams_tts.inference.engine.Engine`.

Per PRD FR5 / PLAN.md Phase 5 step 7: mirrors the structure of an in-tree
Pipecat TTS service rather than inventing new conventions. The closest
match in the installed `pipecat-ai==1.16.0` package is
`pipecat.services.piper.tts.PiperTTSService` — also a *local*, in-process
VITS-family model (Piper is VITS-based) rather than a cloud API, so its
pattern is what this file mirrors:

  - `super().__init__(push_start_frame=True, push_stop_frames=True, ...)`
    so the base class handles `TTSStartedFrame`/`TTSStoppedFrame` and audio
    context creation — the subclass only needs to yield audio frames.
  - `run_tts` wraps the engine's blocking/sync generator in an async
    iterator via `asyncio.to_thread`, then hands that to the base class's
    `_stream_audio_frames_from_iterator` helper, which turns raw PCM bytes
    into `TTSAudioRawFrame`s (handles odd-byte-count alignment, etc.) —
    exactly what `PiperTTSService.run_tts` and the FastAPI server's
    `_iter_chunks_async` both do, just targeting Pipecat's frame type
    instead of a WebSocket message.

Unlike the FastAPI server (`hams_tts.server.main`), this plugin calls the
`Engine` in-process directly rather than over HTTP/WebSocket — going
through the server here would add a full network+serialization round trip
on top of an already-tight TTFA budget for no benefit, since Pipecat and
the engine run in the same process either way.

Real API vs. the brief's paraphrase: the brief describes "implement
run_tts, emit TTSAudioRawFrame" as if that's the whole surface. In the
actually-installed API, `run_tts` is still the one abstract method, but
`TTSService.__init__`'s `push_start_frame`/`push_stop_frames` flags (not
mentioned in the brief) are what let a subclass skip manually
constructing `TTSStartedFrame`/`TTSStoppedFrame`, and
`_stream_audio_frames_from_iterator` (an underscore-prefixed, undocumented
base-class helper, not part of the brief's description at all) is what
actually builds `TTSAudioRawFrame`s in every first-party local/streaming
provider — reimplementing that chunking/alignment logic by hand instead
would have diverged from how every other in-tree service does it.

Same env-var model-path convention as `hams_tts.server.main` (see that
module's docstring) — duplicated here rather than imported from `server/`
since it's three one-line lookups and importing across those two
independent deployment surfaces for that alone isn't worth the coupling.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.tts_service import TTSService

from hams_tts.inference.engine import AudioChunk, Engine

DEFAULT_MODEL_DIR = "data/processed/checkpoint"


def _env_model_dir() -> Path:
    return Path(os.environ.get("HAMS_MODEL_DIR", DEFAULT_MODEL_DIR))


def _env_tokenizer_dir() -> Path | None:
    value = os.environ.get("HAMS_TOKENIZER_DIR")
    return Path(value) if value else None


def _env_speaker_id() -> int | None:
    value = os.environ.get("HAMS_SPEAKER_ID")
    return int(value) if value else None


def _pcm16_bytes(audio) -> bytes:
    """float32 mono [-1, 1] -> little-endian int16 PCM bytes.

    Same conversion as `hams_tts.server.main._pcm16_bytes` — duplicated for
    the same reason as the env-var helpers above.
    """
    import numpy as np

    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    return pcm16.tobytes()


class HamsTTSService(TTSService):
    """Pipecat TTS service for the fine-tuned Levantine Arabic/English
    code-switching VITS/MMS model.

    The checkpoint directory is environment-driven (HAMS_MODEL_DIR /
    HAMS_TOKENIZER_DIR / HAMS_DEVICE / HAMS_SPEAKER_ID — see
    `hams_tts.server.main`'s module docstring), not a constructor argument,
    so the same trained checkpoint is picked up identically by the server
    and this plugin without duplicating path configuration in application
    code that wires up a Pipecat pipeline.

    `default_lang` ("ar" or "en") is the language assumed for any text span
    the input doesn't otherwise disambiguate (see
    `hams_tts.text.pipeline.phonemize_text`'s `default_lang` parameter) —
    kept as a plain constructor argument rather than wired into Pipecat's
    runtime-updatable `TTSSettings` system, since mid-conversation language
    switching isn't a requirement here and that wiring would be unused
    complexity.
    """

    def __init__(
        self,
        *,
        default_lang: str = "ar",
        model_dir: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )

        self._default_lang = default_lang
        resolved_model_dir = Path(model_dir) if model_dir is not None else _env_model_dir()

        logger.debug(f"{self}: loading HamsTTS Engine from {resolved_model_dir}")
        self._engine = Engine(
            model_dir=resolved_model_dir,
            tokenizer_dir=tokenizer_dir or _env_tokenizer_dir(),
            device=device or os.environ.get("HAMS_DEVICE"),
            speaker_id=_env_speaker_id(),
        )
        logger.debug(
            f"{self}: Engine loaded (device={self._engine.device}, "
            f"sampling_rate={self._engine.sampling_rate})"
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize `text` and yield `TTSAudioRawFrame`s (via the base
        class's `_stream_audio_frames_from_iterator`) as each chunk from
        `Engine.synthesize_streaming` becomes ready."""

        def _next(gen) -> AudioChunk | None:
            try:
                return next(gen)
            except StopIteration:
                return None

        async def _pcm_iterator(gen) -> AsyncIterator[bytes]:
            while True:
                chunk = await asyncio.to_thread(_next, gen)
                if chunk is None:
                    return
                yield _pcm16_bytes(chunk.audio)

        try:
            await self.start_tts_usage_metrics(text)

            gen = self._engine.synthesize_streaming(text, default_lang=self._default_lang)
            async for frame in self._stream_audio_frames_from_iterator(
                _pcm_iterator(gen),
                in_sample_rate=self._engine.sampling_rate,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:  # noqa: BLE001 - must not crash the pipeline
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"HamsTTS synthesis failed: {e}")
        finally:
            await self.stop_ttfb_metrics()


# --- Example usage -----------------------------------------------------
#
#   import os
#   os.environ["HAMS_MODEL_DIR"] = "data/processed/checkpoint"
#
#   from pipecat.pipeline.pipeline import Pipeline
#   from pipecat.pipeline.runner import PipelineRunner
#   from pipecat.pipeline.task import PipelineParams, PipelineTask
#   from hams_tts.pipecat_plugin.hams_tts_service import HamsTTSService
#
#   tts = HamsTTSService(default_lang="ar")
#
#   pipeline = Pipeline([
#       transport.input(),
#       stt,
#       llm,
#       tts,
#       transport.output(),
#   ])
#   task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
#   await PipelineRunner().run(task)
