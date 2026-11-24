"""Minimal runnable Pipecat pipeline demo for `HamsTTSService` — the "cut
showing the Pipecat pipeline working" shot in `docs/demo_video_script.md`.

Feeds a few hardcoded text frames (one AR, one EN, one code-switched)
through the plugin and plays them out the system's default audio device,
so recording this running is literally "run the script, it talks."

Requires the `local` extra for PyAudio (not in the base project deps,
since it needs the `portaudio` system library — install on whichever
machine you're recording the demo on, not needed for the rest of this
project's test suite):

    uv add "pipecat-ai[local]"
    # macOS also needs: brew install portaudio
    # Debian/Ubuntu: sudo apt-get install portaudio19-dev (needs root —
    # see AGENTS.md's espeak-ng vendoring note if that's not available;
    # the same apt-get-download + dpkg-deb -x trick works for this too)

Usage:
    HAMS_MODEL_DIR=data/processed/checkpoint uv run python3 scripts/demo_pipecat_pipeline.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEMO_UTTERANCES = [
    ("ar", "مرحبا، كيفك اليوم؟ شو أخبارك؟"),
    ("en", "Good morning! The meeting is scheduled for 3 o'clock."),
    ("ar", "بدي أعمل schedule لـ meeting بكرا الساعة 3"),  # code-switched
]


async def main() -> None:
    from pipecat.frames.frames import EndFrame, TextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    from hams_tts.pipecat_plugin.hams_tts_service import HamsTTSService

    transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))
    tts = HamsTTSService(default_lang="ar")

    pipeline = Pipeline([tts, transport.output()])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False))

    async def feed_utterances():
        for lang, text in DEMO_UTTERANCES:
            print(f"[demo] speaking ({lang}): {text}")
            await task.queue_frame(TextFrame(text))
            await asyncio.sleep(0.1)
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), feed_utterances())


if __name__ == "__main__":
    asyncio.run(main())
