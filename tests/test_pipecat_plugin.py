"""Tests for hams_tts.pipecat_plugin.hams_tts_service using a fake Engine
(no real checkpoint needed — training runs independently, see AGENTS.md).

`Engine` is monkeypatched at the plugin module's import site so
`HamsTTSService.__init__` never tries to load real model weights.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hams_tts.inference.engine import AudioChunk
from hams_tts.pipecat_plugin import hams_tts_service as plugin_mod


class FakeEngine:
    device = "cpu"
    sampling_rate = 16000

    def __init__(self, *args, **kwargs):
        pass

    def synthesize_streaming(self, text: str, default_lang: str = "ar"):
        for i in range(3):
            yield AudioChunk(
                audio=np.zeros(800, dtype=np.float32),
                sampling_rate=self.sampling_rate,
                chunk_index=i,
                is_final=(i == 2),
                gen_seconds=0.01,
            )


class FailingEngine(FakeEngine):
    def synthesize_streaming(self, text: str, default_lang: str = "ar"):
        yield AudioChunk(
            audio=np.zeros(800, dtype=np.float32),
            sampling_rate=self.sampling_rate,
            chunk_index=0,
            is_final=False,
            gen_seconds=0.01,
        )
        raise RuntimeError("synthesis blew up")


@pytest.fixture
def fake_service(monkeypatch):
    monkeypatch.setattr(plugin_mod, "Engine", FakeEngine)
    service = plugin_mod.HamsTTSService(default_lang="en")
    return service


async def _drain(gen):
    return [frame async for frame in gen]


def test_run_tts_yields_audio_frames(fake_service):
    from pipecat.frames.frames import TTSAudioRawFrame

    frames = asyncio.run(_drain(fake_service.run_tts("hello world", context_id="ctx-1")))

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) >= 1
    for f in audio_frames:
        assert f.sample_rate == 16000
        assert f.num_channels == 1
        assert len(f.audio) == 800 * 2  # int16 PCM bytes per chunk
        assert f.context_id == "ctx-1"


def test_run_tts_error_yields_error_frame(monkeypatch):
    from pipecat.frames.frames import ErrorFrame

    monkeypatch.setattr(plugin_mod, "Engine", FailingEngine)
    service = plugin_mod.HamsTTSService(default_lang="en")

    frames = asyncio.run(_drain(service.run_tts("hello", context_id="ctx-2")))

    assert any(isinstance(f, ErrorFrame) for f in frames)
    assert any("blew up" in f.error for f in frames if isinstance(f, ErrorFrame))


def test_pcm16_conversion_round_trip():
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    pcm = plugin_mod._pcm16_bytes(audio)
    decoded = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
    assert np.allclose(decoded, audio, atol=1e-3)
