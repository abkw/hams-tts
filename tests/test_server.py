"""Tests for hams_tts.server.main using a fake Engine (no real checkpoint
needed — training runs independently of this test suite, see AGENTS.md).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hams_tts.inference.engine import AudioChunk
from hams_tts.server import main as server_main


class FakeEngine:
    device = "cpu"
    sampling_rate = 16000

    def synthesize(self, text: str, default_lang: str = "ar"):
        audio = np.zeros(1600, dtype=np.float32)
        return audio, self.sampling_rate

    def synthesize_streaming(self, text: str, default_lang: str = "ar"):
        for i in range(2):
            yield AudioChunk(
                audio=np.zeros(800, dtype=np.float32),
                sampling_rate=self.sampling_rate,
                chunk_index=i,
                is_final=(i == 1),
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
def client_with_fake_engine():
    # `lifespan` resets app.state.engine on TestClient.__enter__ (it attempts
    # a real load and fails since no checkpoint exists in the test env), so
    # the fake must be installed *after* entering, not before.
    with TestClient(server_main.app) as client:
        server_main.app.state.engine = FakeEngine()
        server_main.app.state.load_error = None
        yield client
    server_main.app.state.engine = None


@pytest.fixture
def client_with_no_engine(monkeypatch):
    # `get_engine` retries the load on every request when `app.state.engine`
    # is None — a real feature (recovers once training finishes without a
    # server restart), but it means this fixture can't just rely on the
    # *default* model dir being absent to keep engine=None: once a real
    # checkpoint exists on disk (as it now does after training completed),
    # that retry would succeed mid-test. Point at a path that's guaranteed
    # not to exist instead of relying on ambient repo state.
    monkeypatch.setenv("HAMS_MODEL_DIR", "/nonexistent/path/for/testing")
    with TestClient(server_main.app) as client:
        server_main.app.state.engine = None
        server_main.app.state.load_error = "test: no checkpoint"
        yield client


def test_cors_header_present_for_browser_clients(client_with_fake_engine):
    # scripts/websocket_test_client.html is opened as a file:// page and
    # calls POST /tts via fetch() — without CORS headers the browser
    # blocks reading the response even though the server handled it fine.
    resp = client_with_fake_engine.post(
        "/tts", json={"text": "hello", "lang_hint": "en"}, headers={"Origin": "null"}
    )
    assert resp.headers["access-control-allow-origin"] == "*"


def test_health_reports_loaded(client_with_fake_engine):
    resp = client_with_fake_engine.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["sampling_rate"] == 16000


def test_health_reports_not_loaded(client_with_no_engine):
    resp = client_with_no_engine.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False
    assert "no checkpoint" in body["load_error"]


def test_tts_http_returns_wav(client_with_fake_engine):
    resp = client_with_fake_engine.post("/tts", json={"text": "hello", "lang_hint": "en"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"


def test_tts_http_rejects_empty_text(client_with_fake_engine):
    resp = client_with_fake_engine.post("/tts", json={"text": "   "})
    assert resp.status_code == 400


def test_tts_http_rejects_overlong_text(client_with_fake_engine):
    resp = client_with_fake_engine.post("/tts", json={"text": "a" * 3000})
    assert resp.status_code == 400


def test_tts_http_503_when_model_not_loaded(client_with_no_engine):
    resp = client_with_no_engine.post("/tts", json={"text": "hello"})
    assert resp.status_code == 503


def test_stream_happy_path(client_with_fake_engine):
    with client_with_fake_engine.websocket_connect("/tts/stream") as ws:
        ws.send_json({"text": "hello", "lang_hint": "en"})

        start = ws.receive_json()
        assert start["type"] == "stream_start"
        assert start["sampling_rate"] == 16000

        chunk0_header = ws.receive_json()
        assert chunk0_header["type"] == "chunk"
        assert chunk0_header["chunk_index"] == 0
        assert chunk0_header["is_final"] is False
        chunk0_bytes = ws.receive_bytes()
        assert len(chunk0_bytes) == 800 * 2  # int16 PCM

        chunk1_header = ws.receive_json()
        assert chunk1_header["is_final"] is True
        ws.receive_bytes()

        end = ws.receive_json()
        assert end["type"] == "stream_end"


def test_stream_rejects_empty_text(client_with_fake_engine):
    with client_with_fake_engine.websocket_connect("/tts/stream") as ws:
        ws.send_json({"text": ""})
        resp = ws.receive_json()
        assert resp["type"] == "error"


def test_stream_rejects_non_string_text(client_with_fake_engine):
    with client_with_fake_engine.websocket_connect("/tts/stream") as ws:
        ws.send_json({"text": 12345})
        resp = ws.receive_json()
        assert resp["type"] == "error"


def test_stream_rejects_bad_lang_hint(client_with_fake_engine):
    with client_with_fake_engine.websocket_connect("/tts/stream") as ws:
        ws.send_json({"text": "hello", "lang_hint": "fr"})
        resp = ws.receive_json()
        assert resp["type"] == "error"


def test_stream_can_recover_after_validation_error(client_with_fake_engine):
    """Connection stays open after a validation error so the client can retry."""
    with client_with_fake_engine.websocket_connect("/tts/stream") as ws:
        ws.send_json({"text": ""})
        err = ws.receive_json()
        assert err["type"] == "error"

        ws.send_json({"text": "hello", "lang_hint": "en"})
        start = ws.receive_json()
        assert start["type"] == "stream_start"


def test_stream_mid_synthesis_error_sends_error_and_closes():
    try:
        with TestClient(server_main.app) as client:
            server_main.app.state.engine = FailingEngine()
            server_main.app.state.load_error = None
            with client.websocket_connect("/tts/stream") as ws:
                ws.send_json({"text": "hello", "lang_hint": "en"})
                ws.receive_json()  # stream_start
                ws.receive_json()  # chunk 0 header
                ws.receive_bytes()  # chunk 0 bytes

                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "blew up" in msg["error"]
    finally:
        server_main.app.state.engine = None


def test_stream_404_when_model_not_loaded(client_with_no_engine):
    with client_with_no_engine.websocket_connect("/tts/stream") as ws:
        ws.send_json({"text": "hello"})
        resp = ws.receive_json()
        assert resp["type"] == "error"
        assert "not loaded" in resp["error"]
