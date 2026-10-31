from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import write_sine_wav
from volga_stt.api import app
from volga_stt.engine import MockEngine
from volga_stt.transcribe import TranscriptionPipeline


def test_health() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_transcribe_upload(tmp_path) -> None:
    app.dependency_overrides.clear()
    wav = write_sine_wav(tmp_path / "clip.wav", duration=0.8)

    def _pipeline():
        return TranscriptionPipeline(engine=MockEngine())

    # get_pipeline is not a FastAPI dependency; patch the module singleton path
    import volga_stt.api as api_mod

    api_mod._pipeline = _pipeline()
    client = TestClient(app)
    with wav.open("rb") as fh:
        response = client.post("/v1/transcribe", files={"file": ("clip.wav", fh, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert body["segments"]


def test_rejects_bad_extension() -> None:
    import volga_stt.api as api_mod

    api_mod._pipeline = TranscriptionPipeline(engine=MockEngine())
    client = TestClient(app)
    response = client.post(
        "/v1/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415
