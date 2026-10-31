from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from volga_stt.audio import SUPPORTED_EXTENSIONS, AudioError
from volga_stt.transcribe import TranscriptionPipeline, default_engine

app = FastAPI(
    title="Volga STT",
    version="0.1.0",
    description="Synchronous transcription API. See README for the async production design.",
)

_pipeline: TranscriptionPipeline | None = None


def get_pipeline() -> TranscriptionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TranscriptionPipeline(engine=default_engine())
    return _pipeline


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = None,
    word_timestamps: bool = False,
):
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    max_bytes = int(os.environ.get("VOLGA_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Upload too large")
            tmp.write(chunk)

    try:
        result = get_pipeline().transcribe(
            tmp_path, language=language, word_timestamps=word_timestamps
        )
        return JSONResponse(result.model_dump())
    except AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
