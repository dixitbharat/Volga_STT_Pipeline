from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import write_sine_wav
from volga_stt.engine import EngineResult, MockEngine, RawSegment
from volga_stt.transcribe import TranscriptionPipeline, stitch_chunks


def test_stitch_drops_overlap_on_later_chunks() -> None:
    first = EngineResult(
        language="en",
        language_probability=0.9,
        segments=[RawSegment(start=0.0, end=5.0, text="hello there", words=[])],
    )
    second = EngineResult(
        language="en",
        language_probability=0.9,
        segments=[
            RawSegment(start=0.0, end=1.0, text="there", words=[]),
            RawSegment(start=2.0, end=4.0, text="world", words=[]),
        ],
    )
    stitched = stitch_chunks(
        [(0.0, 5.0, first), (4.0, 5.0, second)],
        overlap_seconds=2.0,
    )
    texts = [t[2] for t in stitched]
    assert "hello there" in texts
    assert "world" in texts
    assert "there" not in texts  # fully inside overlap/2 of chunk 1


def test_pipeline_mock_short(tmp_path: Path) -> None:
    wav = write_sine_wav(tmp_path / "a.wav", duration=1.0)
    result = TranscriptionPipeline(engine=MockEngine()).transcribe(wav, word_timestamps=True)
    assert result.model == "mock"
    assert result.chunk_count == 1
    assert result.segments
    assert result.segments[0].words
    assert "mock transcript" in result.text


def test_pipeline_chunks_long_audio(tmp_path: Path) -> None:
    wav = write_sine_wav(tmp_path / "long.wav", duration=11.0)
    result = TranscriptionPipeline(
        engine=MockEngine(),
        chunk_seconds=5.0,
        overlap_seconds=1.0,
    ).transcribe(wav)
    assert result.chunk_count >= 3
    assert result.duration_seconds == pytest.approx(11.0, abs=0.2)
    assert result.segments[-1].end > 5.0
