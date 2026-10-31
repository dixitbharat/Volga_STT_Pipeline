from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import write_sine_wav
from volga_stt.audio import (
    AudioError,
    convert_to_wav16k_mono,
    prepare_chunks,
    probe_duration_seconds,
    split_wav,
    validate_audio_path,
)


def test_rejects_missing_and_unsupported(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match="not found"):
        validate_audio_path(tmp_path / "nope.wav")
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    with pytest.raises(AudioError, match="Unsupported"):
        validate_audio_path(bad)


def test_convert_and_probe_wav(tmp_path: Path) -> None:
    src = write_sine_wav(tmp_path / "src.wav", duration=1.5, sample_rate=44100)
    dst = tmp_path / "out.wav"
    convert_to_wav16k_mono(src, dst)
    duration = probe_duration_seconds(dst)
    assert 1.4 <= duration <= 1.7


def test_split_long_wav_with_overlap(tmp_path: Path) -> None:
    src = write_sine_wav(tmp_path / "long.wav", duration=12.0)
    chunks = split_wav(src, tmp_path / "chunks", chunk_seconds=5.0, overlap_seconds=1.0)
    assert len(chunks) >= 3
    assert chunks[0].offset_seconds == 0.0
    assert chunks[1].offset_seconds == pytest.approx(4.0)
    assert chunks[-1].offset_seconds + chunks[-1].duration_seconds == pytest.approx(12.0, abs=0.15)


def test_prepare_chunks_short_file_stays_single(tmp_path: Path) -> None:
    src = write_sine_wav(tmp_path / "short.wav", duration=2.0)
    _, chunks, duration = prepare_chunks(
        src, work_dir=tmp_path / "work", chunk_seconds=30.0, overlap_seconds=2.0
    )
    assert duration == pytest.approx(2.0, abs=0.1)
    assert len(chunks) == 1
