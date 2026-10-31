from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest


def write_sine_wav(path: Path, duration: float = 1.0, sample_rate: int = 16000, freq: float = 440.0) -> Path:
    n = int(duration * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            sample = int(10000 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return path


@pytest.fixture
def short_wav(tmp_path: Path) -> Path:
    return write_sine_wav(tmp_path / "short.wav", duration=1.2)
