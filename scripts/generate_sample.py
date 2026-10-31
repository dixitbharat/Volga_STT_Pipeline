#!/usr/bin/env python3
"""Create a short spoken sample (gTTS) plus a fallback tone WAV."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
TEXT = "Hello. This is a sample recording for the Volga transcription assignment."


def write_tone(path: Path, duration: float = 3.0, sample_rate: int = 16000) -> Path:
    n = int(duration * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            sample = int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return path


def write_speech() -> Path | None:
    try:
        from gtts import gTTS
    except ImportError:
        print("gTTS not installed; skipping speech.mp3 (tone.wav is enough for mock runs)")
        return None
    SAMPLES.mkdir(parents=True, exist_ok=True)
    path = SAMPLES / "speech.mp3"
    gTTS(TEXT, lang="en").save(str(path))
    return path


if __name__ == "__main__":
    tone = write_tone(SAMPLES / "tone.wav")
    speech = write_speech()
    print(f"wrote {tone}")
    if speech:
        print(f"wrote {speech}")
