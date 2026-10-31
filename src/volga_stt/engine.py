from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class RawWord:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    words: list[RawWord] = field(default_factory=list)


@dataclass
class EngineResult:
    language: str | None
    language_probability: float | None
    segments: list[RawSegment]


def load_wav_mono_float32(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise RuntimeError(f"expected 16-bit PCM, got sample width {wf.getsampwidth()}")
        nch = wf.getnchannels()
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if nch > 1:
        frames = frames.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return frames.astype(np.float32) / 32768.0


class SpeechToTextEngine(Protocol):
    model_name: str

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> EngineResult: ...


class FasterWhisperEngine:
    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self._model = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> EngineResult:
        # Pass PCM, not a path. We already normalized with ffmpeg; this also
        # avoids faster-whisper 1.2.x + PyAV 19 `metadata_errors` breakage.
        audio = load_wav_mono_float32(audio_path)
        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments: list[RawSegment] = []
        for seg in segments_iter:
            words = []
            if word_timestamps and seg.words:
                words = [
                    RawWord(
                        start=w.start,
                        end=w.end,
                        word=w.word,
                        probability=getattr(w, "probability", None),
                    )
                    for w in seg.words
                ]
            segments.append(
                RawSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )
        return EngineResult(
            language=getattr(info, "language", language),
            language_probability=getattr(info, "language_probability", None),
            segments=segments,
        )


class MockEngine:
    """Deterministic engine for tests / offline demos. Does not load a model."""

    model_name = "mock"

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> EngineResult:
        from volga_stt.audio import probe_duration_seconds

        duration = probe_duration_seconds(audio_path)
        text = f"mock transcript for {audio_path.name}"
        words = []
        if word_timestamps:
            words = [
                RawWord(start=0.0, end=max(duration, 0.1), word=w, probability=1.0)
                for w in text.split()
            ]
        return EngineResult(
            language=language or "en",
            language_probability=1.0,
            segments=[
                RawSegment(start=0.0, end=duration, text=text, words=words)
            ],
        )
