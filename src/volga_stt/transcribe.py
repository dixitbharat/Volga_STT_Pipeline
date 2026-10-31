from __future__ import annotations

import os
import shutil
from pathlib import Path

from volga_stt.audio import prepare_chunks
from volga_stt.engine import EngineResult, FasterWhisperEngine, MockEngine, SpeechToTextEngine
from volga_stt.schemas import Segment, TranscriptionResult, Word


def stitch_chunks(
    results: list[tuple[float, float, EngineResult]],
    overlap_seconds: float,
) -> list[tuple[float, float, str, list]]:
    """Map chunk-local timestamps onto the original timeline and drop overlap.

    For chunk i>0 we discard content in the first overlap/2 seconds so a word
    split across a boundary is kept from exactly one chunk.
    """
    stitched: list[tuple[float, float, str, list]] = []
    for index, (offset, _chunk_duration, engine_result) in enumerate(results):
        trim_before = 0.0 if index == 0 else overlap_seconds / 2.0
        for seg in engine_result.segments:
            if not seg.text.strip():
                continue
            if seg.end <= trim_before:
                continue
            words = []
            for w in seg.words:
                if w.end <= trim_before:
                    continue
                words.append(
                    Word(
                        start=round(offset + w.start, 3),
                        end=round(offset + w.end, 3),
                        word=w.word,
                        probability=w.probability,
                    )
                )
            stitched.append(
                (
                    round(offset + max(seg.start, trim_before), 3),
                    round(offset + seg.end, 3),
                    seg.text.strip(),
                    words or None,
                )
            )
    return stitched


class TranscriptionPipeline:
    def __init__(
        self,
        engine: SpeechToTextEngine | None = None,
        chunk_seconds: float = 300.0,
        overlap_seconds: float = 8.0,
        keep_work_dir: bool = False,
    ) -> None:
        self.engine = engine or default_engine()
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.keep_work_dir = keep_work_dir

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        audio_path = Path(audio_path)
        wav, chunks, duration = prepare_chunks(
            audio_path,
            chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds,
        )
        work_dir = wav.parent
        try:
            engine_results: list[tuple[float, float, EngineResult]] = []
            language_used = language
            language_prob = None
            for chunk in chunks:
                result = self.engine.transcribe(
                    chunk.path,
                    language=language,
                    word_timestamps=word_timestamps,
                )
                language_used = result.language or language_used
                language_prob = result.language_probability or language_prob
                engine_results.append(
                    (chunk.offset_seconds, chunk.duration_seconds, result)
                )

            stitched = stitch_chunks(engine_results, self.overlap_seconds)
            segments = [
                Segment(id=i, start=start, end=end, text=text, words=words)
                for i, (start, end, text, words) in enumerate(stitched)
            ]
            full_text = " ".join(s.text for s in segments).strip()
            return TranscriptionResult(
                text=full_text,
                language=language_used,
                language_probability=language_prob,
                duration_seconds=round(duration, 3),
                chunk_count=len(chunks),
                model=self.engine.model_name,
                segments=segments,
            )
        finally:
            if not self.keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)


def default_engine() -> SpeechToTextEngine:
    backend = os.environ.get("VOLGA_STT_BACKEND", "whisper").lower()
    if backend in {"mock", "fake"}:
        return MockEngine()
    model = os.environ.get("VOLGA_WHISPER_MODEL", "tiny")
    device = os.environ.get("VOLGA_WHISPER_DEVICE", "cpu")
    compute = os.environ.get("VOLGA_WHISPER_COMPUTE", "int8")
    return FasterWhisperEngine(model_name=model, device=device, compute_type=compute)
