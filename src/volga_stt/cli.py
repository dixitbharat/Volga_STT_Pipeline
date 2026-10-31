from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from volga_stt.audio import AudioError
from volga_stt.transcribe import TranscriptionPipeline, default_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio to text with per-segment timestamps."
    )
    parser.add_argument("audio", type=Path, help="Path to WAV/MP3/M4A/FLAC/...")
    parser.add_argument("--language", default=None, help="Force language code, e.g. en")
    parser.add_argument("--words", action="store_true", help="Include word-level timestamps")
    parser.add_argument("--chunk-seconds", type=float, default=300.0)
    parser.add_argument("--overlap-seconds", type=float, default=8.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON to this path")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the mock engine (no model download)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mock:
        from volga_stt.engine import MockEngine

        engine = MockEngine()
    else:
        engine = default_engine()
    pipeline = TranscriptionPipeline(
        engine=engine,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    try:
        result = pipeline.transcribe(
            args.audio, language=args.language, word_timestamps=args.words
        )
    except AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = result.model_dump()
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2))
    if args.as_json or args.out:
        print(json.dumps(payload, indent=2))
    else:
        print(f"language: {result.language}  duration: {result.duration_seconds}s  chunks: {result.chunk_count}")
        for seg in result.segments:
            print(f"[{seg.start:8.2f}s -> {seg.end:8.2f}s] {seg.text}")
        print()
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
