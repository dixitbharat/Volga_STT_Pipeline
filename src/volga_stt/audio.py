from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".mp4"}
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset_seconds: float
    duration_seconds: float


def resolve_ffmpeg() -> str:
    env_path = shutil.which("ffmpeg")
    if env_path:
        return env_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - install-time failure
        raise AudioError(
            "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
        ) from exc


def resolve_ffprobe() -> str | None:
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    ffmpeg = resolve_ffmpeg()
    candidate = str(Path(ffmpeg).with_name("ffprobe"))
    return candidate if Path(candidate).exists() else None


def validate_audio_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise AudioError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(
            f"Command failed ({' '.join(cmd[:3])}...): {result.stderr.strip()[:800]}"
        )
    return result


def convert_to_wav16k_mono(src: Path, dst: Path) -> Path:
    """Normalize any supported container/codec to 16 kHz mono PCM WAV.

    Whisper (and most STT models) expect 16 kHz mono. Doing this once up front
    means the rest of the pipeline never cares about the original format.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg()
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioError(f"ffmpeg produced an empty file from {src}")
    return dst


def probe_duration_seconds(path: Path) -> float:
    ffprobe = resolve_ffprobe()
    if ffprobe:
        result = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(result.stdout or "{}")
        duration = float(data.get("format", {}).get("duration", 0) or 0)
        if duration > 0:
            return duration

    ffmpeg = resolve_ffmpeg()
    result = subprocess.run(
        [ffmpeg, "-i", str(path)], capture_output=True, text=True
    )
    # ffmpeg prints probe info to stderr even on "failure"
    for token in result.stderr.replace(",", " ").split():
        if token.count(":") == 2 and token[:1].isdigit():
            h, m, s = token.split(":")
            try:
                return int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                continue
    raise AudioError(f"Could not determine duration for {path}")


def split_wav(
    wav_path: Path,
    out_dir: Path,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk]:
    """Split a WAV into overlapping windows.

    Overlap exists so we do not cut words on chunk boundaries. The stitcher
    later drops the duplicated overlap region.
    """
    if chunk_seconds <= 0:
        raise AudioError("chunk_seconds must be > 0")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise AudioError("overlap_seconds must be in [0, chunk_seconds)")

    duration = probe_duration_seconds(wav_path)
    if duration <= chunk_seconds:
        return [
            AudioChunk(
                path=wav_path, offset_seconds=0.0, duration_seconds=duration
            )
        ]

    ffmpeg = resolve_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    step = chunk_seconds - overlap_seconds
    chunks: list[AudioChunk] = []
    offset = 0.0
    index = 0
    while offset < duration:
        this_duration = min(chunk_seconds, duration - offset)
        chunk_path = out_dir / f"chunk_{index:04d}.wav"
        _run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{offset:.3f}",
                "-t",
                f"{this_duration:.3f}",
                "-i",
                str(wav_path),
                "-c",
                "copy",
                str(chunk_path),
            ]
        )
        chunks.append(
            AudioChunk(
                path=chunk_path,
                offset_seconds=offset,
                duration_seconds=this_duration,
            )
        )
        index += 1
        if offset + this_duration >= duration:
            break
        offset += step
    return chunks


def prepare_chunks(
    src: Path,
    work_dir: Path | None = None,
    chunk_seconds: float = 300.0,
    overlap_seconds: float = 8.0,
) -> tuple[Path, list[AudioChunk], float]:
    src = validate_audio_path(src)
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="volga-stt-"))
    wav = tmp / "normalized.wav"
    convert_to_wav16k_mono(src, wav)
    duration = probe_duration_seconds(wav)
    chunks = split_wav(wav, tmp / "chunks", chunk_seconds, overlap_seconds)
    return wav, chunks, duration
