# Volga transcription pipeline

A small but real speech-to-text service: **file in → normalized audio → chunked Whisper → timestamped segments out**.

This repo is the assignment submission. Part 1 is implemented. Part 2 is the production design (not coded), written below so you can paste it into the form.

```
audio file  →  ffmpeg normalize (16 kHz mono WAV)
            →  split if long (overlap)
            →  faster-whisper (or mock engine)
            →  stitch timestamps
            →  JSON / CLI / HTTP
```

## Quick start

Python 3.10+ (3.12 recommended). ffmpeg is pulled in via `imageio-ffmpeg` if it is not on `PATH`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install gTTS   # optional, for a spoken MP3 sample

python scripts/generate_sample.py
pytest -q

# mock path (no model download)
volga-stt samples/tone.wav --mock

# real STT (downloads the tiny Whisper model on first run, ~75MB)
volga-stt samples/speech.mp3 --json --out output/speech.json

# HTTP (real Whisper; omit VOLGA_STT_BACKEND=mock)
uvicorn volga_stt.api:app --port 8000
curl -F file=@samples/speech.mp3 http://127.0.0.1:8000/v1/transcribe
```

Env knobs:

| Variable | Default | Meaning |
|---|---|---|
| `VOLGA_STT_BACKEND` | `whisper` | `whisper` or `mock` |
| `VOLGA_WHISPER_MODEL` | `tiny` | `tiny` / `base` / `small` / … |
| `VOLGA_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `VOLGA_WHISPER_COMPUTE` | `int8` | `int8` on CPU, `float16` on GPU |
| `VOLGA_MAX_UPLOAD_BYTES` | 50MB | API upload cap |

---

## Design decisions

The assignment is an engineering problem, not a research one. I did not train a model. I built a pipeline around an existing open-source STT engine and made the parts around it (formats, length, timestamps, API shape) explicit and testable.

### Why this stack

| Choice | Why | What I rejected |
|---|---|---|
| **Python** | Fastest path to STT, ffmpeg wrappers, FastAPI, and tests in one language. Reviewers can run it in minutes. | A from-scratch C++/Rust decoder. Wrong scope. |
| **faster-whisper** (CTranslate2 Whisper) | Local, open source, CPU `int8` works on a laptop, native **segment + word timestamps**, Silero **VAD**. No API key, no data leaving the machine. | Cloud STT (hides the pipeline). Training Whisper (out of scope). `openai-whisper` (heavier PyTorch). Vosk (weaker English, weaker timestamps). |
| **tiny model by default** | First run is ~75MB and seconds, not minutes. Swap with `VOLGA_WHISPER_MODEL=base|small` when quality matters. | Shipping `large-v3` as the demo default. |
| **ffmpeg + `imageio-ffmpeg`** | One decode path for every container/codec. Bundled binary so macOS without Homebrew ffmpeg still runs. | A Python decoder per format (`pydub`/`librosa` only). Fragile and incomplete. |
| **CLI and HTTP over the same pipeline** | Same `TranscriptionPipeline`, two fronts. CLI for local/debug; FastAPI for “expose it as a service.” | Two separate implementations. |

### Architecture (what runs on one file)

1. **Validate** extension and path. Fail fast on `.txt` etc. (HTTP 415).
2. **Normalize** to 16 kHz mono PCM WAV. Whisper’s feature extractor expects this; doing it once means the model never sees MP3/M4A/stereo/44.1 kHz.
3. **Chunk if long** (default 5 min, 8 s overlap). Whisper’s internal ~30 s window is not enough: a 3-hour file still blows RAM and HTTP timeouts if you treat it as one blob.
4. **Transcribe each chunk** with VAD on. Pass **float32 PCM**, not a file path, so we don’t depend on PyAV’s decoder (faster-whisper 1.2.x breaks on PyAV 19).
5. **Stitch**: add chunk offset to every timestamp; drop the first `overlap/2` of chunk `i>0` so a word on a cut is not duplicated.
6. **Emit JSON**: full `text` plus `segments[]` with `start` / `end` / `text` (optional `words[]`).

The STT backend is a **protocol** (`SpeechToTextEngine`). Production uses `FasterWhisperEngine`. Tests and `--mock` use `MockEngine` with the **same result schema**, so chunking/API tests do not download a model.

### Formats

Normalize once, then forget the original codec. Unsupported extensions never reach the model. If this grew: probe streams, reject video-only files, cap bitrate, strip metadata before storage.

### Long audio

Overlap exists because a hard cut in the middle of a word destroys that word. A fixed `overlap/2` trim is simple and good enough for a demo. Production would align the overlap with word timestamps (keep the higher-confidence copy). Chunks could run in parallel on a worker pool; stitch still has to be ordered.

Synchronous HTTP is **wrong** for hour-long media. The demo API is sync on purpose (easy to demo). Part 2 is the real shape: 202 + queue.

### API shape in this repo vs production

Implemented: blocking `POST /v1/transcribe` + `GET /health`. Fine for a reviewer with a 10-second clip. The Whisper model is loaded once in the process (singleton) — correct for a single demo worker, wrong if you scale the API process itself.

Not implemented (by assignment: “no need to implement”): object storage, job queue, retries, auth. Those decisions are written in Part 2 so the form has an answer without fake distributed code.

### Testing

Tests generate sine-wave WAVs and use `MockEngine`. They cover: reject bad types, convert/probe, split with overlap, stitch dropping the overlap, HTTP 200/415. They do **not** assert Whisper quality (that is non-deterministic and needs a download). Real-model checks are manual (`volga-stt samples/speech.mp3`).

### Tradeoffs I accepted

- **tiny** over **small**: demo speed > WER. Reviewers can bump the env var.
- **CPU int8** over GPU: this machine may have no CUDA. Same code path works on `cuda` + `float16` if you set env.
- **Sync API** over jobs: assignment Part 1 is a working script/service; Part 2 is design.
- **Fixed overlap trim** over forced alignment: less code, rare duplicate/drop at boundaries.
- **Temp files then delete** over pure streaming: simpler ffmpeg CLI; `finally` unlinks. Fine until I/O is the bottleneck.

### What I would change first in production

Async jobs (S3 + queue + workers), auth, size/duration caps, per-chunk checkpointing, and a larger model on GPU workers. The JSON schema would stay the same so this CLI/API remain a compatible client.

---

## Part 1 — Transcription pipeline

### Accepts an audio file (WAV, MP3, …)

CLI: `volga-stt path/to/file.mp3`

API: `POST /v1/transcribe` as `multipart/form-data` field `file`.

`audio.validate_audio_path` checks existence and extension. The actual decode is ffmpeg, so container/codec pairs like M4A/AAC work as long as ffmpeg can read them.

### Transcribes spoken language into text

`FasterWhisperEngine` wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 port of OpenAI Whisper). It is local, open source, CPU-friendly, and good enough to judge engineering without training anything.

`MockEngine` exists so tests and reviewers can run the rest of the pipeline without a model download. Same `TranscriptionResult` shape.

### Returns transcription with timestamps per segment

Response schema (`src/volga_stt/schemas.py`):

```json
{
  "text": "hello world",
  "language": "en",
  "language_probability": 0.98,
  "duration_seconds": 12.4,
  "chunk_count": 1,
  "model": "tiny",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.8,
      "text": "hello world",
      "words": [{"start": 0.0, "end": 0.4, "word": "hello", "probability": 0.91}]
    }
  ]
}
```

`--words` / `word_timestamps=true` fills `segments[].words`.

### How do you handle different audio formats?

**Normalize once, then forget the original codec.**

Whisper's feature extractor wants 16 kHz mono float audio. Every input is run through ffmpeg:

```
ffmpeg -i input -vn -ac 1 -ar 16000 -c:a pcm_s16le normalized.wav
```

Why this instead of a Python decoder per format:

- One code path for WAV, MP3, M4A, FLAC, OGG, WebM, AAC, MP4-with-audio.
- ffmpeg already has the messy codec work.
- `imageio-ffmpeg` vendors a binary so the project runs on a Mac without a system ffmpeg.
- Unsupported extensions fail fast (415 / CLI exit 2) instead of dying inside the model.

If I needed more: probe streams, reject video-only files, transcode with a bitrate cap, and strip metadata before storage.

### How do you deal with long audio files?

Three layers, because “long” fails in different ways:

1. **Model window.** Whisper already works in ~30s windows internally. That is not enough by itself — a 3-hour file still has to live in RAM and hold a request open.
2. **Explicit chunking** in `audio.split_wav`: default **5 minutes** with an **8 second overlap**. Each chunk is a separate transcribe call. Timestamps are shifted by the chunk offset.
3. **Overlap stitch** in `transcribe.stitch_chunks`: chunk `i>0` drops content in the first `overlap/2` seconds so a word on a cut is not emitted twice.
4. **VAD** (`vad_filter=True`) so silence is not sent through the decoder.

Tradeoffs I would take next if this went to production:

- Parallelize chunks on a worker pool, but keep order when stitching.
- Align overlap with a cheap text/word-timestamp diff instead of a fixed trim (better at rapid speech).
- Hard cap duration / file size at the API; for hours-long media, switch to the async job design in Part 2.
- Stream ffmpeg (`-f segment`) instead of writing every chunk to disk if I/O becomes the bottleneck.

---

## Part 2 — System design (not implemented)

### How would you handle concurrent uploads?

Do **not** transcribe inside the upload request at scale. Whisper on CPU is minutes per hour of audio. Holding HTTP workers across that will melt under a few concurrent users.

Shape:

```
client → API (auth, validate, size/duration cap)
      → write object to S3/GCS (key = job_id)
      → enqueue job {job_id, object_key, options}  (SQS / Redis / Rabbit)
      → 202 { job_id, status: queued }

N workers pull jobs, download (or stream), run TranscriptionPipeline,
write transcript, mark complete. Horizontal scale = add workers.
GPU workers if the queue latency SLA needs it.
```

Concurrency controls:

- Per-tenant rate limits and a max in-flight jobs quota.
- Idempotency key on upload so double-clicks don't spawn two jobs.
- Backpressure: when the queue depth is high, reject with 429 instead of eating disk.
- The API process should be stateless; the current singleton `WhisperModel` in `api.py` is fine for a demo and **wrong** for multi-worker production (load the model in the worker process, once).

### How would you store audio and transcripts?

Split **blobs** from **records**.

| What | Where | Why |
|---|---|---|
| Original + normalized audio | Object storage (S3), SSE-KMS, lifecycle to IA/Glacier | Cheap, durable, not in Postgres |
| Job row | Postgres | `id, tenant_id, status, audio_uri, duration, model, error, created_at, attempts` |
| Transcript | Postgres JSONB **or** object `s3://…/job_id.json` plus a `text` column for search | JSONB is fine to low millions of jobs; dump to object store if segments get huge |
| Search | Postgres FTS to start, OpenSearch later | “find clip where they said X” |
| Retention | TTL + legal hold flag | Audio is PII; default delete at 30 days |

Never store raw bytes on the API box except as a temp file that is deleted in `finally` (the demo already does that).

### How do you retry or recover failed transcriptions?

Jobs are a state machine: `queued → running → succeeded | failed | dead`.

- Worker **leases** a job (visibility timeout > worst-case transcribe time). Crash before ack → job becomes visible again. This is the recovery story.
- Classify errors: `InvalidAudio` (4xx, do not retry), `Timeout` / `WorkerOOM` / `ModelLoad` (retry), `Dependency` (retry with jitter).
- Exponential backoff, capped attempts (e.g. 3), then **dead-letter queue** + `failed` status with the last error.
- Idempotent writes: transcript PUT uses `job_id` as the key; a retry overwrites the same row.
- Partial progress: persist per-chunk results so a 2-hour file that dies at chunk 18 resumes at 18, not 0.
- Poison-pill isolation: if one file OOMs, that worker dies; the lease returns the job; a smaller-chunk retry path can take it.

### How would you expose this as an API?

Demo (implemented): synchronous `POST /v1/transcribe`.

Production:

```
POST   /v1/jobs                  202 { id, status }
GET    /v1/jobs/{id}             200 { status, result? error? }
GET    /v1/jobs/{id}/transcript  200  (full JSON, 409 if not ready)
DELETE /v1/jobs/{id}             204  (and object lifecycle)
POST   /v1/jobs/{id}/retry       202  (from failed)
GET    /healthz  /readyz
```

Auth: bearer JWT or API keys with tenant id. Webhook `transcript.completed` so clients don't poll. OpenAPI from FastAPI. Same JSON schema as today so the demo client keeps working — the only change is 202 + poll instead of blocking.

---

## Project layout

```
src/volga_stt/
  audio.py        formats, ffmpeg, chunking
  engine.py       faster-whisper + mock
  transcribe.py   pipeline + overlap stitch
  schemas.py      pydantic result
  cli.py
  api.py
tests/            format, chunk, stitch, HTTP
scripts/generate_sample.py
VIDEO_SCRIPT.md   what to say in the recording
```

## Tests

```bash
pytest -q
```

They use `MockEngine` and generated WAVs. They do **not** download Whisper. They do need ffmpeg (via `imageio-ffmpeg`).

## Video

See `VIDEO_SCRIPT.md`. Record ~3–4 minutes in English with this repo on screen: architecture, format/chunk decisions, a live CLI/API run, then Part 2 in one minute.

## License

Assignment code. Whisper weights follow their own licenses (MIT for the code, the model cards for weights).
