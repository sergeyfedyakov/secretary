# secretary

Batch audio transcription to text — fully local, no cloud.

- Engine: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).
- Default model: `large-v3-turbo` (good quality, multilingual, excellent at Russian).
- The model is downloaded automatically on the first run, with a progress bar (speed, ETA).
- Language is auto-detected by default.
- Speaker diarization (`--diarize`): `pyannote/speaker-diarization-community-1` backend.

## Pre-built binaries (Windows)

Grab a ready-to-run exe from [GitHub Releases](https://github.com/sergeyfedyakov/secretary/releases) — no Python required:

| File | Size | Diarization |
|---|---|---|
| `secretary-light.exe` | ~100 MB | no |
| `secretary.exe` | ~300 MB | ✓ |

Models are downloaded automatically on first run. For diarization, drop an `.env` file with your token next to the exe (use `.env.sample` from the release as a template), or set the `HF_TOKEN` environment variable.

[Русская версия](README.ru.md)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest (for tests)
pip install -e .                # console command `secretary`
```

After installation you can use `run.cmd` (Windows) or `run.sh` (Linux/macOS) for one-command launch (no manual venv activation):

```cmd
:: Windows
run.cmd recording.mp3
run.cmd ./recordings --language ru --out-dir ./transcripts
```

```bash
# Linux / macOS
./run.sh recording.mp3
./run.sh ./recordings --language ru --out-dir ./transcripts
```

## Usage

```bash
# a single file
secretary recording.mp3

# a folder (recursively) — one .txt per audio file
secretary ./recordings

# explicit language + output into a separate folder
secretary ./recordings --language ru --out-dir ./transcripts

# smaller model for CPU, silence filter off
secretary a.mp3 --model small --compute-type int8 --no-vad

# diarization: speaker labels [HH:MM:SS] SPEAKER_nn: text
secretary recording.mp3 --diarize

# "SRT-style" format, long utterances split into ~10s chunks
secretary lecture.mp3 --diarize --format srt

# system prompt for recognition
secretary lecture.mp3 --prompt "Transcription of a programming lecture"
```

`python -m secretary ...` works the same way.

## Models with subfolders (pre-quantized, faster-whisper)

Some HF repositories ship several pre-built CTranslate2 variants in subfolders
(e.g. `coriollon/whisper-large-v3-turbo-russian` — a Russian fine-tune with
punctuation). The repo root has no `model.bin`, so the subfolder must be given
explicitly:

```bash
# "int8 weights + fp16 compute" variant (782 MB, current v2)
secretary recording.mp3 --model coriollon/whisper-large-v3-turbo-russian/ct2_int8_float16

# "int16" variant (1.6 GB, marked as the outdated v1 build in the model card)
secretary recording.mp3 --model coriollon/whisper-large-v3-turbo-russian/ct2-int16 --compute-type int16
```

Only the selected subfolder is downloaded. `--compute-type` is forced to `int8`
on CPU automatically (verified to work with `ct2_int8_float16`); on CUDA use
`--compute-type int8_float16` for that variant, as stated in the model card.

## Offline model installation

If the target machine has no internet, download models on a machine with access
and transfer the cache folder.

### Method 1: huggingface-cli (recommended)

```bash
pip install huggingface_hub
# Base Whisper model (~3 GB)
huggingface-cli download Systran/faster-whisper-large-v3-turbo --local-dir ./models/faster-whisper-large-v3-turbo

# Russian fine-tune with punctuation (782 MB)
huggingface-cli download coriollon/whisper-large-v3-turbo-russian --local-dir ./models/whisper-large-v3-turbo-russian --include "ct2_int8_float16/*"

# Diarization model (HF_TOKEN required, ~400 MB)
huggingface-cli download pyannote/speaker-diarization-community-1 --local-dir ./models/speaker-diarization --token hf_xxx
```

Transfer the `./models/` folder to the target machine and set the path:

```cmd
set SECRETARY_MODEL_CACHE=D:\models
secretary recording.mp3 --model D:\models\faster-whisper-large-v3-turbo
```

### Method 2: first run on an internet-connected machine

Run `secretary` once on a machine with internet — the model downloads to
`~/.cache/secretary/models/`. Copy that folder to the target machine
at the same location, or specify `--model-cache` / `SECRETARY_MODEL_CACHE`.

### Method 3: pre-built exe + models separately

Download `secretary.exe` from [GitHub Releases](../../releases). Models are not
bundled into the exe — they are downloaded on first run, or transferred manually
using one of the methods above. The `SECRETARY_MODEL_CACHE` variable works with
the exe version as well.

## Options

| Option | Default | Description |
|---|---|---|
| `--model` | `large-v3-turbo` | alias (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`), an HF repo, `repo/subfolder`, or a local path. Also via env `SECRETARY_MODEL` |
| `--language` | auto | language code (`ru`, `en`, ...) |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--compute-type` | `auto` | `auto` (int8 on CPU, float16 on CUDA), `int8`, `float16`, `float32`, ... |
| `--vad` / `--no-vad` | enabled | silence filter |
| `--diarize` | off | speaker diarization: `SPEAKER_nn` labels (pyannote community-1 backend, requires `HF_TOKEN`) |
| `--format` | `plain` | `plain` — continuous text; `srt` — `[HH:MM:SS] [SPEAKER_nn] text` lines, long utterances split into ~10s chunks |
| `--prompt` | — | system prompt for recognition (e.g. "Transcription of a programming lecture") |
| `--out-dir` | next to the file | output folder |
| `--model-cache` | `~/.cache/secretary/models` | model cache directory (or env `SECRETARY_MODEL_CACHE`) |
| `--ffmpeg-path` | — | path to ffmpeg (fallback) |
| `--verbose` | off | verbose log (language, number of segments/speakers) |
| `--no-progress` | off | hide transcription progress bar (useful in CI/logs) |

## ffmpeg

A separate binary is not required: faster-whisper decodes mp3/m4a/mp4 etc. via
PyAV. If you still need a CLI converter (exotic formats): `winget install ffmpeg`
or the [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) builds; pass the path via
`--ffmpeg-path` or the `FFMPEG_BINARY` environment variable.

## Hugging Face token (`HF_TOKEN`)

Whisper (faster-whisper) models are public — no token needed for recognition.
A token is required for **diarization** (pyannote models are gated).

Where to get one:

1. Sign in / register at [huggingface.co](https://huggingface.co).
2. [Settings → Access Tokens](https://huggingface.co/settings/tokens) → **New token**,
   role **Read** → create and copy (`hf_...`).
3. Accept the pyannote model terms once — open and click **"Agree and access repository"**:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   (all pipeline components live inside that repository)

How to pass it (the token is never committed — see `.gitignore`):

```bash
# .env file in the project root (example — .env.example)
HF_TOKEN=hf_xxxxxxxxxxxxx
```

Or an OS environment variable: `$env:HF_TOKEN = "hf_xxx"` (PowerShell), `set HF_TOKEN=hf_xxx` (cmd).
The `.env` file is picked up automatically on startup.

## Diarization

Backend: `pyannote/speaker-diarization-community-1` (pyannote.audio 4.x).
How it works: STT segments → diarization of the same recording → every word/segment
gets a speaker by timestamp overlap (`exclusive_speaker_diarization` — no
overlapping utterances, more accurate alignment with whisper).

- Models are downloaded on the first `--diarize` run and cached in
  `~/.cache/secretary/pyannote` (env `PYANNOTE_CACHE`) — kept between sessions.
- Speaker-count parameters (`--min-speakers`/`--max-speakers`) are deliberately not
  exposed in the CLI: for batch processing they are unknown in advance.
