# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project

Audio enhancement pipeline for 429 YouTube recordings (341 hours) from The Reading Room BKK, an independent art/intellectual space in Bangkok (2011–2019). 21 ML and DSP pipelines for cleaning up archival audio while preserving ambient character.

## Commands

```bash
# Pipeline comparison
python -m readingroom_audio.compare
python -m readingroom_audio.compare --pipelines original deepfilter_12dB hybrid_demucs_df

# Batch processing (all 429 videos)
python -m readingroom_audio.batch run --pipeline hybrid_demucs_df
python -m readingroom_audio.batch run --pipeline hybrid_demucs_df --resume
python -m readingroom_audio.batch status

# Systematic benchmark
python -m readingroom_audio.benchmark run-all
python -m readingroom_audio.benchmark run-all --target-n 5 --pipelines original ffmpeg_gentle hybrid_demucs_df

# Individual benchmark phases
python -m readingroom_audio.benchmark select    [--target-n 40] [--seed 42]
python -m readingroom_audio.benchmark download
python -m readingroom_audio.benchmark extract   [--duration 45]
python -m readingroom_audio.benchmark baseline
python -m readingroom_audio.benchmark enhance   [--pipelines ...]
python -m readingroom_audio.benchmark analyze

# Listening test
python -m readingroom_audio.listening_test run-all
python -m readingroom_audio.listening_test run-all --target-n 2 --pipelines original ffmpeg_gentle

# Video mux
python -m readingroom_audio.mux verify --pipeline ffmpeg_gentle
python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --resume
python -m readingroom_audio.mux status

# Download
python -m readingroom_audio.download
python -m readingroom_audio.download --video-id VIDEO_ID

# Unified CLI
python -m readingroom_audio --help
python -m readingroom_audio benchmark run-all
```

## Architecture

`src/readingroom_audio/` — modular pipeline package:
- `enhance.py` — 21 enhancement pipelines (DeepFilterNet, Demucs, ClearVoice, MP-SENet, Resemble, SpeechBrain, ffmpeg, hybrids)
- `score.py` — DNSMOS/NISQA/UTMOS quality scoring (NISQA chunked to 9s windows)
- `compare.py` — orchestrated comparison with CLI
- `download.py` — yt-dlp batch download with resume
- `batch.py` — batch processor for all 429 videos (download → enhance → FLAC)
- `mux.py` — video mux pipeline (download video → verify duration → mux enhanced audio)
- `sampling.py` — stratified sample selection from 161 events
- `segment.py` — Silero VAD segment extraction for benchmark
- `benchmark.py` — systematic benchmark runner with statistical analysis
- `listening_test.py` — listening test generator (HTML + MP3s for GitHub Pages)
- `utils.py` — ffmpeg wrappers, FLAC encoding, shared helpers
- `__main__.py` — unified CLI dispatcher

`notebooks/audio_comparison.ipynb` — interactive analysis with Altair visualizations.

## Key data flow

- **Input**: `data/events/*.json` (161 event files, committed, ~1MB total)
- **Downloaded**: `data/audio/raw/*.m4a` (128kbps AAC from YouTube, ~19GB)
- **Enhanced**: `data/audio/enhanced_final/{pipeline}/*.flac` (lossless, ~70GB per pipeline)
- **Video mux**: `data/video/muxed/{pipeline}/*.mp4`

## Dependencies

Python packages: `torch`, `torchaudio`, `soundfile` (core); `demucs`, `clearvoice`, `deepfilternet`, `MPSENet`, `resemble-enhance`, `speechbrain` (enhance); `torchmetrics[audio]` (score); `altair`, `pandas`, `scipy` (viz)
System: `ffmpeg` (via Homebrew on macOS)

Python 3.12 required.

DeepFilterNet requires patch in `.venv/lib/python3.12/site-packages/df/io.py` (replace `torchaudio.backend.common.AudioMetaData` with soundfile fallback).

## Preferred pipeline

**`hybrid_demucs_df`** — Demucs vocals → DeepFilterNet 12dB → ffmpeg loudnorm. Best subjective quality, preserves ambient atmosphere (laughter, room sound).
