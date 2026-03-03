# Reading Room BKK — Audio Enhancement Pipeline

Audio quality enhancement for 429 YouTube recordings (341 hours) from [The Reading Room BKK](https://www.facebook.com/TheReadingRoomBKK/), an independent art and intellectual space in Bangkok (2011–2019).

21 enhancement pipelines are evaluated through systematic benchmarks to find the best approach for improving archival audio quality while preserving ambient character (laughter, room atmosphere).

## Pipeline catalog

| # | Pipeline | Method |
|---|----------|--------|
| 1 | `deepfilter_full` | DeepFilterNet3, full suppression |
| 2 | `deepfilter_12dB` | DeepFilterNet3, 12dB attenuation limit |
| 3 | `deepfilter_6dB` | DeepFilterNet3, 6dB (minimal) |
| 4 | `deepfilter_18dB` | DeepFilterNet3, 18dB (strong) |
| 5 | `mossformer2_48k` | ClearVoice MossFormer2 48kHz |
| 6 | `mossformergan_16k` | ClearVoice MossFormerGAN 16kHz |
| 7 | `frcrn_16k` | ClearVoice FRCRN 16kHz |
| 8 | `demucs_vocals` | Demucs htdemucs vocal separation |
| 9 | `demucs_ft_vocals` | Demucs htdemucs_ft (fine-tuned) |
| 10 | `ffmpeg_gentle` | Traditional DSP chain (highpass + afftdn + acompressor + loudnorm) |
| 11 | `hybrid_demucs_df` | Demucs vocals → DeepFilter 12dB → loudnorm |
| 12 | `hybrid_demucs_ft_df` | Demucs_ft vocals → DeepFilter 12dB → loudnorm |
| 13 | `hybrid_demucs_ft_mossformer` | Demucs_ft → MossFormer2 48K → loudnorm |
| 14 | `hybrid_mossformergan_sr` | MossFormerGAN 16K → SuperRes 48K → loudnorm |
| 15 | `superres_48k` | ClearVoice MossFormer2 super-resolution |
| 16 | `mpsenet_dns` | MP-SENet magnitude+phase 16kHz |
| 17 | `hybrid_mpsenet_sr` | MP-SENet → SuperRes 48K → loudnorm |
| 18 | `resemble_denoise` | Resemble Enhance denoise-only 44.1kHz |
| 19 | `resemble_full` | Resemble Enhance denoise + upscale 44.1kHz |
| 20 | `sepformer_wham16k` | SpeechBrain SepFormer WHAM! 16kHz |
| 21 | `original` | No processing (baseline) |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- System: `ffmpeg` (via Homebrew on macOS: `brew install ffmpeg`)

### Install

```bash
# Core + scoring
uv sync --extra score

# Everything (all 21 pipelines + scoring + visualization)
uv sync --extra all
```

### DeepFilterNet patch

DeepFilterNet requires a compatibility patch for torchaudio:

```
In .venv/lib/python3.12/site-packages/df/io.py, replace:
  from torchaudio.backend.common import AudioMetaData
with a soundfile-based fallback.
```

### Data setup

Event metadata (161 JSON files) is committed in `data/events/`. Audio files are downloaded at runtime.

## Commands

### Pipeline comparison

```bash
python -m readingroom_audio.compare
python -m readingroom_audio.compare --pipelines original deepfilter_12dB hybrid_demucs_df
python -m readingroom_audio.compare --input-dir data/audio/raw --output-dir data/audio/enhanced
```

### Batch processing (all 429 videos)

```bash
python -m readingroom_audio.batch run --pipeline hybrid_demucs_df
python -m readingroom_audio.batch run --pipeline ffmpeg_gentle --limit 2
python -m readingroom_audio.batch run --pipeline hybrid_demucs_df --resume
python -m readingroom_audio.batch status
```

### Systematic benchmark

```bash
# Full benchmark (~3 hours): stratified sample → download → VAD segment → enhance → analyze
python -m readingroom_audio.benchmark run-all

# Quick test (5 samples, 3 pipelines)
python -m readingroom_audio.benchmark run-all --target-n 5 \
    --pipelines original ffmpeg_gentle hybrid_demucs_df

# Individual phases
python -m readingroom_audio.benchmark select
python -m readingroom_audio.benchmark download
python -m readingroom_audio.benchmark extract
python -m readingroom_audio.benchmark baseline
python -m readingroom_audio.benchmark enhance
python -m readingroom_audio.benchmark analyze
```

### Listening test

```bash
python -m readingroom_audio.listening_test run-all
python -m readingroom_audio.listening_test run-all --target-n 2 --pipelines original ffmpeg_gentle
```

### Video mux

```bash
python -m readingroom_audio.mux verify --pipeline ffmpeg_gentle
python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --limit 2
python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --resume
python -m readingroom_audio.mux status
```

### Download

```bash
python -m readingroom_audio.download
python -m readingroom_audio.download --video-id VIDEO_ID
python -m readingroom_audio.download --limit 10
```

### Unified CLI

```bash
python -m readingroom_audio --help
python -m readingroom_audio benchmark run-all --target-n 5
python -m readingroom_audio batch status
```

## Quality metrics

- **DNSMOS** — Deep Noise Suppression MOS (P.808): SIG, BAK, OVRL scores 1–5
- **NISQA** — Non-Intrusive Speech Quality Assessment (chunked to 9s windows)
- **UTMOS** — UTokyo-SaruLab MOS prediction

## Project structure

```
readingroom-audio/
├── pyproject.toml
├── src/readingroom_audio/
│   ├── __init__.py
│   ├── __main__.py         # Unified CLI dispatcher
│   ├── enhance.py          # 21 enhancement pipelines
│   ├── score.py            # DNSMOS/NISQA/UTMOS scoring
│   ├── compare.py          # Pipeline comparison orchestrator
│   ├── download.py         # yt-dlp batch download
│   ├── batch.py            # Batch processor for 429 videos
│   ├── benchmark.py        # Systematic benchmark runner
│   ├── mux.py              # Video mux pipeline
│   ├── listening_test.py   # Listening test generator (HTML + MP3s)
│   ├── sampling.py         # Stratified sample selection
│   ├── segment.py          # Silero VAD segment extraction
│   └── utils.py            # ffmpeg wrappers, shared helpers
├── data/events/            # 161 event JSONs (committed)
├── docs/audio-pipeline.md  # Full documentation
└── notebooks/audio_comparison.ipynb
```

## Preferred pipeline

**`hybrid_demucs_df`** — three-stage: Demucs vocal separation → DeepFilterNet 12dB → ffmpeg loudnorm. Best subjective quality in listening tests, preserves ambient atmosphere.

See `docs/audio-pipeline.md` for full documentation, pilot results, and decision log.
