# Audio Enhancement Pipeline

## Project Overview

The Reading Room BKK retrospective project has 429 YouTube videos (341 hours, 2010–2019) documenting talks, screenings, performances, and discussions at an independent art/intellectual space in Bangkok. Most recordings have significant audio quality issues — room noise, echo, low volume, clipping, and background interference.

This pipeline aims to improve audio quality for archival purposes and potential podcast/listening distribution, while preserving the ambient character (laughter, room atmosphere) that makes these recordings authentic.

## Source Material

| Property | Value |
|----------|-------|
| Videos | 429 |
| Total duration | 341 hours |
| Period | 2010–2019 |
| Source quality | ~128kbps AAC from YouTube |
| Content types | Talks, panel discussions, film screenings Q&A, performances, Thai/English/mixed |
| Common issues | Room noise, echo/reverb, low volume, clipping, audience noise |

### Content type distribution (estimated)

- **Talks/lectures** (~60%): Single or dual speaker, moderate room noise
- **Panel discussions** (~15%): Multiple speakers, varying mic distances
- **Film screenings + Q&A** (~15%): Often poor audio during Q&A segments
- **Performances/events** (~10%): Music, ambient sound, mixed content

## Enhancement Pipelines Tested

### 1. DeepFilterNet3 — Full (`deepfilter_full`)
- **Method**: Real-time noise suppression using deep filtering
- **Settings**: No attenuation limit (maximum suppression)
- **Pros**: Strong noise reduction, fast processing
- **Cons**: Can sound over-processed, removes ambient atmosphere

### 2. DeepFilterNet3 — 12dB (`deepfilter_12dB`)
- **Method**: Same as above with 12dB attenuation limit
- **Settings**: `atten_lim_db=12`
- **Pros**: More natural than full, preserves some ambient
- **Cons**: Less noise reduction than full mode

### 3. ClearVoice MossFormer2 48kHz (`mossformer2_48k`)
- **Method**: Neural speech enhancement at 48kHz
- **Pros**: High-quality output, preserves speech naturalness
- **Cons**: Slower processing, large model

### 4. ClearVoice FRCRN 16kHz (`frcrn_16k`)
- **Method**: Neural speech enhancement at 16kHz
- **Pros**: Good noise reduction, fast
- **Cons**: Downsamples to 16kHz (loses high-frequency detail)

### 5. Demucs htdemucs (`demucs_vocals`)
- **Method**: Music source separation — isolates vocal stem
- **Pros**: Excellent at separating speech from background
- **Cons**: Designed for music, may produce artifacts on pure speech

### 6. ffmpeg Gentle (`ffmpeg_gentle`)
- **Method**: Traditional signal processing chain
- **Settings**: `highpass=80Hz → afftdn(nf=-25,nr=10) → acompressor → loudnorm`
- **Pros**: No ML required, fast, deterministic, preserves ambient
- **Cons**: Limited noise reduction compared to ML methods

### 7. Hybrid Demucs + DeepFilter (`hybrid_demucs_df`)
- **Method**: Three-stage pipeline
  1. Demucs vocal separation (isolate speech)
  2. DeepFilterNet 12dB (gentle noise suppression on vocals)
  3. ffmpeg loudnorm (normalize loudness)
- **Pros**: Best overall quality in pilot testing, preserves ambient atmosphere
- **Cons**: Slowest processing (two ML models + ffmpeg)
- **Status**: User's current preferred pipeline

## Quality Metrics

### DNSMOS (Deep Noise Suppression MOS)
Non-intrusive metric based on ITU-T P.808. Scores 1–5:
- **P808**: Overall quality prediction
- **SIG**: Speech signal quality (distortion, naturalness)
- **BAK**: Background noise quality (suppression effectiveness)
- **OVRL**: Overall quality (combines signal + background)

### NISQA (Non-Intrusive Speech Quality Assessment)
Predicts MOS and sub-dimensions:
- **MOS**: Overall quality (1–5)
- **Noisiness**: Background noise level
- **Discontinuity**: Temporal artifacts
- **Coloration**: Spectral distortion
- **Loudness**: Volume perception

**Known issue**: NISQA fails on audio >10 seconds with "Maximum number of mel spectrogram windows exceeded." Needs chunked scoring implementation.

## Pilot Results

Tested 3 representative files across all 7 pipelines:

| File | Description | Duration |
|------|-------------|----------|
| `01_earliest_talk` | Early recording, poor mic | ~60 min |
| `02_screening_talk` | Film screening Q&A | ~60 min |
| `03_thai_talk` | Thai language talk | ~60 min |

### DNSMOS Scores (scored on 30–90s segment)

| Pipeline | 01 OVRL | 02 OVRL | 03 OVRL | Avg OVRL |
|----------|---------|---------|---------|----------|
| original | 1.09 | 1.10 | 1.13 | 1.11 |
| deepfilter_full | 2.12 | 2.19 | 2.21 | **2.17** |
| deepfilter_12dB | 1.29 | 1.29 | 1.30 | 1.30 |
| mossformer2_48k | 1.96 | 1.82 | 1.81 | 1.86 |
| frcrn_16k | 2.06 | 1.82 | 1.94 | 1.94 |
| demucs_vocals | 1.11 | 1.29 | 1.39 | 1.26 |
| ffmpeg_gentle | 1.09 | 1.08 | 1.31 | 1.16 |
| hybrid_demucs_df | 1.45 | 1.64 | **2.20** | 1.76 |

### Key Findings

1. **deepfilter_full** scores highest on DNSMOS overall, but sounds over-processed in listening tests
2. **hybrid_demucs_df** shows strong improvement especially on Thai talk (2.20 OVRL), with better subjective quality
3. **ffmpeg_gentle** provides minimal improvement — ML methods significantly outperform traditional DSP
4. **demucs_vocals** alone doesn't help much — needs post-processing (as in hybrid pipeline)
5. DNSMOS scores don't fully capture perceived quality — listening tests are essential
6. All NISQA scores failed due to audio length limitation

## Expanded Pipelines (Phase 1 — zero new installs)

### 8. MossFormerGAN 16kHz (`mossformergan_16k`)
- **Method**: ClearVoice GAN-based speech enhancement (PESQ 3.57)
- **Pros**: Perceptually superior to regression models, fast
- **Cons**: 16kHz output

### 9. DeepFilterNet3 — 6dB (`deepfilter_6dB`)
- **Method**: Same as deepfilter_12dB with minimal attenuation
- **Settings**: `atten_lim_db=6` — preserves maximum ambient sound

### 10. DeepFilterNet3 — 18dB (`deepfilter_18dB`)
- **Method**: Same with strong attenuation
- **Settings**: `atten_lim_db=18` — more aggressive than 12dB, less than full

### 11. Demucs htdemucs_ft (`demucs_ft_vocals`)
- **Method**: Fine-tuned Demucs model for better vocal separation
- **Pros**: Higher quality than base htdemucs
- **Cons**: ~4x slower processing

### 12. SuperRes 48kHz (`superres_48k`)
- **Method**: ClearVoice MossFormer2 speech super-resolution
- **Pros**: Recovers high-frequency detail from compressed/downsampled audio
- **Cons**: Not a denoiser — best used after denoising stage

### 13. Hybrid MossFormerGAN+SR (`hybrid_mossformergan_sr`)
- **Method**: MossFormerGAN 16K → SuperRes 48K → loudnorm
- **Pros**: GAN-quality denoising with wideband output

### 14. Hybrid Demucs_ft+DF (`hybrid_demucs_ft_df`)
- **Method**: Demucs_ft vocals → DeepFilter 12dB → loudnorm
- **Pros**: Premium vocal separation + gentle noise suppression (like hybrid_demucs_df but with fine-tuned model)

### 15. Hybrid Demucs_ft+MossFormer (`hybrid_demucs_ft_mossformer`)
- **Method**: Demucs_ft vocals → MossFormer2 48K → loudnorm
- **Pros**: Best vocal isolation + neural enhancement, 48kHz output

## Expanded Pipelines (Phase 2 — new packages)

### 16. MP-SENet (`mpsenet_dns`)
- **Method**: Magnitude+phase speech enhancement (16kHz)
- **Package**: MPSENet
- **Pros**: Joint mag-phase processing for higher quality

### 17. Hybrid MP-SENet+SR (`hybrid_mpsenet_sr`)
- **Method**: MP-SENet → SuperRes 48K → loudnorm

### 18. Resemble Denoise (`resemble_denoise`)
- **Method**: Resemble Enhance denoise-only mode (44.1kHz)
- **Package**: resemble-enhance

### 19. Resemble Full (`resemble_full`)
- **Method**: Resemble Enhance denoise + enhance/upscale (44.1kHz)
- **Package**: resemble-enhance

### 20. SepFormer WHAM! (`sepformer_wham16k`)
- **Method**: SpeechBrain SepFormer trained on WHAM! noise (16kHz)
- **Package**: speechbrain

## Batch Processing

Process all 429 videos with the winning pipeline, output as FLAC (lossless).

### Per-video flow

```
download m4a (yt-dlp, skip if exists)
  → convert to WAV 48kHz (ffmpeg, temp file)
  → enhance with chosen pipeline
  → encode to FLAC (lossless, ~60% smaller than WAV)
  → delete WAV intermediates
  → update batch_status.json
```

### Output structure

```
data/audio/
  raw/                        # Downloaded m4a files
  enhanced_final/{pipeline}/  # Enhanced FLAC files
  batch_status.json           # Per-video status tracking
```

### Commands

```bash
uv run python -m readingroom_audio.batch run --pipeline hybrid_demucs_df
uv run python -m readingroom_audio.batch run --pipeline hybrid_demucs_df --limit 10
uv run python -m readingroom_audio.batch run --pipeline hybrid_demucs_df --resume
uv run python -m readingroom_audio.batch status
```

### Storage estimate (all 429 videos)

| Data | Size |
|------|------|
| m4a originals (128kbps) | ~19 GB |
| FLAC enhanced (1 pipeline) | ~70 GB |
| WAV intermediates | 0 (deleted after FLAC) |

## Research Roadmap

### Completed
- [x] Set up 7 enhancement pipelines
- [x] Pilot comparison on 3 representative files
- [x] DNSMOS automated scoring
- [x] Organized modular codebase (`src/readingroom_audio/`)
- [x] Fix NISQA scoring (chunk audio into 9s windows)
- [x] Add UTMOS scorer (graceful fallback)
- [x] Systematic benchmark framework (stratified sampling, statistical tests)
- [x] Expand to 21 pipelines (8 Phase 1 + 5 Phase 2)
- [x] Batch processing script for all 429 files
- [x] Resemble Enhance integration (installed with --no-deps)

### In Progress
- [ ] Run full benchmark on 40 stratified samples across all 21 pipelines

### Planned
- [ ] Per-content-type pipeline selection (based on benchmark results)
- [ ] A/B listening test framework
- [ ] Explore ensemble approaches (combine pipeline strengths)

## Systematic Benchmark

### Overview

The pilot tested only 3 files — statistically insufficient to pick a best pipeline. The systematic benchmark tests all 8 pipelines on ~40 stratified samples drawn from 161 events across 15 series, 10 years, and multiple formats. Uses proper statistical tests (Friedman + post-hoc Wilcoxon) to find the best pipeline overall and per content type.

### Sampling Strategy

Events are classified into 8 series groups and 3 eras, then ~40 are selected via proportional stratified sampling:

| Group | Includes | Pop | Sample |
|-------|----------|-----|--------|
| `other` | Other | 49 | 12 |
| `readrink` | Readrink, Book Club | 45 | 11 |
| `screening` | Screening & Talk, Filmvirus | 24 | 6 |
| `talk_series` | Talk Series, Artist Talk, This is Not Fiction, Solidarities | 12 | 3 |
| `definitions` | Definitions Series, Right Here Right Now | 9 | 2 |
| `sleepover` | Sleepover | 8 | 2 |
| `night_school` | Night School | 7 | 2 |
| `re_reading` | re:reading, Reading Group | 7 | 2 |

For each selected event, a 45-second speech-active segment is extracted using Silero VAD to avoid scoring silence/applause.

### Pipeline Flow

```
select   → benchmark_manifest.json (40 entries with strata labels)
download → benchmark_downloads/*.m4a (yt-dlp, resumable)
extract  → benchmark_segments/E{NNN}_{vid}.wav (45s via VAD)
baseline → update manifest with baseline_scores
enhance  → benchmark_enhanced/{pipeline}/*.wav + scoring
analyze  → benchmark_results.json + benchmark_report.md + charts/
```

### Statistical Analysis

1. **Friedman test** — non-parametric repeated-measures across 8 pipelines on DNSMOS OVRL
2. **Post-hoc Wilcoxon** signed-rank with Bonferroni correction (28 pairs, α≈0.0018)
3. **Per-stratum** — same tests per series_group to detect content-type interactions
4. **Effect sizes** — rank-biserial correlation

### Benchmark Commands

```bash
# Full benchmark (~3 hours)
uv run python -m readingroom_audio.benchmark run-all

# Quick test (5 samples, 3 pipelines)
uv run python -m readingroom_audio.benchmark run-all --target-n 5 \
    --pipelines original ffmpeg_gentle hybrid_demucs_df

# Individual phases (all resumable)
uv run python -m readingroom_audio.benchmark select
uv run python -m readingroom_audio.benchmark download
uv run python -m readingroom_audio.benchmark extract
uv run python -m readingroom_audio.benchmark baseline
uv run python -m readingroom_audio.benchmark enhance
uv run python -m readingroom_audio.benchmark analyze
```

### Outputs

- `data/audio/benchmark_manifest.json` — sample selection + status tracking
- `data/audio/benchmark_results.json` — all pipeline scores per segment
- `data/audio/benchmark_report.md` — statistical analysis report
- `data/audio/benchmark_charts/` — Altair HTML visualizations

## How to Run

### Prerequisites

```bash
# Install dependencies (Python 3.12 project)
uv sync

# Patch DeepFilterNet (required for torchaudio compatibility)
# In .venv/lib/python3.12/site-packages/df/io.py, replace:
#   from torchaudio.backend.common import AudioMetaData
# with soundfile-based fallback (see project setup docs)

# System dependency
brew install pango ffmpeg
```

### Commands

```bash
# Download audio for specific video
uv run python -m readingroom_audio.download --video-id VIDEO_ID

# Run full pipeline comparison on pilot files
uv run python -m readingroom_audio.compare

# Run specific pipelines only
uv run python -m readingroom_audio.compare --pipelines original hybrid_demucs_df deepfilter_full

# Compare on custom input directory
uv run python -m readingroom_audio.compare \
    --input-dir data/audio/raw \
    --output-dir data/audio/enhanced \
    --report data/audio/quality_report.json
```

### Interactive Analysis

Open `notebooks/audio_comparison.ipynb` for:
- Visual comparison of DNSMOS scores across pipelines
- Radar charts for multi-dimensional quality assessment
- Audio playback widgets for A/B listening tests

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-03-02 | Pilot 7 pipelines on 3 files | Establish baseline before committing to approach |
| 2025-03-02 | Prefer `hybrid_demucs_df` | Best subjective quality, preserves ambient atmosphere |
| 2025-03-02 | Use DNSMOS over NISQA | NISQA fails on long audio; DNSMOS works with truncation |
| 2026-03-02 | Unified Python 3.12 venv | Eliminate venv switching; 3.12 supports all dependencies |
| 2026-03-02 | Modular `src/readingroom_audio/` package | Separate concerns: enhance, score, compare, download |
| 2026-03-03 | Systematic benchmark framework | 3-file pilot insufficient; need 40 stratified samples + statistical tests |
| 2026-03-03 | Fixed NISQA chunking (9s windows) | Pilot NISQA failed on all files due to mel spectrogram overflow |
| 2026-03-03 | Added UTMOS scorer | Third metric for cross-validation of quality assessment |
| 2026-03-03 | Expand to 21 pipelines | Test more variations (DeepFilter attenuation sweep, htdemucs_ft, GAN models, super-resolution, new packages) to find truly best approach |
| 2026-03-03 | Add batch processing | Enable processing all 429 videos once winning pipeline is determined |
| 2026-03-03 | FLAC output for batch | Lossless compression (~60% smaller than WAV) for archival output |
