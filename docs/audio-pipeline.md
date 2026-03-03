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

All metrics are **non-intrusive** (no-reference) — essential because clean originals don't exist for YouTube archival sources. Full-reference metrics (PESQ, POLQA, ViSQOL, WARP-Q) are not applicable.

### Why these three metrics?

The combination of DNSMOS + NISQA + UTMOS provides complementary coverage:

| Metric | Strength | Weakness for our use case |
|--------|----------|--------------------------|
| DNSMOS | Decomposed SIG/BAK/OVRL — diagnoses speech distortion vs. residual noise | BAK sub-score penalizes preserved ambient (laughter, room sound) |
| NISQA | Detects coloration, discontinuity artifacts from neural enhancement | Designed for telecom degradation, not archival audio |
| UTMOS | Highest correlation with human MOS (URGENT 2024 challenge) | Single score, no diagnostic decomposition |

**Key insight for our archival use case**: DNSMOS BAK improvement should be treated as an **over-suppression warning signal**, not a success signal. A large BAK delta (e.g., >1.0) combined with SIG decrease likely means the pipeline is stripping ambient character alongside noise.

### Recommended interpretation

```
Primary quality signal:    UTMOS (best human correlation, holistic)
Diagnostic decomposition:  NISQA (noisiness, coloration, discontinuity)
Speech distortion guard:   DNSMOS SIG (catch speech artifacts)
Over-suppression detector: DNSMOS BAK delta (large increase = possible over-processing)
```

### DNSMOS (Deep Noise Suppression MOS)

Microsoft's P.808/P.835 predictor, trained on DNS Challenge crowdsourced ratings. Scores 1–5:
- **P808**: Overall quality prediction (single composite)
- **SIG**: Speech signal quality — distortion, naturalness of speech itself
- **BAK**: Background noise quality — how quiet/unobtrusive the background is
- **OVRL**: Overall quality — combines signal + background perception

**Strengths**: De facto standard in speech enhancement; decomposed scores diagnose trade-offs; fast inference.

**Limitations**: Trained on English telephony/conferencing data. BAK structurally rewards silence — a system that aggressively removes all non-speech (including desirable ambient) scores higher. Per-clip predictions are noisy; reliable comparison requires averaging over many samples. URGENT 2024 Challenge found DNSMOS has lower rank correlation (KRCC) with human MOS than UTMOS/SCOREQ.

### NISQA (Non-Intrusive Speech Quality Assessment)

TU Berlin's CNN-Self-Attention model, trained on crowdsourced MOS across communication channels.
- **MOS**: Overall quality (1–5)
- **Noisiness**: Perceived noise level (tracks perceptual noise without penalizing pleasant ambient)
- **Discontinuity**: Temporal artifacts — catches dropouts and clipping from neural processing
- **Coloration**: Spectral distortion — detects muffled/tinny artifacts from enhancement
- **Loudness**: Volume perception

**Known issue**: Internal mel spectrogram overflows on audio >10 seconds. **Fixed** in `score.py` via 9-second chunked scoring with averaging.

**Strengths**: Multi-dimensional decomposition uniquely useful for detecting enhancement artifacts. URGENT 2024: "NISQA tends to be more consistent with MOS in terms of both ranks and values" vs. DNSMOS.

### UTMOS (UTokyo-SaruLab MOS Prediction)

Ensemble of fine-tuned SSL models (wav2vec 2.0, HuBERT) for general perceptual quality.
- **Score**: Single MOS value (1–5)

**Strengths**: Highest correlation with human ratings among all tested metrics at URGENT 2024 Speech Enhancement Challenge. Strong generalization across speech codecs, TTS, and enhancement tasks.

**Limitations**: Single score with no sub-dimension breakdown. Heavier inference (SSL backbone). Originally designed for TTS evaluation.

### Metrics not used (and why)

| Metric | Type | Why excluded |
|--------|------|-------------|
| PESQ (P.862) | Full-reference | Requires clean original — not available |
| POLQA (P.863) | Full-reference | Same — successor to PESQ |
| ViSQOL | Full-reference | Google's perceptual metric — same limitation |
| WARP-Q | Full-reference | Designed for neural codecs but still needs reference |
| SpeechLMScore | No-reference | Low discriminative power between enhancement methods (URGENT 2024) |
| SCOREQ | No-reference | Strong performer at URGENT 2024 but not yet in torchmetrics; candidate for future addition |

### References

- [DNSMOS P.835](https://arxiv.org/abs/2110.01763) — Reddy et al., 2021
- [NISQA v2.0](https://github.com/gabrielmittag/NISQA) — Mittag et al., TU Berlin
- [UTMOS](https://arxiv.org/abs/2204.02152) — Saeki et al., UTokyo
- [URGENT 2024 Challenge](https://arxiv.org/html/2506.01611v1) — Comparative evaluation of all metrics

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
6. All NISQA scores failed due to audio length limitation (fixed via 9s chunked scoring)

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
- [x] Run full benchmark on 40 stratified samples (9 of 21 pipelines scored — 12 had import/dependency failures)
- [x] Per-content-type pipeline selection — per-stratum Friedman tests + series heatmap chart
- [x] A/B listening test framework (`listening_test.py` — 6-phase CLI with HTML audio player)
- [x] Audio preview page (`benchmark preview` — 8 diverse segments, interactive playback)
- [x] Benchmark export with PNG charts and audio samples (`benchmark export`)
- [x] Video mux pipeline (`mux.py` — remux enhanced audio into original video)
- [x] Multi-seed sensitivity analysis (`benchmark sensitivity`)

### Not Started
- [ ] Explore ensemble approaches (combine pipeline strengths per content type)
- [ ] Formal MUSHRA/AB listening test at scale (current framework is informal comparison)

## Systematic Benchmark

### Overview

The pilot tested only 3 files — statistically insufficient to pick a best pipeline. The systematic benchmark tests 9 enhancement pipelines (of 21 defined — 12 failed due to dependency issues) on 40 stratified samples drawn from 161 events across 15 series, 10 years, and multiple formats. Uses proper statistical tests (Friedman + post-hoc Wilcoxon) to find the best pipeline overall and per content type.

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

Tests are run for each metric independently (DNSMOS OVRL, UTMOS, NISQA MOS):

1. **Friedman test** — non-parametric repeated-measures across pipelines per metric
2. **Post-hoc Wilcoxon** signed-rank with Bonferroni correction (28 pairs, α≈0.0018)
3. **Bootstrap 95% CIs** — 10,000 resamples on pipeline means (percentile method)
4. **Cross-metric agreement** — Spearman ρ between improvement deltas across metric pairs
5. **Per-stratum** — same tests per series_group to detect content-type interactions
6. **Effect sizes** — rank-biserial correlation (practical significance: |r| > 0.3)

| Test | Purpose | Threshold |
|------|---------|-----------|
| Friedman | Overall pipeline differences | p < 0.05 |
| Wilcoxon (post-hoc) | Pairwise comparisons | Bonferroni-corrected α |
| Bootstrap CI | Uncertainty on means | 95% percentile interval |
| Spearman ρ | Cross-metric agreement | \|ρ\| < 0.4 → weak agreement warning |
| Rank-biserial r | Effect size | \|r\| > 0.3 → practical significance |

### Benchmark Commands

```bash
# Full benchmark (~3 hours)
uv run python -m readingroom_audio.benchmark run-all

# Quick test (5 samples, 3 pipelines, DNSMOS only, 15s segments)
uv run python -m readingroom_audio.benchmark run-all --quick

# Custom pipeline selection
uv run python -m readingroom_audio.benchmark run-all --target-n 5 \
    --pipelines original ffmpeg_gentle hybrid_demucs_df

# Individual phases (all resumable)
uv run python -m readingroom_audio.benchmark select
uv run python -m readingroom_audio.benchmark download
uv run python -m readingroom_audio.benchmark extract
uv run python -m readingroom_audio.benchmark baseline
uv run python -m readingroom_audio.benchmark enhance
uv run python -m readingroom_audio.benchmark analyze

# Export report with PNG charts + audio samples
uv run python -m readingroom_audio.benchmark export [--output-dir ...] [--n-samples 3]

# Generate interactive HTML audio preview page
uv run python -m readingroom_audio.benchmark preview [--output-dir ...] [--n-samples 8]

# Multi-seed sensitivity analysis
uv run python -m readingroom_audio.benchmark sensitivity --target-n 40 --seeds 42 123 456 789 1337
```

### Outputs

- `data/audio/benchmark_manifest.json` — sample selection + status tracking
- `data/audio/benchmark_results.json` — all pipeline scores per segment (40 segments × 9 pipelines)
- `data/audio/benchmark_report.md` — statistical analysis report with per-stratum breakdown
- `data/audio/benchmark_charts/` — 9 Altair HTML visualizations (boxplot, heatmap, CI forest, etc.)
- `docs/benchmark-report/` — exported report with PNG charts + audio samples (via `export`)
- `docs/audio-preview/` — interactive HTML audio preview page (via `preview`)

## Listening Test

Interactive HTML audio comparison page for side-by-side pipeline evaluation. Separate from the benchmark — uses its own 10-sample selection with 30-second segments.

### Pipeline

```
select   → listening-test/manifest.json (10 stratified samples)
download → raw/*.m4a (shared cache with benchmark)
extract  → listening-test/segments/*.wav (30s via VAD)
enhance  → listening-test/enhanced/{pipeline}/*.wav
score    → listening-test/scores.json (DNSMOS + NISQA + UTMOS)
build    → docs/listening-test/index.html + audio/*.mp3
```

### Commands

```bash
# Full pipeline (all 6 phases)
uv run python -m readingroom_audio.listening_test run-all

# Quick test
uv run python -m readingroom_audio.listening_test run-all --target-n 2 \
    --pipelines original ffmpeg_gentle hybrid_demucs_df

# Individual phases
uv run python -m readingroom_audio.listening_test select
uv run python -m readingroom_audio.listening_test download
uv run python -m readingroom_audio.listening_test extract
uv run python -m readingroom_audio.listening_test enhance
uv run python -m readingroom_audio.listening_test score
uv run python -m readingroom_audio.listening_test build
```

### Output

Self-contained HTML page at `docs/listening-test/index.html` with:
- Summary table (mean OVRL, UTMOS per pipeline)
- Pipeline descriptions
- Per-sample cards with audio players (singleton playback — clicking one stops the previous)
- DNSMOS SIG/BAK/OVRL + UTMOS scores with visual bars
- `hybrid_demucs_df` highlighted as recommended pipeline
- Mobile-responsive, `preload="none"` for fast page load

## Audio Preview (Benchmark)

Lightweight audio preview page generated directly from benchmark results (no separate pipeline needed). Selects 8 diverse segments from the 40 benchmarked samples.

### Commands

```bash
uv run python -m readingroom_audio.benchmark preview
uv run python -m readingroom_audio.benchmark preview --n-samples 12
```

### Output

`docs/audio-preview/index.html` — same interactive player design as the listening test, but sourced from benchmark data. Segments selected for diversity across quality range (DNSMOS OVRL) and content types (series_group).

## Video Mux

Remux enhanced audio back into the original YouTube video stream. Downloads video-only (no audio), verifies duration alignment, and produces MP4 with AAC audio.

### Per-video flow

```
identify enhanced FLAC (from batch processing)
  → download video-only stream (yt-dlp, best quality)
  → verify duration match (±0.5s tolerance)
  → mux video + enhanced audio → MP4 (AAC 192kbps)
  → update mux_status.json
```

### Commands

```bash
uv run python -m readingroom_audio.mux run --pipeline hybrid_demucs_df
uv run python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --resume
uv run python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --limit 10
uv run python -m readingroom_audio.mux verify --pipeline ffmpeg_gentle
uv run python -m readingroom_audio.mux status
```

### Output structure

```
data/video/
  raw/                          # Video-only streams (no audio)
  muxed/{pipeline}/*.mp4        # Final video with enhanced audio
  mux_status.json               # Per-video status tracking
```

## Evaluation Methodology

### Bias Mitigation

#### 1. Metric Selection Bias

Using a single metric risks confirmation bias — the chosen metric may favor certain artifacts
while missing others. This evaluation uses three complementary non-intrusive metrics:

- **UTMOS** (primary quality): Highest correlation with human MOS in URGENT 2024 challenge.
  Single holistic score, less susceptible to component-level artifacts.
- **DNSMOS** (diagnostic): P.808-based; decomposes into SIG (speech), BAK (background), OVRL (overall).
  Useful for detecting speech distortion vs background improvement trade-offs.
- **NISQA** (artifact decomposition): Noisiness, discontinuity, coloration, loudness.
  Catches specific artifact types (musical noise, tonal shift) that holistic metrics may miss.

Cross-metric agreement (Spearman ρ between improvement deltas) is computed automatically.
Weak agreement (|ρ| < 0.4) triggers a warning — pipelines should be evaluated per-metric
rather than relying on any single ranking.

#### 2. Sample Selection Bias

The 429-video population is heterogeneous (Thai/English, talks/performances, 2010–2019 equipment).
Stratified sampling by `series_group` and `era` ensures the benchmark sample (n=40) reflects this
diversity proportionally. Multi-seed sensitivity analysis (`benchmark sensitivity`) tests whether
pipeline rankings are stable across different random samples.

#### 3. Confirmation Bias (BAK Trap)

DNSMOS BAK measures background "cleanliness" — higher BAK means less background noise. For
archival audio where ambient atmosphere matters, a large BAK delta with negative SIG delta
signals over-suppression (noise removed, but speech quality degraded and room character lost).
The evaluation explicitly flags BAK delta > 1.0 combined with SIG delta < 0 as a warning.

#### 4. Multiple Comparisons

With 9 pipelines and 3 metrics, naive pairwise testing inflates false discovery rate. Bonferroni
correction is applied to Wilcoxon post-hoc tests, and practical significance requires both
statistical significance (corrected p-value) and meaningful effect size (rank-biserial |r| > 0.3).

### Limitations and Known Biases

1. **Language bias**: All three metrics (DNSMOS, NISQA, UTMOS) are trained primarily on English
   speech. ~40% of the Reading Room content is Thai — metric calibration may differ for Thai
   speech, particularly for tonal distinctions.

2. **VAD segment bias**: Benchmark segments are extracted using Silero VAD, which selects
   speech-dense regions. This underrepresents non-speech content (music, ambient sections)
   and may over-weight speech quality in the evaluation.

3. **No subjective validation at scale**: The 3-file pilot included subjective listening
   comparison, but the 40-sample benchmark relies entirely on objective metrics.
   No formal listening test (MUSHRA, AB preference) has been conducted at scale.

4. **Single-pass scoring**: Each segment is scored once per metric with no measurement
   reliability check (test-retest). DNSMOS and NISQA are deterministic, but UTMOS may
   have minor variance from model loading state.

5. **Temporal equipment evolution**: Recording equipment changed between 2010–2019.
   Earlier recordings may have fundamentally different noise profiles that some pipelines
   handle better than others. The `era` stratification partially addresses this but
   doesn't fully control for equipment-specific effects.

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
uv run python -m readingroom_audio.compare --pipelines original hybrid_demucs_df deepfilter_full

# Systematic benchmark (40 stratified samples)
uv run python -m readingroom_audio.benchmark run-all
uv run python -m readingroom_audio.benchmark preview   # HTML audio preview

# Listening test (10 samples, interactive HTML)
uv run python -m readingroom_audio.listening_test run-all

# Batch processing (all 429 videos)
uv run python -m readingroom_audio.batch run --pipeline hybrid_demucs_df --resume
uv run python -m readingroom_audio.batch status

# Video mux (remux enhanced audio into video)
uv run python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --resume
uv run python -m readingroom_audio.mux status

# Unified CLI (all modules accessible)
uv run python -m readingroom_audio --help
```

### Interactive Analysis

Open `notebooks/audio_comparison.ipynb` for:
- Visual comparison of DNSMOS scores across pipelines
- Multi-dimensional quality profile (line chart across SIG/BAK/OVRL/P.808)
- Per-file heatmaps and improvement-over-baseline charts
- Audio playback widgets for A/B listening tests (30s preview)
- Statistical analysis of benchmark data (when available): box plots, bootstrap CIs, cross-metric scatter, sample representativeness

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
| 2026-03-03 | Documented metric rationale | DNSMOS BAK penalizes ambient; UTMOS recommended as primary quality signal; see Quality Metrics section |
| 2026-03-03 | Score model caching | Cache DNSMOS/NISQA/UTMOS models at module level (like enhance.py) — saves ~7 min over 429 videos |
| 2026-03-03 | FLAC compression 8→5 | ~3% larger files but significantly faster encoding for batch processing |
| 2026-03-03 | Multi-metric analysis | Run Friedman/Wilcoxon for DNSMOS OVRL, UTMOS, and NISQA MOS independently — avoids single-metric confirmation bias |
| 2026-03-03 | Bootstrap 95% CIs | Report uncertainty on pipeline means (10K resamples) — bare means without CIs invite over-interpretation |
| 2026-03-03 | Cross-metric agreement | Spearman ρ between improvement deltas flags when metrics disagree on pipeline ranking |
| 2026-03-03 | Multi-seed sensitivity | Test sampling stability across 5 seeds — ensures pipeline rankings aren't artifacts of a single random draw |
| 2026-03-03 | Listening test HTML page | Interactive A/B comparison via GitHub Pages — GitHub README doesn't support `<audio>` tags |
| 2026-03-03 | Video mux pipeline | Remux enhanced FLAC back into original video stream for distribution (AAC 192kbps) |
| 2026-03-03 | Fail-fast tracker | Auto-disable pipelines after 2 consecutive same-category failures — prevents wasting hours on broken dependencies |
| 2026-03-03 | Benchmark structured logger | JSONL per-run logs for crash safety and post-hoc analysis of enhance/score timings |
| 2026-03-04 | Benchmark export subcommand | Export report with PNG charts (Altair → static images) and representative audio samples for offline review |
| 2026-03-04 | Benchmark preview subcommand | Lightweight HTML audio preview from existing benchmark data — 8 diverse segments, no separate pipeline needed |
| 2026-03-04 | 9 of 21 pipelines scored | 12 pipelines failed due to dependency issues (TorchCodec, model weights); 9 provide sufficient coverage of approach families |
