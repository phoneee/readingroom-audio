# Audio Enhancement for Low-Quality Archival Recordings

A systematic evaluation of audio enhancement pipelines for 429 YouTube recordings (341 hours) from [The Reading Room BKK](https://www.facebook.com/TheReadingRoomBKK/). Fourteen ML and DSP pipelines benchmarked across 161 events (one representative clip per event, from 429 total clips) using three perceptual quality metrics. Key finding: hybrid multi-stage pipelines significantly outperform single-model approaches on real-world degraded audio, but the highest-scoring pipelines strip the ambient atmosphere that gives archival recordings their documentary value.

## The Problem

The Reading Room BKK was an independent art and intellectual space in Bangkok (2010–2019) that hosted lectures, discussions, screenings, and performances — publicly valuable discourse recorded informally with consumer cameras, no professional audio setup. The 429 videos were uploaded to YouTube as 128 kbps AAC: room noise, echo, clipping, low volume, and variable recording quality across a decade of events.

This makes it an ideal real-world test case. Most speech enhancement research benchmarks on synthetic noise added to clean recordings. Here, the degradation is genuine — and the content is worth preserving. The challenge: enhance speech clarity without destroying the ambient character (laughter, audience reactions, room atmosphere) that gives archival recordings documentary value.

## Key Findings

Pipeline quality scores across 161 segments (higher = better). All pipelines significantly improve on the original (Friedman p < 0.001, Wilcoxon pairwise with Bonferroni correction).

| Pipeline | OVRL | SIG | BAK | P.808 | NISQA | ΔOVRL |
|----------|------|-----|-----|-------|-------|-------|
| mossformergan_16k | 2.50 | 3.00 | 3.29 | 3.21 | 2.30 | +1.25 |
| deepfilter_full | 2.50 | 2.89 | 3.62 | 3.47 | 2.24 | +1.26 |
| mossformer2_48k | 2.22 | 2.76 | 2.93 | 3.20 | 2.07 | +0.97 |
| hybrid_demucs_ft_df | 2.20 | 2.84 | 2.77 | 3.14 | 1.79 | +0.98 |
| hybrid_demucs_ft_mossformer | 2.15 | 2.78 | 2.78 | 3.11 | 1.73 | +0.94 |
| frcrn_16k | 2.10 | 2.65 | 2.65 | 3.06 | 2.09 | +0.86 |
| deepfilter_18dB | 2.11 | 2.66 | 2.61 | 3.14 | 2.02 | +0.89 |
| **hybrid_demucs_df** | **2.08** | **2.71** | **2.47** | **3.12** | **1.81** | **+0.83** |
| deepfilter_12dB | 1.83 | 2.34 | 2.15 | 3.00 | 1.82 | +0.59 |
| demucs_ft_vocals | 1.72 | 2.18 | 2.12 | 2.94 | 1.48 | +0.50 |
| demucs_vocals | 1.43 | 1.75 | 1.61 | 2.82 | 1.40 | +0.18 |
| deepfilter_6dB | 1.41 | 1.78 | 1.57 | 2.79 | 1.48 | +0.20 |
| ffmpeg_gentle | 1.31 | 1.61 | 1.43 | 2.70 | 1.50 | +0.06 |
| original | 1.25 | 1.47 | 1.33 | 2.64 | 1.22 | — |

_OVRL/SIG/BAK = DNSMOS sub-scores, P.808 = ITU-T P.808, NISQA = speech quality MOS. UTMOS omitted (saturated at ~1.27 across all pipelines)._

![Pipeline DNSMOS OVRL box plots](docs/benchmark-report/images/pipeline_boxplot.png)

## Signal vs Background Tradeoff

Aggressive pipelines push background suppression (BAK) high but can strip ambient atmosphere along with noise. The upper-right corner represents the ideal: high signal quality with strong background suppression. Note how `deepfilter_full` achieves the highest BAK (3.62) but at the cost of removing room character that hybrid pipelines preserve.

![Signal vs Background Tradeoff](docs/benchmark-report/images/sig_bak_tradeoff.png)

## Default Pipeline

**`hybrid_demucs_df`** — Demucs vocal separation → DeepFilterNet 12 dB → ffmpeg loudnorm.

Not the highest DNSMOS OVRL (8th of 14), but selected for archival use because:

1. **Preserves ambient atmosphere.** BAK of 2.47 vs deepfilter_full's 3.62 — it removes steady-state noise without silencing audience laughter, room ambience, and the acoustic texture of the space.
2. **Strong signal clarity.** SIG of 2.71 is competitive with the top scorers (mossformergan_16k: 3.00), meaning speech is genuinely clearer.
3. **Minimal artifacts.** Demucs separates voice from noise at the source level; DeepFilterNet cleans residual hum. This two-stage approach avoids the "underwater" quality of single-model aggressive enhancement.

The highest scorers (`deepfilter_full`, `mossformergan_16k`) optimize for clean-room speech quality — appropriate for podcast production, but not for archival recordings where the ambient character *is* part of the documentary record.

> **Content-type caveat**: Demucs-based pipelines use source separation that treats non-speech audio as "background noise." This is destructive for:
> - **Screenings** (film audio played through speakers) → use `deepfilter_12dB` instead
> - **Performances** (music, sound art, capoeira) → use `ffmpeg_gentle` instead
>
> Use `--auto-pipeline` in batch mode to automatically select the right pipeline per content type.

### Known Limitation: Within-File Content Changes

Some events have content transitions within a single recording — e.g. a lecture intro → film screening → Q&A discussion. The current pipeline assignment is per-event based on the dominant content type from event metadata. Future work: VAD + content classifier for per-segment pipeline switching.

## Methodology

- **N = 161** segments (one representative clip per event, longest video selected from 429 total clips), stratified across 8 series groups, 4 content types, and 3 recording eras
- **Segment selection**: 45-second speech-active windows identified by Silero VAD
- **3 metric families, 10 sub-metrics**: DNSMOS (OVRL, SIG, BAK, P.808), NISQA (MOS, Noisiness, Coloration, Discontinuity, Loudness), UTMOS
- **Statistical tests**: Friedman omnibus test → Wilcoxon signed-rank pairwise comparisons with Bonferroni correction (α = 0.0005 for 91 pairs)
- **Confidence intervals**: 95% bootstrap CIs (10,000 resamples)
- **Cross-metric agreement**: DNSMOS↔NISQA show strong correlation; UTMOS saturates near floor (~1.27) on this quality range, providing no discrimination

![Confidence Intervals — pipeline means with 95% bootstrap CIs](docs/benchmark-report/images/ci_forest_plot.png)

## Pipelines Tested

**14 benchmarked** (9 core + 5 additional), from a catalog of 21 implemented:

| Category | Pipelines | Method |
|----------|-----------|--------|
| Single-model ML | deepfilter_{full,18dB,12dB,6dB} | DeepFilterNet3 at varying attenuation limits |
| Single-model ML | mossformergan_16k, mossformer2_48k, frcrn_16k | ClearVoice speech enhancement models |
| Source separation | demucs_vocals, demucs_ft_vocals | Demucs htdemucs vocal isolation |
| Hybrid multi-stage | hybrid_demucs_df, hybrid_demucs_ft_df | Demucs → DeepFilter → loudnorm |
| Hybrid multi-stage | hybrid_demucs_ft_mossformer | Demucs_ft → MossFormer2 → loudnorm |
| DSP-only | ffmpeg_gentle | High-pass + FFT denoise + compression + loudnorm |

7 additional pipelines implemented but not in the main benchmark: superres_48k, hybrid_mossformergan_sr, mpsenet_dns, hybrid_mpsenet_sr, resemble_denoise, resemble_full, sepformer_wham16k.

## Live Demo

- **[Audio Preview](https://phoneee.github.io/readingroom-audio/audio-preview/)** — 161 segments × 9 pipelines, interactive per-metric highlighting
- **[Benchmark Report](https://phoneee.github.io/readingroom-audio/benchmark-report/)** — full statistical analysis, charts, pairwise comparisons

## Reproducing the Benchmark

```bash
# Prerequisites: Python 3.12+, uv, ffmpeg
git clone https://github.com/phoneee/readingroom-audio && cd readingroom-audio
uv sync --extra all

# Full benchmark (161 samples × 14 pipelines)
python -m readingroom_audio.benchmark run-all

# Quick test (5 samples, 3 pipelines)
python -m readingroom_audio.benchmark run-all --target-n 5 \
    --pipelines original ffmpeg_gentle hybrid_demucs_df

# Batch process all 429 videos with default pipeline
python -m readingroom_audio.batch run --pipeline hybrid_demucs_df --resume

# Auto-select pipeline per content type (lecture→hybrid_demucs_df, screening→deepfilter_12dB, performance→ffmpeg_gentle)
python -m readingroom_audio.batch run --auto-pipeline --resume

# Generate GitHub Pages reports
python -m readingroom_audio.benchmark export
python -m readingroom_audio.benchmark preview
```

## Architecture

```
src/readingroom_audio/
├── enhance.py          # 21 enhancement pipelines
├── score.py            # DNSMOS / NISQA / UTMOS scoring
├── benchmark.py        # Benchmark runner + statistical analysis + export
├── batch.py            # Batch processor for 429 videos
├── mux.py              # Video mux (enhanced audio → MP4)
├── compare.py          # Interactive pipeline comparison
├── listening_test.py   # Listening test generator
├── download.py         # yt-dlp batch download
├── sampling.py         # Stratified sample selection
├── segment.py          # Silero VAD segment extraction
└── utils.py            # ffmpeg wrappers, shared helpers
```

**Data flow**: event JSONs → yt-dlp download (128 kbps AAC) → Silero VAD segment extraction → pipeline enhancement → perceptual scoring → statistical analysis → GitHub Pages report.

## About The Reading Room BKK

An independent art and intellectual space in Bangkok (2010–2019) that hosted talks, reading groups, screenings, performances, and discussions — 161 events recorded on consumer cameras and uploaded to YouTube. The content spans contemporary art, political theory, postcolonial studies, Thai education reform, philosophy, and experimental music — publicly valuable discourse that deserves better audio than what cheap cameras could capture.

[Facebook](https://www.facebook.com/TheReadingRoomBKK/)
