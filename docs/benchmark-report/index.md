---
layout: default
title: Benchmark Report
---

# Audio Enhancement Benchmark Report

## Overview

Benchmark of **10 audio enhancement pipelines** across **161 stratified segments** (45-second speech-active excerpts selected via Silero VAD) from 429 YouTube recordings of The Reading Room BKK (2011-2019).

Generated: 2026-03-05

## Pipeline Quality Profile

Mean scores across all sub-metrics (higher = better for all except Noise).

| Pipeline | OVRL | SIG | BAK | P.808 | UTMOS | NISQA | Noise | Color | Discont | Loud |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 1.25 | 1.47 | 1.33 | 2.64 | 1.30 | 1.22 | 1.46 | 1.87 | 2.82 | 1.99 |
| deepfilter_12dB | 1.83 | 2.34 | 2.15 | 3.00 | 1.26 | 1.82 | 1.84 | 2.24 | 3.14 | 2.57 |
| deepfilter_full | 2.50 | 2.89 | 3.62 | 3.47 | 1.28 | 2.24 | 3.17 | 2.38 | 2.99 | 3.02 |
| demucs_vocals | 1.43 | 1.75 | 1.61 | 2.82 | 1.27 | 1.40 | 1.63 | 2.07 | 2.99 | 2.20 |
| ffmpeg_gentle | 1.31 | 1.61 | 1.43 | 2.70 | 1.27 | 1.50 | 1.49 | 2.06 | 3.09 | 2.27 |
| frcrn_16k | 2.10 | 2.65 | 2.65 | 3.06 | 1.28 | 2.09 | 2.27 | 2.46 | 3.31 | 2.73 |
| hybrid_demucs_df | 2.08 | 2.71 | 2.47 | 3.12 | 1.27 | 1.81 | 1.85 | 2.30 | 3.00 | 2.78 |
| hybrid_demucs_remix | 1.96 | 2.65 | 2.24 | 3.07 | 1.26 | 1.79 | 1.85 | 2.31 | 3.05 | 2.57 |
| mossformer2_48k | 2.22 | 2.76 | 2.93 | 3.20 | 1.27 | 2.07 | 2.49 | 2.47 | 3.30 | 2.86 |
| mossformergan_16k | 2.50 | 3.00 | 3.29 | 3.21 | 1.29 | 2.30 | 2.71 | 2.47 | 3.11 | 3.08 |

_OVRL=Overall, SIG=Signal quality, BAK=Background noise, P.808=ITU-T P.808, NISQA=MOS, Noise=Noisiness, Color=Coloration, Discont=Discontinuity, Loud=Loudness_

## Improvement Over Original

Mean improvement delta (pipeline − original). Positive = better.

| Pipeline | ΔOVRL | ΔSIG | ΔBAK | ΔP.808 | ΔUTMOS | ΔNISQA | ΔNoise | ΔColor | ΔDiscont | ΔLoud |
|---|---|---|---|---|---|---|---|---|---|---|
| deepfilter_12dB | +0.59 | +0.87 | +0.82 | +0.35 | -0.04 | +0.60 | +0.38 | +0.38 | +0.32 | +0.58 |
| deepfilter_full | +1.26 | +1.42 | +2.29 | +0.82 | -0.02 | +1.02 | +1.71 | +0.52 | +0.17 | +1.03 |
| demucs_vocals | +0.18 | +0.28 | +0.28 | +0.18 | -0.04 | +0.18 | +0.16 | +0.20 | +0.18 | +0.21 |
| ffmpeg_gentle | +0.06 | +0.14 | +0.10 | +0.05 | -0.03 | +0.28 | +0.03 | +0.20 | +0.27 | +0.28 |
| frcrn_16k | +0.86 | +1.19 | +1.32 | +0.42 | -0.03 | +0.86 | +0.80 | +0.59 | +0.49 | +0.74 |
| hybrid_demucs_df | +0.83 | +1.25 | +1.14 | +0.47 | -0.04 | +0.59 | +0.38 | +0.44 | +0.18 | +0.79 |
| hybrid_demucs_remix | +0.72 | +1.18 | +0.91 | +0.43 | -0.04 | +0.57 | +0.38 | +0.44 | +0.23 | +0.58 |
| mossformer2_48k | +0.97 | +1.29 | +1.61 | +0.56 | -0.03 | +0.85 | +1.02 | +0.60 | +0.48 | +0.86 |
| mossformergan_16k | +1.25 | +1.53 | +1.96 | +0.56 | -0.02 | +1.08 | +1.24 | +0.61 | +0.30 | +1.08 |

## Score Distribution

Distribution of DNSMOS OVRL scores across all segments for each pipeline.

![Score Distribution](images/pipeline_boxplot.png)

## Signal vs Background Tradeoff

Each point is one segment. Upper-right corner = best (high signal quality + high background suppression).

![Signal vs Background Tradeoff](images/sig_bak_tradeoff.png)

## NISQA Sub-dimensions

NISQA decomposes speech quality into noisiness, coloration, discontinuity, and loudness.

![NISQA Sub-dimensions](images/nisqa_subdimensions.png)

## Cross-Metric Correlation

DNSMOS OVRL vs UTMOS — assessing agreement between two independent quality metrics.

![Cross-Metric Correlation](images/cross_metric_scatter.png)

## Pipeline Scores by Series Group

Mean DNSMOS OVRL improvement over original, broken down by content series.

![Pipeline Scores by Series Group](images/series_heatmap.png)

## Pipeline Scores by Content Type

Mean DNSMOS OVRL improvement over original, broken down by content type (lecture, screening, performance, etc.).

![Pipeline Scores by Content Type](images/format_heatmap.png)

## Confidence Intervals

Pipeline means with 95% bootstrap confidence intervals across all metrics.

![Confidence Intervals](images/ci_forest_plot.png)

## Audio Comparison

Representative segments selected for diversity: lowest, median, and highest original DNSMOS OVRL.

### [Readrink#8 สิ้นแสงฉาน 2 OCT 2016](https://www.youtube.com/watch?v=MTw0IZwypU8) (readrink/middle) — Baseline OVRL: 1.08

| Pipeline | OVRL | SIG | BAK | Audio |
|----------|------|-----|-----|-------|
| original | 1.08 | 1.18 | 1.13 | <audio controls src="audio/E117_MTw0IZwypU8/original.mp3"></audio> |
| deepfilter_12dB | 1.54 | 2.05 | 1.69 | <audio controls src="audio/E117_MTw0IZwypU8/deepfilter_12dB.mp3"></audio> |
| deepfilter_full | 2.44 | 2.85 | 3.54 | <audio controls src="audio/E117_MTw0IZwypU8/deepfilter_full.mp3"></audio> |
| demucs_vocals | 1.10 | 1.21 | 1.13 | <audio controls src="audio/E117_MTw0IZwypU8/demucs_vocals.mp3"></audio> |
| ffmpeg_gentle | 1.09 | 1.20 | 1.15 | <audio controls src="audio/E117_MTw0IZwypU8/ffmpeg_gentle.mp3"></audio> |
| frcrn_16k | 2.33 | 2.93 | 3.06 | <audio controls src="audio/E117_MTw0IZwypU8/frcrn_16k.mp3"></audio> |
| hybrid_demucs_df | 1.73 | 2.52 | 1.79 | <audio controls src="audio/E117_MTw0IZwypU8/hybrid_demucs_df.mp3"></audio> |
| hybrid_demucs_remix | 1.90 | 2.80 | 1.96 | <audio controls src="audio/E117_MTw0IZwypU8/hybrid_demucs_remix.mp3"></audio> |
| mossformer2_48k | 1.87 | 2.61 | 2.16 | <audio controls src="audio/E117_MTw0IZwypU8/mossformer2_48k.mp3"></audio> |
| mossformergan_16k | 2.05 | 2.55 | 2.51 | <audio controls src="audio/E117_MTw0IZwypU8/mossformergan_16k.mp3"></audio> |

### [Sleepover2 What does it mean to be Avant-Garde](https://www.youtube.com/watch?v=RHiBru--TYw) (sleepover/middle) — Baseline OVRL: 1.14

| Pipeline | OVRL | SIG | BAK | Audio |
|----------|------|-----|-----|-------|
| original | 1.14 | 1.25 | 1.17 | <audio controls src="audio/E107_RHiBru--TYw/original.mp3"></audio> |
| deepfilter_12dB | 1.58 | 2.07 | 1.77 | <audio controls src="audio/E107_RHiBru--TYw/deepfilter_12dB.mp3"></audio> |
| deepfilter_full | 2.50 | 2.90 | 3.62 | <audio controls src="audio/E107_RHiBru--TYw/deepfilter_full.mp3"></audio> |
| demucs_vocals | 1.15 | 1.26 | 1.19 | <audio controls src="audio/E107_RHiBru--TYw/demucs_vocals.mp3"></audio> |
| ffmpeg_gentle | 1.52 | 2.26 | 1.64 | <audio controls src="audio/E107_RHiBru--TYw/ffmpeg_gentle.mp3"></audio> |
| frcrn_16k | 2.21 | 2.90 | 2.76 | <audio controls src="audio/E107_RHiBru--TYw/frcrn_16k.mp3"></audio> |
| hybrid_demucs_df | 2.03 | 2.99 | 2.18 | <audio controls src="audio/E107_RHiBru--TYw/hybrid_demucs_df.mp3"></audio> |
| hybrid_demucs_remix | 2.24 | 3.19 | 2.47 | <audio controls src="audio/E107_RHiBru--TYw/hybrid_demucs_remix.mp3"></audio> |
| mossformer2_48k | 2.32 | 2.84 | 3.28 | <audio controls src="audio/E107_RHiBru--TYw/mossformer2_48k.mp3"></audio> |
| mossformergan_16k | 2.71 | 3.15 | 3.64 | <audio controls src="audio/E107_RHiBru--TYw/mossformergan_16k.mp3"></audio> |

### [SLEEPOVER#6: เกษียณสโมสร](https://www.youtube.com/watch?v=tQpM9aZapc0) (sleepover/middle) — Baseline OVRL: 3.16

| Pipeline | OVRL | SIG | BAK | Audio |
|----------|------|-----|-----|-------|
| original | 3.16 | 3.56 | 3.85 | <audio controls src="audio/E123_tQpM9aZapc0/original.mp3"></audio> |
| deepfilter_12dB | 3.33 | 3.61 | 4.12 | <audio controls src="audio/E123_tQpM9aZapc0/deepfilter_12dB.mp3"></audio> |
| deepfilter_full | 3.39 | 3.63 | 4.19 | <audio controls src="audio/E123_tQpM9aZapc0/deepfilter_full.mp3"></audio> |
| demucs_vocals | 3.21 | 3.55 | 3.98 | <audio controls src="audio/E123_tQpM9aZapc0/demucs_vocals.mp3"></audio> |
| ffmpeg_gentle | 2.95 | 3.31 | 3.80 | <audio controls src="audio/E123_tQpM9aZapc0/ffmpeg_gentle.mp3"></audio> |
| frcrn_16k | 3.19 | 3.54 | 3.95 | <audio controls src="audio/E123_tQpM9aZapc0/frcrn_16k.mp3"></audio> |
| hybrid_demucs_df | 3.16 | 3.48 | 3.97 | <audio controls src="audio/E123_tQpM9aZapc0/hybrid_demucs_df.mp3"></audio> |
| hybrid_demucs_remix | 3.16 | 3.46 | 4.01 | <audio controls src="audio/E123_tQpM9aZapc0/hybrid_demucs_remix.mp3"></audio> |
| mossformer2_48k | 3.39 | 3.63 | 4.19 | <audio controls src="audio/E123_tQpM9aZapc0/mossformer2_48k.mp3"></audio> |
| mossformergan_16k | 3.31 | 3.59 | 4.11 | <audio controls src="audio/E123_tQpM9aZapc0/mossformergan_16k.mp3"></audio> |

## Statistical Analysis

### Friedman Test

| Metric | Statistic | p-value | Significant |
|--------|-----------|---------|-------------|
| DNSMOS OVRL | 1155.311 | 0.000000 | Yes |
| DNSMOS SIG | 986.124 | 0.000000 | Yes |
| DNSMOS BAK | 1203.207 | 0.000000 | Yes |
| UTMOS | 189.284 | 0.000000 | Yes |
| NISQA MOS | 1071.292 | 0.000000 | Yes |

### Key Findings

- **Strong agreement** between DNSMOS BAK vs DNSMOS OVRL (rho=0.963)
- **Strong agreement** between DNSMOS BAK vs DNSMOS SIG (rho=0.842)
- **Strong agreement** between DNSMOS OVRL vs DNSMOS SIG (rho=0.929)
- **Weak agreement** between DNSMOS BAK vs UTMOS (rho=0.098)
- **Weak agreement** between DNSMOS OVRL vs UTMOS (rho=0.098)
- **Weak agreement** between DNSMOS SIG vs UTMOS (rho=-0.030)
- **Weak agreement** between NISQA MOS vs UTMOS (rho=0.025)

## Recommendation

**`hybrid_demucs_df`** (Demucs vocal separation + DeepFilterNet 12dB + loudnorm) is recommended for batch processing. While not the highest-scoring pipeline on DNSMOS OVRL, it provides meaningful improvement over the original while preserving ambient character (laughter, room atmosphere, audience reactions) that gives these archival recordings their documentary value.

More aggressive pipelines (e.g., `deepfilter_full`) score higher on objective metrics but risk over-suppressing the ambient soundscape. The SIG vs BAK tradeoff chart above illustrates this tension.

## Pipeline by Content Type

Not all content types benefit from the same pipeline. Demucs-based pipelines use source separation that strips non-speech audio — destructive for screenings (film audio) and performances (music).

| Content Type | N | Default Pipeline | Rationale |
|---|---|---|---|
| book_club | 53 | `hybrid_demucs_df` | Reading/discussion; same approach as lecture |
| lecture | 37 | `hybrid_demucs_df` | Speech-dominant; Demucs isolates voice cleanly |
| panel | 41 | `hybrid_demucs_df` | Multi-speaker speech; same approach as lecture |
| performance | 2 | `hybrid_demucs_remix` | Music/sound art IS the content; minimal processing only |
| screening | 28 | `deepfilter_12dB` | Film audio through speakers; Demucs strips it. Mild filtering preserves film sound |

![Content Type Heatmap](images/format_heatmap.png)

## Methodology

- **Sampling**: N=161 segments (one representative clip per event, from 429 total clips), stratified by series group, content type, and era
- **Segment extraction**: 45-second speech-active windows via Silero VAD
- **Metrics**: Non-intrusive quality (DNSMOS P.835, UTMOS, NISQA) — 10 sub-metrics from 3 independent model families
- **Statistics**: Friedman test (omnibus) + Wilcoxon signed-rank (pairwise, Bonferroni corrected) + bootstrap 95% CIs
- **Source**: 429 YouTube recordings (128kbps AAC) from The Reading Room BKK (2011-2019)
- **Metric sufficiency**: Intrusive metrics (PESQ, POLQA) require clean reference audio, unavailable for archival material. Cross-metric agreement analysis confirms DNSMOS and NISQA provide consistent assessments; UTMOS shows saturation on this quality level.
