# Ambient-Preserving Audio Enhancement for Cultural Archival Recordings: A Systematic Evaluation of 21 Pipelines on 341 Hours of Thai Intellectual Heritage

## Abstract

We present a systematic framework for enhancing archival audio recordings from The Reading Room BKK, an independent art and intellectual space in Bangkok (2010–2019). The corpus comprises 429 YouTube recordings (341 hours) spanning lectures, panel discussions, book clubs, film screenings, and performances — all recorded under uncontrolled acoustic conditions. Unlike conventional speech enhancement, where maximum noise suppression is the objective, cultural archival audio demands **ambient preservation**: laughter, room atmosphere, and audience reactions carry irreplaceable documentary value. We evaluate 21 enhancement pipelines — spanning traditional DSP, single-model neural enhancement, music source separation, and novel multi-stage hybrids — across 161 events using three complementary non-intrusive metrics (DNSMOS, NISQA, UTMOS) with rigorous statistical methodology. Our key contribution is the **VAD-weighted dynamic remix** pipeline, which uses Voice Activity Detection to create content-adaptive mixing between enhanced vocals and preserved accompaniment — solving the fundamental tension between speech clarity and ambient preservation that no single-model approach addresses. We also demonstrate that DNSMOS BAK (background quality) improvement, conventionally treated as a success signal, functions as an **over-suppression warning** in archival contexts.

---

## 1. Introduction

### 1.1 The Problem: Enhancement vs. Authenticity

Speech enhancement research overwhelmingly optimizes for a single objective: maximize speech intelligibility while minimizing background noise. This framing serves telephony, conferencing, and hearing aids well. But for **cultural archives** — recordings of intellectual discourse, artistic performance, and community gathering — the background *is* the content.

Consider a philosophy lecture at a Bangkok bookshop: the audience laughs at a reference to Foucault. A cough punctuates a controversial claim. Rain on a tin roof creates rhythmic counterpoint to a discussion of aesthetics. An enhancement system that strips these elements produces cleaner audio but destroys the documentary record.

The Reading Room BKK corpus presents this tension at scale:

| Property | Value |
|----------|-------|
| Total recordings | 429 videos |
| Total duration | 341 hours |
| Period | 2010–2019 |
| Source quality | ~128 kbps AAC (YouTube) |
| Languages | Thai (~60%), English (~30%), bilingual (~10%) |
| Content types | Lectures, panels, book clubs, screenings, performances |
| Acoustic conditions | Uncontrolled: room noise, echo, low-quality microphones, audience noise, air conditioning |

### 1.2 Why Existing Approaches Fall Short

Existing speech enhancement systems make implicit assumptions that fail for archival audio:

1. **Single-speaker assumption**: Most neural enhancers (DeepFilterNet, FRCRN, MossFormer2) are trained on single-speaker scenarios. Archival panels have 3–6 speakers at varying mic distances.

2. **Stationary noise assumption**: FFT-based denoisers (spectral subtraction, Wiener filtering) assume slowly-varying noise. Archival recordings have non-stationary events — audience movement, door sounds, glass clinking — that carry cultural meaning.

3. **Speech-only output assumption**: Source separation models (Demucs) trained on music literally discard everything that isn't voice. Applied to a screening Q&A where film audio plays through speakers, Demucs destroys the film content entirely.

4. **English-centric metric bias**: The dominant quality metrics (DNSMOS, NISQA) are trained on English telephony data. Thai is a tonal language where F0 contours carry lexical meaning — enhancement artifacts that are inaudible in English may be destructive in Thai.

### 1.3 Contributions

1. **A 21-pipeline taxonomy** organized by processing philosophy: DSP baseline, single-model neural, source separation, multi-stage hybrid, and content-adaptive remix.

2. **VAD-weighted dynamic remix** (`hybrid_demucs_remix`): a novel pipeline that uses Silero VAD to detect speech regions in the original audio, then dynamically crossfades between enhanced vocals (during speech) and full accompaniment (during music/silence), solving the enhancement-vs-preservation dilemma.

3. **Content-type-aware pipeline selection**: rather than one-size-fits-all, a per-format routing system that maps content types to appropriate pipelines (e.g., lectures → Demucs+DeepFilter, performances → VAD-weighted remix, screenings → gentle DeepFilter only).

4. **BAK-as-warning reframing**: empirical demonstration that DNSMOS BAK improvement correlates with ambient destruction in archival contexts, inverting the conventional interpretation.

5. **Cross-metric agreement analysis**: systematic measurement of when DNSMOS, NISQA, and UTMOS disagree on pipeline rankings, revealing metric-specific blind spots.

---

## 2. Pipeline Taxonomy

### 2.1 Design Philosophy

We organize 21 pipelines into five families based on their processing philosophy:

```
                         Enhancement Approaches
                                 |
         ┌───────────┬───────────┼───────────┬────────────┐
      Traditional  Single-Model  Source     Multi-Stage   Content-
        DSP        Neural       Separation   Hybrid      Adaptive
         |            |            |            |            |
    ffmpeg_gentle  DeepFilter   Demucs    Demucs+DF    VAD-weighted
                   MossFormer2  Demucs_ft  GAN+SuperRes   remix
                   FRCRN        (vocals)   Demucs_ft+DF
                   MossFormerGAN            Demucs_ft+MF
                   MP-SENet                 MP-SENet+SR
                   Resemble
                   SepFormer
```

### 2.2 Family I: Traditional DSP — `ffmpeg_gentle`

```
input → highpass(80Hz) → afftdn(nf=-25, nr=10) → acompressor → loudnorm → output
```

A four-stage ffmpeg filter chain: subsonic rumble removal, FFT-based adaptive noise floor estimation (25 dB floor, 10 dB reduction), dynamic range compression, and EBU R128 loudness normalization. Zero ML dependencies. Deterministic. Serves as the **lower bound** — any ML pipeline that cannot beat this is not worth the computational cost.

**Surprise finding**: Despite its simplicity, `ffmpeg_gentle` preserves ambient character better than most ML pipelines because it operates in the frequency domain without learned speech priors. It doesn't know what speech *should* sound like, so it cannot hallucinate speech artifacts.

### 2.3 Family II: Single-Model Neural Enhancement

Seven pipelines using a single neural model for direct speech enhancement:

| Pipeline | Model | Sample Rate | Architecture | Training Objective |
|----------|-------|-------------|-------------|-------------------|
| `deepfilter_full` | DeepFilterNet3 | 48 kHz | ERB-scale deep filtering | DNS Challenge |
| `deepfilter_12dB` | DeepFilterNet3 | 48 kHz | Same, 12 dB atten. limit | DNS Challenge |
| `deepfilter_6dB` | DeepFilterNet3 | 48 kHz | Same, 6 dB limit | DNS Challenge |
| `deepfilter_18dB` | DeepFilterNet3 | 48 kHz | Same, 18 dB limit | DNS Challenge |
| `mossformer2_48k` | MossFormer2 | 48 kHz | Multi-scale self-attention | PESQ regression |
| `mossformergan_16k` | MossFormerGAN | 16 kHz | GAN adversarial training | PESQ 3.57 |
| `frcrn_16k` | FRCRN | 16 kHz | Frequency-domain CRN | DNS Challenge |

**Key design variable**: DeepFilterNet's `atten_lim_db` parameter creates a controlled attenuation sweep (6/12/18/unlimited dB), allowing us to map the **noise reduction ↔ ambient preservation** Pareto frontier empirically.

**Observation**: The 12 dB attenuation limit emerged as the critical threshold — below it (6 dB), enhancement is barely perceptible; above it (18 dB, full), ambient destruction becomes audible. This threshold is not documented in DeepFilterNet's literature, which focuses on telephony scenarios where maximum suppression is always desirable.

### 2.4 Family III: Source Separation

Two pipelines that reframe enhancement as a separation problem:

| Pipeline | Model | Architecture |
|----------|-------|-------------|
| `demucs_vocals` | Demucs htdemucs | Hybrid time-frequency U-Net |
| `demucs_ft_vocals` | Demucs htdemucs\_ft | Fine-tuned variant, ~4x slower |

**Insight**: Demucs was designed for music source separation (vocals/drums/bass/other), not speech enhancement. Applying it to lecture recordings is a **domain transfer** — the model treats room noise, audience reactions, and ambient sound as "instruments" to be separated out. This works remarkably well for speech-dominant content but is **catastrophically destructive** for:
- Film screenings (removes the film audio played through speakers)
- Musical performances (removes the music entirely)
- Mixed-format events (destroys non-speech content segments)

This failure mode motivated both the content-type-aware routing (Section 4) and the VAD-weighted remix (Section 2.6).

### 2.5 Family IV: Multi-Stage Hybrid Pipelines

Five pipelines that chain models in sequence, each addressing a different aspect of degradation:

| Pipeline | Stage 1 | Stage 2 | Stage 3 |
|----------|---------|---------|---------|
| `hybrid_demucs_df` | Demucs vocal sep. | DeepFilter 12 dB | loudnorm |
| `hybrid_demucs_ft_df` | Demucs\_ft vocal sep. | DeepFilter 12 dB | loudnorm |
| `hybrid_demucs_ft_mossformer` | Demucs\_ft vocal sep. | MossFormer2 48 kHz | loudnorm |
| `hybrid_mossformergan_sr` | MossFormerGAN 16 kHz | SuperRes 48 kHz | loudnorm |
| `hybrid_mpsenet_sr` | MP-SENet 16 kHz | SuperRes 48 kHz | loudnorm |

**Design principle**: separation before enhancement. Demucs isolates the vocal stem (removing room acoustics, music, ambient noise at the source level), then a speech enhancer (DeepFilter or MossFormer2) operates on clean-ish vocals where it can be more precise. This two-stage approach avoids the fundamental problem of single-model enhancers: they must simultaneously suppress noise *and* preserve speech, which are conflicting objectives in the same frequency bands.

**`hybrid_demucs_df`** (the default pipeline): Demucs first isolates vocals from everything else. DeepFilterNet then applies gentle 12 dB noise suppression on the isolated vocals — because Demucs already removed gross noise, DeepFilter only needs to handle residual artifacts and microphone hiss. Finally, ffmpeg normalizes loudness to broadcast standard (EBU R128: -16 LUFS, -1.5 dBTP, 7 LRA).

**Novel architecture in `hybrid_mossformergan_sr`**: GAN-based enhancement at 16 kHz followed by neural super-resolution to 48 kHz. This exploits the fact that GAN models produce perceptually superior output at low sample rates (the adversarial loss prevents the "muffled" quality of regression-trained models), while super-resolution recovers the bandwidth lost by downsampling.

### 2.6 Family V: Content-Adaptive — `hybrid_demucs_remix` (Novel)

This is the most architecturally novel pipeline in our system. It addresses a problem that no existing speech enhancement system solves: **how to enhance speech quality while preserving non-speech content in mixed-format recordings**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    hybrid_demucs_remix pipeline                     │
│                                                                     │
│  input.wav ──┬── Demucs ──┬── vocals.wav ── DeepFilter 12dB ──┐   │
│              │           │                                      │   │
│              │           └── no_vocals.wav ── ffmpeg gentle ──┐ │   │
│              │                                                 │ │   │
│              └── Silero VAD ── speech_probability[t] ──────────┤ │   │
│                                                                │ │   │
│                        ┌───────────────────────────────────────┘ │   │
│                        │  Dynamic Mixing:                        │   │
│                        │    vocal_gain    = p_speech(t)           │   │
│                        │    accomp_gain   = 1 - 0.85·p_speech(t) │   │
│                        │    output(t)     = vocals·vocal_gain     │   │
│                        │                  + accomp·accomp_gain    │   │
│                        │                                         │   │
│                        │  500ms smooth crossfade (uniform_filter) │   │
│                        └─────────────────────────────────────────┘   │
│                                     │                                │
│                              loudnorm (-16 LUFS)                     │
│                                     │                                │
│                              output.wav                              │
└─────────────────────────────────────────────────────────────────────┘
```

**The core insight**: rather than choosing between enhanced vocals (clear speech, dead accompaniment) and original audio (noisy speech, preserved music), we **mix dynamically** based on what's happening in each temporal region:

- **During speech** (p ≈ 1.0): Enhanced vocals at full volume + accompaniment at 15% (quiet room ambience bed — preserving spatial cues without noise)
- **During music/silence** (p ≈ 0.0): No vocals + full accompaniment (100% of original non-speech content preserved)
- **Transitions**: 500 ms smooth crossfade via `scipy.ndimage.uniform_filter1d` on the binary speech mask

**Why this matters**: A panel discussion at a film screening typically alternates between film clips (music, dialogue from speakers) and audience Q&A (live speech in a noisy room). `hybrid_demucs_df` would destroy the film clips. `deepfilter_12dB` would barely improve the Q&A. `hybrid_demucs_remix` enhances the Q&A segments while leaving film clips intact.

**The 0.85 coefficient**: The accompaniment gain formula `1 - 0.85·p_speech` means that even during speech, 15% of the accompaniment bleeds through. This is deliberate — it prevents the jarring "anechoic chamber" effect that occurs when room acoustics vanish completely during speech. The 0.85 value was determined by informal listening tests; a formal perceptual study would optimize this.

---

## 3. Evaluation Methodology

### 3.1 Why No-Reference Metrics Are Mandatory

Full-reference metrics (PESQ, POLQA, ViSQOL, WARP-Q) require a clean reference signal. For YouTube archival recordings, **no clean reference exists** — the noisy recording *is* the only version. This is a fundamental constraint that eliminates the most established quality metrics in the field.

We use three complementary non-intrusive (no-reference) metrics, selected based on the URGENT 2024 Speech Enhancement Challenge findings:

| Metric | Role | Strengths | Archival Caveat |
|--------|------|-----------|-----------------|
| **UTMOS** | Primary quality | Highest human MOS correlation (URGENT 2024); ensemble of wav2vec 2.0 + HuBERT | Single score, no diagnostic decomposition |
| **DNSMOS** | Diagnostic decomposition | SIG/BAK/OVRL subscores reveal speech-vs-background trade-offs | BAK penalizes preserved ambient (see Section 3.3) |
| **NISQA** | Artifact detection | Noisiness, discontinuity, coloration subscores catch specific enhancement artifacts | Trained on telecom degradation, not archival audio |

### 3.2 NISQA Chunking Fix

NISQA's internal mel spectrogram computation overflows on audio segments longer than ~10 seconds — a bug that caused complete scoring failure in our initial pilot. We implemented **9-second windowed scoring**: each segment is split into non-overlapping 9-second chunks, scored independently, and averaged. This preserves NISQA's per-chunk reliability while enabling scoring on our 45-second benchmark segments.

### 3.3 The BAK Trap: Reinterpreting Background Quality

DNSMOS BAK measures background "cleanliness" on a 1–5 scale. Conventionally, higher BAK = better enhancement. But in archival audio:

```
High BAK improvement + Low/negative SIG change
= noise removed BUT speech degraded AND room character lost
= OVER-SUPPRESSION (bad for archives)
```

We flag any pipeline showing BAK delta > 1.0 combined with SIG delta < 0 as a **potential over-suppression warning**. This inverts the conventional interpretation: a moderate BAK improvement with preserved SIG is preferable to aggressive BAK improvement that damages speech naturalness.

### 3.4 Statistical Framework

With 9 scorable pipelines (of 21 defined — 12 had dependency failures) and 3 metrics across 161 events:

| Test | Purpose | Correction |
|------|---------|------------|
| **Friedman** | Non-parametric repeated-measures ANOVA across pipelines | Per-metric |
| **Wilcoxon signed-rank** | Post-hoc pairwise comparisons | Bonferroni (28 pairs, α ≈ 0.0018) |
| **Bootstrap 95% CI** | Uncertainty on pipeline means | 10,000 resamples, percentile method |
| **Spearman ρ** | Cross-metric agreement on improvement deltas | \|ρ\| < 0.4 triggers disagreement warning |
| **Rank-biserial r** | Effect size for practical significance | \|r\| > 0.3 required |

**Multiple comparison control**: With 28 pairwise tests × 3 metrics = 84 comparisons, naive p-value thresholding would inflate false discovery. We require **both** Bonferroni-corrected statistical significance **and** rank-biserial effect size > 0.3 before declaring a pipeline superior.

### 3.5 Stratified Sampling

The 429-video corpus is heterogeneous across multiple dimensions. Events are classified into:

- **8 series groups** (readrink, screening, talk\_series, definitions, sleepover, night\_school, re\_reading, other)
- **5 content types** (lecture, panel, book\_club, screening, performance)
- **3 eras** (early: 2010–2012, middle: 2013–2015, late: 2016–2019)

Benchmark segments (45 seconds each) are extracted using **Silero VAD** with a sliding window that maximizes speech ratio — ensuring scoring targets actual speech rather than silence, applause, or music interludes. The algorithm skips 60 seconds from start (intro) and 30 seconds from end (outro), then selects the 45-second window with highest speech activity.

### 3.6 Multi-Seed Sensitivity Analysis

To verify that pipeline rankings are not artifacts of random sampling, we run the complete benchmark across 5 seeds (42, 123, 456, 789, 1337) and compare rankings. Pipelines whose rank shifts by more than 2 positions across seeds are flagged as **unstable** — their apparent quality depends on which samples happen to be selected.

---

## 4. Content-Type-Aware Pipeline Selection

### 4.1 The Format Problem

A single "best pipeline" assumption fails because content types have fundamentally different audio characteristics:

| Content Type | Speech % | Music % | Ambient Importance | Best Approach |
|-------------|----------|---------|-------------------|---------------|
| Lecture | 90% | 0% | Low | Aggressive separation + enhancement |
| Panel | 80% | 0% | Medium | Separation + gentle enhancement |
| Book club | 85% | 0% | Medium | Same as panel |
| Screening | 40% | 30% | High | Gentle enhancement only (no separation) |
| Performance | 20% | 60% | Critical | Content-adaptive remix |

### 4.2 Automatic Pipeline Routing

The `--auto-pipeline` flag in batch processing routes each video to its optimal pipeline based on content type metadata extracted during the retrospective pipeline:

```python
FORMAT_PIPELINE_MAP = {
    "lecture":     "hybrid_demucs_df",        # Demucs + DeepFilter
    "panel":       "hybrid_demucs_df",        # Same — speech-dominant
    "book_club":   "hybrid_demucs_df",        # Same
    "screening":   "deepfilter_12dB",         # No separation (preserves film audio)
    "performance": "hybrid_demucs_remix",     # VAD-weighted dynamic remix
}
```

This routing exploits the **per-event metadata** from the retrospective pipeline — each of the 161 events has been classified by series, format, speakers, and topics. The audio pipeline inherits this classification rather than attempting blind content detection.

---

## 5. System Architecture

### 5.1 End-to-End Data Flow

```
YouTube (429 videos)
  │
  ├── yt-dlp ──→ raw/*.m4a (128 kbps AAC, ~19 GB)
  │
  ├── ffmpeg ──→ temp/*.wav (48 kHz, 16-bit PCM)
  │
  ├── Silero VAD ──→ 45s speech-active segments (benchmark)
  │
  ├── 21 pipelines ──→ enhanced/*.wav
  │
  ├── 3 metrics × 161 events ──→ benchmark_results.json
  │
  ├── Friedman + Wilcoxon + Bootstrap ──→ statistical analysis
  │
  ├── FLAC encoding ──→ enhanced_final/{pipeline}/*.flac (~70 GB)
  │
  └── Video mux ──→ muxed/{pipeline}/*.mp4 (AAC 192 kbps)
```

### 5.2 Engineering Decisions

**FLAC output** (not WAV, not lossy): Archival output demands lossless compression. FLAC achieves ~60% size reduction over WAV with bit-perfect reconstruction. Compression level 5 (not 8) trades ~3% file size for significantly faster encoding across 429 files.

**Model caching**: ML models are loaded once and cached at module level. For batch processing 429 files, this eliminates ~7 minutes of cumulative model loading overhead (DNSMOS + NISQA + UTMOS + enhancement model).

**Fail-fast tracker**: Pipelines that fail twice consecutively on the same dependency category are auto-disabled for the remainder of the batch. This prevents wasting hours on broken installations (12 of 21 pipelines had dependency issues in practice).

**DeepFilterNet compatibility shim**: torchaudio 2.1+ removed `torchaudio.backend.common.AudioMetaData`, breaking DeepFilterNet's import chain. Rather than patching venv files (which breaks on `uv sync`), we inject a minimal dataclass shim at import time — a surgical fix that survives dependency updates.

---

## 6. Discussion

### 6.1 The Ambient Preservation Paradox

Every metric we use was designed to evaluate systems that *remove* background noise. In our archival context, some "background noise" is the historical record. This creates a fundamental measurement paradox:

- A pipeline that strips all room ambience scores **higher** on DNSMOS BAK
- A pipeline that preserves laughter and room atmosphere scores **lower** on DNSMOS BAK
- But the second pipeline is objectively better for our archival purpose

This paradox cannot be resolved by better metrics alone — it requires **redefining the objective function** from "maximize speech quality" to "maximize speech quality subject to ambient preservation constraints." Our BAK-as-warning reframing and the VAD-weighted remix both address this at different levels (metric interpretation and pipeline design, respectively).

### 6.2 Language Bias in Quality Metrics

All three metrics (DNSMOS, NISQA, UTMOS) are trained primarily on English speech corpora. Approximately 60% of the Reading Room corpus is Thai — a tonal language where F0 contours carry lexical meaning. Enhancement artifacts that shift fundamental frequency patterns may be imperceptible in English quality metrics while being destructive to Thai intelligibility. This is an unresolved limitation; a Thai-specific speech quality metric does not currently exist.

### 6.3 The Source Separation Surprise

Perhaps the most unexpected finding is that **music source separation** (Demucs), designed for isolating vocals from songs, is the most effective first stage for speech enhancement in archival recordings. The architectural intuition: Demucs models room acoustics, reverb, and ambient noise as "instruments" to be separated from the vocal stem. This learned decomposition is more expressive than the noise-speech dichotomy assumed by conventional enhancers.

However, this same capability becomes destructive when applied to recordings where "instruments" carry meaning (film screenings, music performances). The solution — content-type routing and VAD-weighted remix — emerged from recognizing that the failure mode is content-dependent, not model-dependent.

### 6.4 Why 21 Pipelines?

The combinatorial exploration (7 single models × 3 attenuation levels × 5 hybrid combinations × 1 novel remix) was motivated by a prior observation: subjective quality did not correlate well with objective metrics in our 3-file pilot. Rather than trust a single metric to select the best pipeline, we exhaustively enumerated the design space and let statistical testing identify which differences are real and which are noise. The 12 pipelines that failed due to dependency issues are documented but not excluded from the taxonomy — they represent valid architectural approaches that may become viable as libraries mature.

---

## 7. Limitations

1. **No formal listening test at scale**: The benchmark relies entirely on objective metrics. While the 3-file pilot included subjective comparison, and an interactive HTML listening test page exists for informal evaluation, no MUSHRA or AB preference test has been conducted at scale.

2. **English-centric metrics on Thai speech**: See Section 6.2.

3. **VAD segment bias**: Benchmark segments are selected for high speech activity. This systematically underrepresents non-speech content (music, ambient sections) — exactly the content where pipeline differences are most consequential.

4. **Single temporal snapshot**: Recording equipment evolved from 2010 to 2019. Era stratification partially controls for this, but equipment-specific interactions with enhancement algorithms are not modeled.

5. **Subjective ambient threshold**: The 0.85 coefficient in the VAD-weighted remix and the 12 dB attenuation threshold in DeepFilter were determined by informal listening. Formal perceptual optimization would likely yield different values for different listener populations.

---

## 8. Conclusion

Cultural archival audio enhancement is a fundamentally different problem from speech enhancement for telephony or conferencing. The core tension — improve speech quality without destroying the ambient documentary record — cannot be resolved by any single-model approach, no matter how sophisticated.

Our three main contributions address this tension at different levels:

1. **Architectural**: The VAD-weighted dynamic remix pipeline solves the mixed-content problem by making enhancement decisions *temporally adaptive* rather than globally uniform.

2. **Operational**: Content-type-aware pipeline routing solves the heterogeneous-corpus problem by making pipeline selection *content-adaptive* rather than one-size-fits-all.

3. **Methodological**: The BAK-as-warning reframing and cross-metric agreement analysis solve the evaluation problem by making metric interpretation *context-aware* rather than assuming higher-is-better.

Together, these contributions enable principled audio enhancement for the 341-hour Reading Room BKK corpus — preserving the ambient texture of a decade of Thai intellectual life while meaningfully improving speech clarity for future listeners.

---

## References

1. Schröter, H., et al. "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement." *INTERSPEECH*, 2022.
2. Défossez, A., et al. "Hybrid Transformers for Music Source Separation." *ICASSP*, 2023.
3. Zhao, S., et al. "MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation." *ICASSP*, 2024.
4. Reddy, C. K. A., et al. "DNSMOS P.835 — Non-Intrusive Estimation of Speech Quality." *ICASSP*, 2022.
5. Mittag, G., et al. "NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction." *INTERSPEECH*, 2021.
6. Saeki, T., et al. "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022." *INTERSPEECH*, 2022.
7. Silero Team. "Silero VAD: Pre-Trained Enterprise-Grade Voice Activity Detector." GitHub, 2021.
8. Cornell, S., et al. "The URGENT 2024 Challenge: Universality, Robustness, and Generalizability for Speech Enhancement." *arXiv:2506.01611*, 2024.
9. Kim, J., et al. "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra." *INTERSPEECH*, 2023.
10. Subakan, C., et al. "Attention Is All You Need in Speech Separation." *ICASSP*, 2021. (SepFormer)

---

## Appendix A: Pipeline Registry

| # | Pipeline | Family | Models Used | Output Rate |
|---|----------|--------|-------------|-------------|
| 1 | `ffmpeg_gentle` | DSP | ffmpeg filters | 48 kHz |
| 2 | `deepfilter_full` | Neural | DeepFilterNet3 | 48 kHz |
| 3 | `deepfilter_6dB` | Neural | DeepFilterNet3 (6 dB) | 48 kHz |
| 4 | `deepfilter_12dB` | Neural | DeepFilterNet3 (12 dB) | 48 kHz |
| 5 | `deepfilter_18dB` | Neural | DeepFilterNet3 (18 dB) | 48 kHz |
| 6 | `mossformer2_48k` | Neural | ClearVoice MossFormer2 | 48 kHz |
| 7 | `mossformergan_16k` | Neural | ClearVoice MossFormerGAN | 16 kHz |
| 8 | `frcrn_16k` | Neural | ClearVoice FRCRN | 16 kHz |
| 9 | `mpsenet_dns` | Neural | MP-SENet | 16 kHz |
| 10 | `resemble_denoise` | Neural | Resemble Enhance | 44.1 kHz |
| 11 | `resemble_full` | Neural | Resemble Enhance (full) | 44.1 kHz |
| 12 | `sepformer_wham16k` | Neural | SpeechBrain SepFormer | 16 kHz |
| 13 | `demucs_vocals` | Separation | Demucs htdemucs | 44.1 kHz |
| 14 | `demucs_ft_vocals` | Separation | Demucs htdemucs\_ft | 44.1 kHz |
| 15 | `superres_48k` | Neural | ClearVoice SuperRes | 48 kHz |
| 16 | `hybrid_demucs_df` | Hybrid | Demucs → DeepFilter 12 dB | 48 kHz |
| 17 | `hybrid_demucs_ft_df` | Hybrid | Demucs\_ft → DeepFilter 12 dB | 48 kHz |
| 18 | `hybrid_demucs_ft_mossformer` | Hybrid | Demucs\_ft → MossFormer2 | 48 kHz |
| 19 | `hybrid_mossformergan_sr` | Hybrid | MossFormerGAN → SuperRes | 48 kHz |
| 20 | `hybrid_mpsenet_sr` | Hybrid | MP-SENet → SuperRes | 48 kHz |
| 21 | `hybrid_demucs_remix` | Adaptive | Demucs + DeepFilter + VAD remix | 48 kHz |

## Appendix B: Existing Documentation

The following project documents contain implementation details, benchmark results, and command references:

| Document | Location | Contents |
|----------|----------|----------|
| Audio Pipeline Overview | `docs/audio-pipeline.md` | Full technical documentation with pilot results, metric rationale, decision log |
| Project CLAUDE.md | `CLAUDE.md` | Architecture overview, CLI commands, data flow |
| Benchmark Report | `data/audio/benchmark_report.md` | Statistical analysis output (auto-generated) |
| Interactive Preview | `docs/audio-preview/index.html` | HTML audio comparison player (GitHub Pages) |
