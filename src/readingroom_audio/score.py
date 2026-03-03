"""DNSMOS, NISQA, and UTMOS quality scoring for audio enhancement evaluation.

Provides non-intrusive (no reference needed) audio quality metrics:
- DNSMOS: Deep Noise Suppression Mean Opinion Score (P.808)
- NISQA: Non-Intrusive Speech Quality Assessment (chunked to 9s windows)
- UTMOS: UTokyo-SaruLab MOS prediction (graceful fallback)
"""

import json
from pathlib import Path

import torch
import torchaudio

# NISQA max window length in seconds (avoids mel spectrogram overflow)
_NISQA_CHUNK_SEC = 9


def score_audio(wav_path: str, max_seconds: int = 60,
                skip_seconds: int = 30) -> dict:
    """Score audio quality using DNSMOS and NISQA (non-intrusive, no reference).

    Args:
        wav_path: Path to WAV file to score.
        max_seconds: Maximum duration to score (for speed). Default 60s.
        skip_seconds: Skip this many seconds from start (avoid intros). Default 30s.

    Returns:
        Dict with score keys like dnsmos_p808, dnsmos_sig, dnsmos_bak, dnsmos_ovrl,
        and nisqa_mos, nisqa_noisiness, etc. Error keys if scoring fails.
    """
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)  # mono

    # Resample to 16kHz for quality metrics
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # Extract representative segment
    max_samples = 16000 * max_seconds
    if wav.shape[0] > max_samples:
        start = 16000 * skip_seconds
        wav = wav[start : start + max_samples]

    scores = {}

    # DNSMOS scoring
    scores.update(_score_dnsmos(wav))

    # NISQA scoring (chunked)
    scores.update(_score_nisqa(wav))

    return scores


def score_segment(wav_path: str, metrics: list[str] | None = None) -> dict:
    """Score a pre-extracted segment (no truncation/skipping).

    Designed for benchmark segments that are already the right length.

    Args:
        wav_path: Path to WAV segment file.
        metrics: List of metric sets to compute. None = all available.
                 Options: "dnsmos", "nisqa", "utmos"

    Returns:
        Dict with all available metric scores.
    """
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)  # mono

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    if metrics is None:
        metrics = ["dnsmos", "nisqa", "utmos"]

    scores = {}
    if "dnsmos" in metrics:
        scores.update(_score_dnsmos(wav))
    if "nisqa" in metrics:
        scores.update(_score_nisqa(wav))
    if "utmos" in metrics:
        scores.update(_score_utmos(wav))

    return scores


def _score_dnsmos(wav: torch.Tensor) -> dict:
    """Score using Deep Noise Suppression MOS (P.808)."""
    try:
        from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
        dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
        dns_scores = dnsmos(wav)
        return {
            "dnsmos_p808": float(dns_scores[0]),
            "dnsmos_sig": float(dns_scores[1]),
            "dnsmos_bak": float(dns_scores[2]),
            "dnsmos_ovrl": float(dns_scores[3]),
        }
    except Exception as e:
        return {"dnsmos_error": str(e)}


def _score_nisqa(wav: torch.Tensor) -> dict:
    """Score using NISQA, chunked to 9s windows to avoid mel overflow.

    NISQA's internal mel spectrogram computation fails on audio longer
    than ~10 seconds. We chunk into _NISQA_CHUNK_SEC windows and average.
    """
    try:
        from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
        nisqa = NonIntrusiveSpeechQualityAssessment(fs=16000)

        chunk_samples = 16000 * _NISQA_CHUNK_SEC
        total_samples = wav.shape[0]

        if total_samples <= chunk_samples:
            nisqa_scores = nisqa(wav)
            return _nisqa_dict(nisqa_scores)

        # Chunk into windows and average scores
        all_scores = []
        for start in range(0, total_samples, chunk_samples):
            chunk = wav[start : start + chunk_samples]
            # Skip very short trailing chunks (< 2 seconds)
            if chunk.shape[0] < 16000 * 2:
                continue
            try:
                chunk_scores = nisqa(chunk)
                all_scores.append(chunk_scores)
            except Exception:
                continue

        if not all_scores:
            return {"nisqa_error": "all chunks failed"}

        # Average across chunks
        stacked = torch.stack(all_scores)
        avg = stacked.mean(dim=0)
        return _nisqa_dict(avg)

    except Exception as e:
        return {"nisqa_error": str(e)}


def _nisqa_dict(scores: torch.Tensor) -> dict:
    """Convert NISQA score tensor to named dict."""
    return {
        "nisqa_mos": float(scores[0]),
        "nisqa_noisiness": float(scores[1]),
        "nisqa_discontinuity": float(scores[2]),
        "nisqa_coloration": float(scores[3]),
        "nisqa_loudness": float(scores[4]),
    }


def _score_utmos(wav: torch.Tensor) -> dict:
    """Score using UTMOS (UTokyo-SaruLab MOS predictor).

    Graceful fallback if utmos is not installed.
    """
    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
            trust_repo=True,
        )
        # UTMOS expects (batch, samples) at 16kHz
        score = predictor(wav.unsqueeze(0), sr=16000)
        return {"utmos_score": float(score.item())}
    except Exception as e:
        return {"utmos_error": str(e)}


def load_report(report_path: str | Path) -> dict:
    """Load a quality report JSON file."""
    with open(report_path) as f:
        return json.load(f)


def save_report(results: dict, report_path: str | Path):
    """Save quality scores to JSON report file."""
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
