"""
Multi-model audio enhancement comparison with automated quality scoring.

Runs multiple enhancement approaches on pilot files and scores them using
DNSMOS and NISQA metrics.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
ORIG_DIR = BASE / "pilot" / "original"
OUT_DIR = BASE / "pilot" / "models"
REPORT_PATH = BASE / "pilot" / "quality_report.json"

PILOT_FILES = [
    "01_earliest_talk",
    "02_screening_talk",
    "03_thai_talk",
]


# ── Quality scoring ────────────────────────────────────────────────────
def score_audio(wav_path: str) -> dict:
    """Score audio quality using DNSMOS and NISQA (non-intrusive, no reference)."""
    from torchmetrics.audio import (
        DeepNoiseSuppressionMeanOpinionScore,
        NonIntrusiveSpeechQualityAssessment,
    )

    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)  # mono

    # Resample to 16kHz for quality metrics
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # Truncate to 60 seconds max for scoring speed
    max_samples = 16000 * 60
    if wav.shape[0] > max_samples:
        # Score from 30s-90s (skip intro, get representative section)
        start = 16000 * 30
        wav = wav[start : start + max_samples]

    scores = {}

    # DNSMOS
    try:
        dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False)
        dns_scores = dnsmos(wav)
        scores["dnsmos_p808"] = float(dns_scores[0])
        scores["dnsmos_sig"] = float(dns_scores[1])
        scores["dnsmos_bak"] = float(dns_scores[2])
        scores["dnsmos_ovrl"] = float(dns_scores[3])
    except Exception as e:
        scores["dnsmos_error"] = str(e)

    # NISQA
    try:
        nisqa = NonIntrusiveSpeechQualityAssessment(fs=16000)
        nisqa_scores = nisqa(wav)
        scores["nisqa_mos"] = float(nisqa_scores[0])
        scores["nisqa_noisiness"] = float(nisqa_scores[1])
        scores["nisqa_discontinuity"] = float(nisqa_scores[2])
        scores["nisqa_coloration"] = float(nisqa_scores[3])
        scores["nisqa_loudness"] = float(nisqa_scores[4])
    except Exception as e:
        scores["nisqa_error"] = str(e)

    return scores


# ── Enhancement methods ────────────────────────────────────────────────
def enhance_clearvoice_mossformer2(input_wav: str, output_wav: str):
    """ClearerVoice-Studio MossFormer2 48kHz speech enhancement."""
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    result = cv(input_path=input_wav, online_write=False)
    cv.write(result, output_path=output_wav)


def enhance_clearvoice_frcrn(input_wav: str, output_wav: str):
    """ClearerVoice-Studio FRCRN 16kHz speech enhancement."""
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    result = cv(input_path=input_wav, online_write=False)
    cv.write(result, output_path=output_wav)


def enhance_deepfilternet(input_wav: str, output_wav: str, atten_lim: int = None):
    """DeepFilterNet3 noise suppression."""
    from df.enhance import enhance, init_df, load_audio, save_audio

    model, df_state, _ = init_df()
    audio, info = load_audio(input_wav, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim)
    save_audio(output_wav, enhanced, df_state.sr())


def enhance_demucs_vocals(input_wav: str, output_wav: str):
    """Demucs htdemucs vocal separation — extract speech stem."""
    out_dir = str(Path(output_wav).parent / "_demucs_tmp")
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", out_dir,
        input_wav,
    ]
    subprocess.run(cmd, capture_output=True, timeout=1200)

    # Copy vocals stem to output path
    stem_name = Path(input_wav).stem
    vocals_path = Path(out_dir) / "htdemucs" / stem_name / "vocals.wav"
    if vocals_path.exists():
        data, sr = sf.read(str(vocals_path))
        sf.write(output_wav, data, sr)


def enhance_ffmpeg_gentle(input_wav: str, output_wav: str):
    """ffmpeg-only: high-pass + mild FFT denoise + compression + loudnorm."""
    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-af", (
            "highpass=f=80,"
            "afftdn=nf=-25:nr=10:nt=w,"
            "acompressor=threshold=-25dB:ratio=2:attack=10:release=100:makeup=1,"
            "loudnorm=I=-16:TP=-1.5:LRA=9"
        ),
        output_wav,
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)


def enhance_hybrid_demucs_df(input_wav: str, output_wav: str):
    """Hybrid: Demucs vocals → DeepFilterNet denoise → loudnorm."""
    tmp_vocals = output_wav.replace(".wav", "_tmp_vocals.wav")
    tmp_df = output_wav.replace(".wav", "_tmp_df.wav")

    # Step 1: Demucs vocal separation
    enhance_demucs_vocals(input_wav, tmp_vocals)

    if not os.path.exists(tmp_vocals):
        return

    # Step 2: DeepFilterNet on vocals (gentle, atten_lim=12)
    enhance_deepfilternet(tmp_vocals, tmp_df, atten_lim=12)

    # Step 3: ffmpeg loudnorm
    cmd = [
        "ffmpeg", "-y", "-i", tmp_df,
        "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=7",
        output_wav,
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)

    # Cleanup
    for f in [tmp_vocals, tmp_df]:
        if os.path.exists(f):
            os.remove(f)


# ── Pipeline definitions ──────────────────────────────────────────────
PIPELINES = {
    "original": None,  # No processing, just score
    "deepfilter_full": lambda i, o: enhance_deepfilternet(i, o, atten_lim=None),
    "deepfilter_12dB": lambda i, o: enhance_deepfilternet(i, o, atten_lim=12),
    "mossformer2_48k": enhance_clearvoice_mossformer2,
    "frcrn_16k": enhance_clearvoice_frcrn,
    "demucs_vocals": enhance_demucs_vocals,
    "ffmpeg_gentle": enhance_ffmpeg_gentle,
    "hybrid_demucs_df": enhance_hybrid_demucs_df,
}


# ── Main ──────────────────────────────────────────────────────────────
def main():
    results = {}

    for fname in PILOT_FILES:
        input_wav = str(ORIG_DIR / f"{fname}.wav")
        if not os.path.exists(input_wav):
            print(f"[SKIP] {input_wav} not found")
            continue

        results[fname] = {}
        print(f"\n{'='*70}")
        print(f"FILE: {fname}")
        print(f"{'='*70}")

        for pipe_name, pipe_fn in PIPELINES.items():
            pipe_dir = OUT_DIR / pipe_name
            pipe_dir.mkdir(parents=True, exist_ok=True)

            if pipe_name == "original":
                wav_path = input_wav
            else:
                wav_path = str(pipe_dir / f"{fname}.wav")

            # Enhance (skip if output already exists)
            if pipe_name != "original":
                if os.path.exists(wav_path):
                    print(f"  [{pipe_name}] Using cached output")
                else:
                    print(f"  [{pipe_name}] Enhancing...", end=" ", flush=True)
                    t0 = time.time()
                    try:
                        pipe_fn(input_wav, wav_path)
                        elapsed = time.time() - t0
                        print(f"done ({elapsed:.1f}s)")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        results[fname][pipe_name] = {"error": str(e)}
                        continue

            if not os.path.exists(wav_path):
                results[fname][pipe_name] = {"error": "output file not created"}
                continue

            # Score
            print(f"  [{pipe_name}] Scoring...", end=" ", flush=True)
            t0 = time.time()
            try:
                scores = score_audio(wav_path)
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s)")
                results[fname][pipe_name] = scores
            except Exception as e:
                print(f"FAILED: {e}")
                results[fname][pipe_name] = {"error": str(e)}

            # Print key scores
            s = results[fname].get(pipe_name, {})
            if "dnsmos_ovrl" in s:
                dns_sig = s.get("dnsmos_sig", 0)
                dns_bak = s.get("dnsmos_bak", 0)
                dns_ovrl = s.get("dnsmos_ovrl", 0)
                n_mos = s.get("nisqa_mos", 0)
                n_noise = s.get("nisqa_noisiness", 0)
                print(
                    f"           DNSMOS: sig={dns_sig:.2f} "
                    f"bak={dns_bak:.2f} "
                    f"ovrl={dns_ovrl:.2f}  |  "
                    f"NISQA: mos={n_mos:.2f} "
                    f"noise={n_noise:.2f}"
                )

    # Save full report
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {REPORT_PATH}")

    # Print summary table
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Pipeline':<22} {'DNSMOS Sig':>10} {'DNSMOS Bak':>10} {'DNSMOS Ovrl':>11} {'NISQA MOS':>10} {'Noisiness':>10}")
    print("-" * 90)

    for fname in PILOT_FILES:
        print(f"\n  {fname}:")
        if fname not in results:
            continue
        for pipe_name in PIPELINES:
            s = results[fname].get(pipe_name, {})
            if "error" in s:
                print(f"    {pipe_name:<20} ERROR: {s['error'][:40]}")
            elif "dnsmos_ovrl" in s:
                print(
                    f"    {pipe_name:<20} "
                    f"{s.get('dnsmos_sig', 0):>8.2f}  "
                    f"{s.get('dnsmos_bak', 0):>8.2f}  "
                    f"{s.get('dnsmos_ovrl', 0):>9.2f}  "
                    f"{s.get('nisqa_mos', 0):>8.2f}  "
                    f"{s.get('nisqa_noisiness', 0):>8.2f}"
                )


if __name__ == "__main__":
    main()
