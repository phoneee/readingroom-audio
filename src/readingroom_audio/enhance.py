"""Core enhancement functions — all audio enhancement pipelines.

Each pipeline takes an input WAV path and output WAV path.
Pipelines are registered in PIPELINES dict for easy iteration.
"""

import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .utils import load_audio, save_audio


# ── DeepFilterNet compatibility shim ─────────────────────────────────
# df/io.py imports torchaudio.backend.common.AudioMetaData which was
# removed in torchaudio 2.1+.  Inject a minimal stand-in so df can
# import without patching venv files (survives uv sync).

if "torchaudio.backend.common" not in sys.modules:

    @dataclass
    class _AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    _common = types.ModuleType("torchaudio.backend.common")
    _common.AudioMetaData = _AudioMetaData
    sys.modules["torchaudio.backend.common"] = _common

    if "torchaudio.backend" not in sys.modules:
        _backend = types.ModuleType("torchaudio.backend")
        _backend.common = _common
        sys.modules["torchaudio.backend"] = _backend


# ── Model cache ─────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def _get_cached_model(key: str, loader_fn):
    """Return a cached model instance, creating it on first access.

    Thread-safe for read-only inference: PyTorch model weights are
    immutable during eval(), so concurrent forward passes are safe.
    """
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = loader_fn()
    return _MODEL_CACHE[key]


# ── Device detection ────────────────────────────────────────────────

def _best_demucs_device() -> str:
    """Detect the best available device for Demucs."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ── Enhancement methods ──────────────────────────────────────────────


def enhance_clearvoice_mossformer2(input_wav: str, output_wav: str):
    """ClearVoice-Studio MossFormer2 48kHz speech enhancement.

    Best for: general speech enhancement with good quality preservation.
    Characteristics: preserves natural speech quality, moderate noise reduction.
    """
    from clearvoice import ClearVoice

    cv = _get_cached_model(
        "clearvoice_mossformer2_48k",
        lambda: ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"]),
    )
    result = cv(input_path=input_wav, online_write=False)
    cv.write(result, output_path=output_wav)


def enhance_clearvoice_frcrn(input_wav: str, output_wav: str):
    """ClearVoice-Studio FRCRN 16kHz speech enhancement.

    Best for: highly noisy recordings where 16kHz output is acceptable.
    Characteristics: aggressive noise reduction, downsamples to 16kHz.
    """
    from clearvoice import ClearVoice

    cv = _get_cached_model(
        "clearvoice_frcrn_16k",
        lambda: ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"]),
    )
    result = cv(input_path=input_wav, online_write=False)
    cv.write(result, output_path=output_wav)


def enhance_deepfilternet(input_wav: str, output_wav: str,
                          atten_lim: int | None = None):
    """DeepFilterNet3 noise suppression.

    Args:
        atten_lim: Maximum attenuation in dB. None = unlimited (aggressive).
                   12 = gentle suppression preserving more ambient sound.

    Best for: targeted noise suppression while preserving speech naturalness.
    """
    from df.enhance import enhance, init_df
    import torchaudio

    model, df_state = _get_cached_model(
        "deepfilternet3",
        lambda: init_df()[:2],  # init_df returns (model, df_state, _)
    )
    audio, sr = load_audio(input_wav)
    if sr != df_state.sr():
        audio = torchaudio.functional.resample(audio, sr, df_state.sr())
    enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim)
    save_audio(output_wav, enhanced, df_state.sr())


def enhance_demucs_vocals(input_wav: str, output_wav: str):
    """Demucs htdemucs vocal separation — extract speech stem.

    Best for: isolating speech from music/background in mixed audio.
    Characteristics: separates vocals from accompaniment using music source separation.
    Uses MPS GPU when available (with CPU fallback).
    """
    out_dir = str(Path(output_wav).parent / "_demucs_tmp")
    device = _best_demucs_device()

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "--device", device,
        "-o", out_dir,
        input_wav,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=1200)

    # Fallback to CPU if MPS fails
    if result.returncode != 0 and device == "mps":
        cmd[cmd.index("--device") + 1] = "cpu"
        result = subprocess.run(cmd, capture_output=True, timeout=1200)

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"Demucs htdemucs failed (exit {result.returncode}): {stderr}")

    stem_name = Path(input_wav).stem
    vocals_path = Path(out_dir) / "htdemucs" / stem_name / "vocals.wav"
    if vocals_path.exists():
        data, sr = sf.read(str(vocals_path))
        sf.write(output_wav, data, sr)

    # Cleanup Demucs temp directory
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)


def enhance_ffmpeg_gentle(input_wav: str, output_wav: str):
    """ffmpeg-only: high-pass + mild FFT denoise + compression + loudnorm.

    Best for: lightweight processing without ML models. Fast, deterministic.
    Characteristics: no ML dependency, good baseline, preserves ambient sound.
    """
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-i", input_wav,
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
    """Hybrid: Demucs vocals -> DeepFilterNet 12dB -> loudnorm.

    Best for: maximum quality improvement while preserving ambient atmosphere.
    Characteristics: three-stage pipeline combining vocal separation,
    gentle noise suppression, and loudness normalization. User's preferred pipeline.
    """
    tmp_vocals = output_wav.replace(".wav", "_tmp_vocals.wav")
    tmp_df = output_wav.replace(".wav", "_tmp_df.wav")

    try:
        # Stage 1: Demucs vocal separation
        enhance_demucs_vocals(input_wav, tmp_vocals)
        if not os.path.exists(tmp_vocals):
            return

        # Stage 2: DeepFilterNet on vocals (gentle, atten_lim=12)
        enhance_deepfilternet(tmp_vocals, tmp_df, atten_lim=12)

        # Stage 3: ffmpeg loudnorm
        cmd = [
            "ffmpeg", "-y", "-threads", "0", "-i", tmp_df,
            "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=7",
            output_wav,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
    finally:
        for f in [tmp_vocals, tmp_df]:
            if os.path.exists(f):
                os.remove(f)


# ── Phase 1: Zero-install pipelines ─────────────────────────────────


def enhance_mossformergan_16k(input_wav: str, output_wav: str):
    """ClearVoice MossFormerGAN 16kHz GAN-based speech enhancement.

    Best for: perceptually superior enhancement (PESQ 3.57).
    Characteristics: GAN training produces more natural-sounding results
    than regression-based models, but outputs at 16kHz.
    """
    from clearvoice import ClearVoice

    cv = _get_cached_model(
        "clearvoice_mossformergan_16k",
        lambda: ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"]),
    )
    result = cv(input_path=input_wav, online_write=False)
    cv.write(result, output_path=output_wav)


def enhance_demucs_ft_vocals(input_wav: str, output_wav: str):
    """Demucs htdemucs_ft fine-tuned vocal separation.

    Best for: higher-quality vocal isolation than base htdemucs.
    Characteristics: ~4x slower than htdemucs, but better separation quality.
    Uses MPS GPU when available (with CPU fallback).
    """
    out_dir = str(Path(output_wav).parent / "_demucs_ft_tmp")
    device = _best_demucs_device()

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs_ft",
        "--two-stems", "vocals",
        "--device", device,
        "-o", out_dir,
        input_wav,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=2400)

    # Fallback to CPU if MPS fails
    if result.returncode != 0 and device == "mps":
        cmd[cmd.index("--device") + 1] = "cpu"
        result = subprocess.run(cmd, capture_output=True, timeout=2400)

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"Demucs htdemucs_ft failed (exit {result.returncode}): {stderr}")

    stem_name = Path(input_wav).stem
    vocals_path = Path(out_dir) / "htdemucs_ft" / stem_name / "vocals.wav"
    if vocals_path.exists():
        data, sr = sf.read(str(vocals_path))
        sf.write(output_wav, data, sr)

    # Cleanup Demucs temp directory
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)


def enhance_superres_48k(input_wav: str, output_wav: str):
    """ClearVoice MossFormer2 speech super-resolution to 48kHz.

    Best for: upsampling low-quality audio to wideband 48kHz.
    Characteristics: recovers high-frequency detail lost in compression.
    Not a denoiser — best used after a denoising stage.
    """
    from clearvoice import ClearVoice

    cv = _get_cached_model(
        "clearvoice_sr_48k",
        lambda: ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"]),
    )
    result = cv(input_path=input_wav, online_write=False)
    cv.write(result, output_path=output_wav)


def enhance_hybrid_mossformergan_sr(input_wav: str, output_wav: str):
    """Hybrid: MossFormerGAN 16K → SuperRes 48K → loudnorm.

    Best for: GAN-quality enhancement with wideband output.
    Characteristics: GAN denoising at 16kHz, then super-resolution
    recovers high frequencies, loudnorm normalizes output.
    """
    tmp_gan = output_wav.replace(".wav", "_tmp_gan.wav")
    tmp_sr = output_wav.replace(".wav", "_tmp_sr.wav")

    try:
        # Stage 1: MossFormerGAN denoising at 16kHz
        enhance_mossformergan_16k(input_wav, tmp_gan)
        if not os.path.exists(tmp_gan):
            return

        # Stage 2: Super-resolution to 48kHz
        enhance_superres_48k(tmp_gan, tmp_sr)
        if not os.path.exists(tmp_sr):
            return

        # Stage 3: loudnorm
        cmd = [
            "ffmpeg", "-y", "-threads", "0", "-i", tmp_sr,
            "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=7",
            output_wav,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
    finally:
        for f in [tmp_gan, tmp_sr]:
            if os.path.exists(f):
                os.remove(f)


def enhance_hybrid_demucs_ft_df(input_wav: str, output_wav: str):
    """Hybrid: Demucs_ft vocals → DeepFilter 12dB → loudnorm.

    Best for: premium vocal separation + gentle noise suppression.
    Characteristics: like hybrid_demucs_df but with fine-tuned Demucs model.
    """
    tmp_vocals = output_wav.replace(".wav", "_tmp_ft_vocals.wav")
    tmp_df = output_wav.replace(".wav", "_tmp_ft_df.wav")

    try:
        # Stage 1: Fine-tuned Demucs vocal separation
        enhance_demucs_ft_vocals(input_wav, tmp_vocals)
        if not os.path.exists(tmp_vocals):
            return

        # Stage 2: DeepFilterNet gentle suppression
        enhance_deepfilternet(tmp_vocals, tmp_df, atten_lim=12)

        # Stage 3: loudnorm
        cmd = [
            "ffmpeg", "-y", "-threads", "0", "-i", tmp_df,
            "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=7",
            output_wav,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
    finally:
        for f in [tmp_vocals, tmp_df]:
            if os.path.exists(f):
                os.remove(f)


def enhance_hybrid_demucs_ft_mossformer(input_wav: str, output_wav: str):
    """Hybrid: Demucs_ft vocals → MossFormer2 48K → loudnorm.

    Best for: fine-tuned separation + neural enhancement.
    Characteristics: combines best vocal isolation with MossFormer2's
    natural speech enhancement, maintains 48kHz output.
    """
    tmp_vocals = output_wav.replace(".wav", "_tmp_ft_vocals2.wav")
    tmp_mf = output_wav.replace(".wav", "_tmp_ft_mf.wav")

    try:
        # Stage 1: Fine-tuned Demucs vocal separation
        enhance_demucs_ft_vocals(input_wav, tmp_vocals)
        if not os.path.exists(tmp_vocals):
            return

        # Stage 2: MossFormer2 48K enhancement
        enhance_clearvoice_mossformer2(tmp_vocals, tmp_mf)

        # Stage 3: loudnorm
        cmd = [
            "ffmpeg", "-y", "-threads", "0", "-i", tmp_mf,
            "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=7",
            output_wav,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
    finally:
        for f in [tmp_vocals, tmp_mf]:
            if os.path.exists(f):
                os.remove(f)


# ── Phase 2: New-package pipelines ──────────────────────────────────


def enhance_mpsenet(input_wav: str, output_wav: str):
    """MP-SENet magnitude+phase speech enhancement (16kHz).

    Best for: joint magnitude-phase enhancement with strong DNS scores.
    Characteristics: processes both magnitude and phase of STFT, outputs 16kHz.
    Requires: MPSENet package.
    """
    try:
        from MPSENet import MPSENet
    except ImportError:
        raise RuntimeError("MPSENet not installed: pip install MPSENet")

    import numpy as np
    import torch
    import torchaudio

    model = MPSENet.from_pretrained("JacobLinCool/MP-SENet-DNS")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    wav, sr = load_audio(input_wav)
    wav = wav.mean(dim=0)  # mono

    # Resample to 16kHz (MPSENet's native rate)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    audio_np = wav.numpy()
    enhanced_np, out_sr, _ = model(audio_np)
    enhanced = torch.from_numpy(enhanced_np).unsqueeze(0)
    save_audio(output_wav, enhanced, out_sr)


def enhance_hybrid_mpsenet_sr(input_wav: str, output_wav: str):
    """Hybrid: MP-SENet → SuperRes 48K → loudnorm.

    Best for: MP-SENet quality with wideband output via super-resolution.
    """
    tmp_mpsenet = output_wav.replace(".wav", "_tmp_mpsenet.wav")
    tmp_sr = output_wav.replace(".wav", "_tmp_mpsenet_sr.wav")

    try:
        enhance_mpsenet(input_wav, tmp_mpsenet)
        if not os.path.exists(tmp_mpsenet):
            return

        enhance_superres_48k(tmp_mpsenet, tmp_sr)
        if not os.path.exists(tmp_sr):
            return

        cmd = [
            "ffmpeg", "-y", "-threads", "0", "-i", tmp_sr,
            "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=7",
            output_wav,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
    finally:
        for f in [tmp_mpsenet, tmp_sr]:
            if os.path.exists(f):
                os.remove(f)


def enhance_resemble_denoise(input_wav: str, output_wav: str):
    """Resemble Enhance — denoise only (44.1kHz).

    Best for: high-quality denoising without upscaling.
    Characteristics: operates at 44.1kHz, preserves natural quality.
    Requires: resemble-enhance + deepspeed (CUDA only).
    """
    try:
        from resemble_enhance.enhancer.inference import denoise
    except ImportError as e:
        raise RuntimeError(
            f"resemble-enhance not available: {e}. "
            "Requires deepspeed (CUDA only, not available on macOS)."
        )

    import torch
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sr = load_audio(input_wav)
    wav = wav.mean(dim=0)  # mono

    # Resample to 44.1kHz (Resemble's native rate)
    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)

    enhanced = denoise(wav, 44100, device)
    save_audio(output_wav, enhanced.unsqueeze(0).cpu(), 44100)


def enhance_resemble_full(input_wav: str, output_wav: str):
    """Resemble Enhance — denoise + enhance/upscale (44.1kHz).

    Best for: maximum quality improvement with upscaling.
    Characteristics: two-stage model (denoise → enhance), operates at 44.1kHz.
    Requires: resemble-enhance + deepspeed (CUDA only).
    """
    try:
        from resemble_enhance.enhancer.inference import denoise, enhance
    except ImportError as e:
        raise RuntimeError(
            f"resemble-enhance not available: {e}. "
            "Requires deepspeed (CUDA only, not available on macOS)."
        )

    import torch
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sr = load_audio(input_wav)
    wav = wav.mean(dim=0)  # mono

    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)

    # Stage 1: Denoise
    denoised = denoise(wav, 44100, device)
    # Stage 2: Enhance/upscale
    enhanced, new_sr = enhance(denoised, 44100, device, nfe=32)
    save_audio(output_wav, enhanced.unsqueeze(0).cpu(), new_sr)


def enhance_sepformer_wham16k(input_wav: str, output_wav: str):
    """SpeechBrain SepFormer WHAM! 16kHz denoising.

    Best for: speech separation/denoising in noisy environments.
    Characteristics: trained on WHAM! noisy data, 16kHz output.
    Requires: speechbrain package (may have torchaudio version issues).
    """
    try:
        from speechbrain.inference.separation import SepformerSeparation
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"speechbrain not available: {e}. "
            "May need compatible torchaudio version."
        )

    import torchaudio

    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wham16k-enhancement",
        savedir="/tmp/speechbrain_sepformer",
    )
    est_sources = model.separate_file(path=input_wav)

    # SepFormer returns (batch, time, sources) — take first source
    enhanced = est_sources[:, :, 0].squeeze(0)
    save_audio(output_wav, enhanced.unsqueeze(0).cpu(), 16000)


# ── Pipeline availability probing ────────────────────────────────────

_PIPELINE_DEPS: dict[str, str] = {
    "mossformer2_48k": "clearvoice",
    "mossformergan_16k": "clearvoice",
    "frcrn_16k": "clearvoice",
    "superres_48k": "clearvoice",
    "hybrid_mossformergan_sr": "clearvoice",
    "hybrid_demucs_ft_mossformer": "clearvoice",
    "resemble_denoise": "resemble_enhance",
    "resemble_full": "resemble_enhance",
    "mpsenet_dns": "MPSENet",
    "hybrid_mpsenet_sr": "MPSENet",
    "sepformer_wham16k": "speechbrain",
}


def get_available_pipelines() -> list[str]:
    """Return pipeline names whose dependencies are importable."""
    available = []
    for name in PIPELINES:
        dep = _PIPELINE_DEPS.get(name)
        if dep is None:
            available.append(name)
        else:
            try:
                __import__(dep)
                available.append(name)
            except Exception:
                pass
    return available


# ── Pipeline registry ────────────────────────────────────────────────

PIPELINES = {
    # Original baseline
    "original": None,
    # Single-model pipelines
    "deepfilter_full": lambda i, o: enhance_deepfilternet(i, o, atten_lim=None),
    "deepfilter_12dB": lambda i, o: enhance_deepfilternet(i, o, atten_lim=12),
    "deepfilter_6dB": lambda i, o: enhance_deepfilternet(i, o, atten_lim=6),
    "deepfilter_18dB": lambda i, o: enhance_deepfilternet(i, o, atten_lim=18),
    "mossformer2_48k": enhance_clearvoice_mossformer2,
    "mossformergan_16k": enhance_mossformergan_16k,
    "frcrn_16k": enhance_clearvoice_frcrn,
    "demucs_vocals": enhance_demucs_vocals,
    "demucs_ft_vocals": enhance_demucs_ft_vocals,
    "superres_48k": enhance_superres_48k,
    "ffmpeg_gentle": enhance_ffmpeg_gentle,
    # Hybrid pipelines
    "hybrid_demucs_df": enhance_hybrid_demucs_df,
    "hybrid_mossformergan_sr": enhance_hybrid_mossformergan_sr,
    "hybrid_demucs_ft_df": enhance_hybrid_demucs_ft_df,
    "hybrid_demucs_ft_mossformer": enhance_hybrid_demucs_ft_mossformer,
    # Phase 2: new-package pipelines
    "mpsenet_dns": enhance_mpsenet,
    "hybrid_mpsenet_sr": enhance_hybrid_mpsenet_sr,
    "resemble_denoise": enhance_resemble_denoise,
    "resemble_full": enhance_resemble_full,
    "sepformer_wham16k": enhance_sepformer_wham16k,
}

PIPELINE_DESCRIPTIONS = {
    "original": "No processing (baseline)",
    "deepfilter_full": "DeepFilterNet3 — unlimited attenuation (aggressive)",
    "deepfilter_12dB": "DeepFilterNet3 — 12dB attenuation limit (gentle)",
    "deepfilter_6dB": "DeepFilterNet3 — 6dB attenuation limit (minimal)",
    "deepfilter_18dB": "DeepFilterNet3 — 18dB attenuation limit (strong)",
    "mossformer2_48k": "ClearVoice MossFormer2 48kHz speech enhancement",
    "mossformergan_16k": "ClearVoice MossFormerGAN 16kHz GAN-based enhancement",
    "frcrn_16k": "ClearVoice FRCRN 16kHz speech enhancement",
    "demucs_vocals": "Demucs htdemucs vocal separation",
    "demucs_ft_vocals": "Demucs htdemucs_ft fine-tuned vocal separation",
    "superres_48k": "ClearVoice MossFormer2 super-resolution to 48kHz",
    "ffmpeg_gentle": "ffmpeg high-pass + FFT denoise + compression + loudnorm",
    "hybrid_demucs_df": "Demucs vocals → DeepFilterNet 12dB → loudnorm",
    "hybrid_mossformergan_sr": "MossFormerGAN 16K → SuperRes 48K → loudnorm",
    "hybrid_demucs_ft_df": "Demucs_ft vocals → DeepFilter 12dB → loudnorm",
    "hybrid_demucs_ft_mossformer": "Demucs_ft vocals → MossFormer2 48K → loudnorm",
    "mpsenet_dns": "MP-SENet magnitude+phase speech enhancement (16kHz)",
    "hybrid_mpsenet_sr": "MP-SENet → SuperRes 48K → loudnorm",
    "resemble_denoise": "Resemble Enhance — denoise only (44.1kHz)",
    "resemble_full": "Resemble Enhance — denoise + enhance/upscale (44.1kHz)",
    "sepformer_wham16k": "SpeechBrain SepFormer WHAM! 16kHz denoising",
}
