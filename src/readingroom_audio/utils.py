"""Shared helpers for audio enhancement pipeline — ffmpeg wrappers, file conversion."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio file as torch Tensor.

    Uses soundfile directly (torchaudio 2.10+ removed backend support
    and requires torchcodec which may not be installed).

    Returns:
        (waveform, sample_rate) — waveform shape: (channels, samples).
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels), torch expects (channels, samples)
    tensor = torch.from_numpy(data.T)
    return tensor, sr


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int):
    """Save torch Tensor as audio file.

    Args:
        path: Output file path (.wav, .flac, etc.).
        waveform: Shape (channels, samples) or (samples,).
        sample_rate: Sample rate in Hz.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # (channels, samples) -> (samples, channels) for soundfile
    data = waveform.cpu().numpy().T
    sf.write(path, data, sample_rate)


def ensure_wav(input_path: str, output_wav: str, sr: int = 48000) -> str:
    """Convert any audio file to WAV at specified sample rate.

    Returns path to WAV file (may be same as input if already WAV).
    """
    input_ext = Path(input_path).suffix.lower()
    if input_ext == ".wav":
        return input_path

    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-i", input_path,
        "-ar", str(sr), "-ac", "1",
        output_wav,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    return output_wav


def ffmpeg_loudnorm(input_path: str, output_path: str,
                    integrated: float = -16, tp: float = -1.5, lra: float = 7):
    """Apply ffmpeg loudness normalization with optional high-pass filter."""
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-i", input_path,
        "-af", f"highpass=f=80,loudnorm=I={integrated}:TP={tp}:LRA={lra}",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=300)


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def get_project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def encode_flac(input_wav: str, output_flac: str, compression_level: int = 5):
    """Encode WAV to FLAC (lossless, ~60% smaller than WAV).

    Args:
        input_wav: Path to input WAV file.
        output_flac: Path to output FLAC file.
        compression_level: FLAC compression level 0-12 (5 = good speed/size trade-off).
    """
    Path(output_flac).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-i", input_wav,
        "-c:a", "flac",
        "-compression_level", str(compression_level),
        output_flac,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=300)


def get_video_stream_info(path: str) -> dict:
    """Get video stream info (codec, resolution, fps, duration) via ffprobe.

    Returns dict with keys: codec, width, height, fps, duration.
    Raises subprocess.CalledProcessError if ffprobe fails.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(result.stdout)

    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})

    # Parse fractional fps (e.g. "30000/1001")
    fps_str = stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {
        "codec": stream.get("codec_name", "unknown"),
        "width": stream.get("width", 0),
        "height": stream.get("height", 0),
        "fps": round(fps, 3),
        "duration": float(fmt.get("duration", 0)),
    }


def check_ffmpeg():
    """Verify ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
