"""VAD-based segment extraction for audio benchmark evaluation.

Uses Silero VAD (via torch.hub) to find 45-second speech-active segments.
Avoids intros/outros and silence-heavy sections for more reliable scoring.
"""

import torch
import torchaudio


_VAD_MODEL = None
_VAD_UTILS = None


def load_vad_model():
    """Load Silero VAD model (cached after first call)."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        _VAD_MODEL = model
        _VAD_UTILS = utils
    return _VAD_MODEL, _VAD_UTILS


def find_best_segment(
    wav_path: str,
    duration: float = 45.0,
    skip_start: float = 60.0,
    skip_end: float = 30.0,
    step: float = 5.0,
    min_speech_ratio: float = 0.30,
) -> tuple[float, float, float]:
    """Find the best speech-active segment using Silero VAD.

    Algorithm:
    1. Load audio at 16kHz (VAD requirement)
    2. Run Silero VAD to get speech timestamps
    3. Slide a window of `duration` seconds, skipping intro/outro
    4. Pick the window with the highest speech ratio
    5. Fallback: if all windows < min_speech_ratio, use 25% offset

    Args:
        wav_path: Path to audio file.
        duration: Segment length in seconds.
        skip_start: Skip this many seconds from the beginning.
        skip_end: Skip this many seconds from the end.
        step: Window sliding step in seconds.
        min_speech_ratio: Minimum speech ratio threshold.

    Returns:
        (start_sec, end_sec, speech_ratio)
    """
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)  # mono

    # Resample to 16kHz for VAD
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    total_seconds = wav.shape[0] / sr

    # Get speech timestamps from Silero VAD
    model, utils = load_vad_model()
    get_speech_ts = utils[0]  # get_speech_timestamps

    speech_timestamps = get_speech_ts(wav, model, sampling_rate=sr)

    # Build a binary speech mask at 1-second resolution
    total_frames = int(total_seconds)
    speech_mask = [0.0] * total_frames
    for ts in speech_timestamps:
        start_s = ts["start"] / sr
        end_s = ts["end"] / sr
        for i in range(max(0, int(start_s)), min(total_frames, int(end_s) + 1)):
            speech_mask[i] = 1.0

    # Define search range
    search_start = min(skip_start, total_seconds * 0.25)
    search_end = max(search_start + duration, total_seconds - skip_end)
    window_samples = int(duration)

    best_start = search_start
    best_ratio = 0.0

    t = search_start
    while t + duration <= search_end:
        start_idx = int(t)
        end_idx = min(start_idx + window_samples, total_frames)
        if end_idx <= start_idx:
            t += step
            continue
        window = speech_mask[start_idx:end_idx]
        ratio = sum(window) / len(window) if window else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = t
        t += step

    # Fallback: if no good speech window found, use 25% offset
    if best_ratio < min_speech_ratio:
        best_start = total_seconds * 0.25
        best_ratio = 0.0  # signal that we used fallback

    # Clamp to audio bounds
    best_start = max(0, min(best_start, total_seconds - duration))
    best_end = best_start + duration

    return (round(best_start, 2), round(best_end, 2), round(best_ratio, 3))


def extract_segment(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    sr: int = 48000,
) -> str:
    """Extract a time segment from audio and save at target sample rate.

    Args:
        input_path: Source audio file.
        output_path: Output WAV path.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        sr: Output sample rate (48kHz for pipeline compatibility).

    Returns:
        Path to extracted segment.
    """
    wav, orig_sr = torchaudio.load(input_path)
    wav = wav.mean(dim=0)  # mono

    start_sample = int(start_sec * orig_sr)
    end_sample = int(end_sec * orig_sr)
    segment = wav[start_sample:end_sample]

    # Resample to target sr
    if orig_sr != sr:
        segment = torchaudio.functional.resample(segment, orig_sr, sr)

    # Save as mono WAV
    torchaudio.save(output_path, segment.unsqueeze(0), sr)
    return output_path
