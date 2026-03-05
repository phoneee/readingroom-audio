"""Smoke tests for readingroom-audio package.

Fast tests that verify imports, pipeline registration, sampling logic,
and CLI entry points work without requiring ML model downloads or audio files.
"""

import subprocess
import sys

# ── Import tests ─────────────────────────────────────────────────────


def test_import_package():
    import readingroom_audio  # noqa: F401


def test_import_enhance():
    from readingroom_audio.enhance import PIPELINE_DESCRIPTIONS, PIPELINES

    assert isinstance(PIPELINES, dict)
    assert isinstance(PIPELINE_DESCRIPTIONS, dict)


def test_import_sampling():
    from readingroom_audio.sampling import (
        FORMAT_PIPELINE_MAP,
        SERIES_GROUP_MAP,
        stratified_sample,
    )

    assert isinstance(FORMAT_PIPELINE_MAP, dict)
    assert isinstance(SERIES_GROUP_MAP, dict)
    assert callable(stratified_sample)


def test_import_utils():
    from readingroom_audio.utils import (
        check_ffmpeg,
        encode_flac,
        encode_mp3,
        ensure_wav,
        get_project_root,
        load_audio,
        save_audio,
    )

    assert callable(load_audio)
    assert callable(save_audio)
    assert callable(ensure_wav)
    assert callable(encode_flac)
    assert callable(encode_mp3)
    assert callable(check_ffmpeg)
    assert callable(get_project_root)


def test_import_score():
    from readingroom_audio.score import score_segment

    assert callable(score_segment)


# ── Pipeline registry tests ──────────────────────────────────────────


def test_pipeline_registry_consistency():
    from readingroom_audio.enhance import PIPELINE_DESCRIPTIONS, PIPELINES

    assert set(PIPELINES.keys()) == set(PIPELINE_DESCRIPTIONS.keys()), (
        "PIPELINES and PIPELINE_DESCRIPTIONS must have the same keys"
    )


def test_pipeline_original_is_none():
    from readingroom_audio.enhance import PIPELINES

    assert PIPELINES["original"] is None


def test_pipeline_functions_are_callable():
    from readingroom_audio.enhance import PIPELINES

    for name, fn in PIPELINES.items():
        if name == "original":
            continue
        assert callable(fn), f"Pipeline '{name}' is not callable"


def test_known_pipelines_exist():
    from readingroom_audio.enhance import PIPELINES

    expected = [
        "original",
        "ffmpeg_gentle",
        "deepfilter_12dB",
        "hybrid_demucs_df",
        "hybrid_demucs_remix",
        "demucs_vocals",
    ]
    for name in expected:
        assert name in PIPELINES, f"Expected pipeline '{name}' not found"


def test_get_available_pipelines():
    from readingroom_audio.enhance import get_available_pipelines

    available = get_available_pipelines()
    assert isinstance(available, list)
    assert "original" in available
    assert "ffmpeg_gentle" in available


# ── Sampling logic tests ─────────────────────────────────────────────


def test_classify_era():
    from readingroom_audio.sampling import _classify_era

    assert _classify_era("2011-06-08") == "early"
    assert _classify_era("2014-05-01") == "middle"
    assert _classify_era("2017-09-30") == "late"
    assert _classify_era("") == "middle"
    assert _classify_era("invalid") == "middle"


def test_classify_format():
    from readingroom_audio.sampling import _classify_format

    assert _classify_format({"baseline_extraction": {"formats": ["Screening"]}}) == "screening"
    assert _classify_format({"baseline_extraction": {"formats": ["Lecture"]}}) == "lecture"
    assert _classify_format({"baseline_extraction": {"formats": ["Panel Discussion"]}}) == "panel"
    assert _classify_format({"baseline_extraction": {"formats": ["Book Club"]}}) == "book_club"
    assert _classify_format({"baseline_extraction": {"formats": ["Performance"]}}) == "performance"
    assert _classify_format({}) == "lecture"


def test_classify_duration():
    from readingroom_audio.sampling import _classify_duration

    assert _classify_duration(600) == "short"
    assert _classify_duration(3000) == "medium"
    assert _classify_duration(7200) == "long"


def test_format_pipeline_map_values():
    from readingroom_audio.enhance import PIPELINES
    from readingroom_audio.sampling import FORMAT_PIPELINE_MAP

    for fmt, pipeline in FORMAT_PIPELINE_MAP.items():
        assert pipeline in PIPELINES, (
            f"FORMAT_PIPELINE_MAP['{fmt}'] = '{pipeline}' not in PIPELINES"
        )


# ── Utility tests ────────────────────────────────────────────────────


def test_load_config():
    from readingroom_audio.utils import load_config

    config = load_config()
    assert isinstance(config, dict)
    assert "enhance-workers" in config
    assert "default-pipeline" in config
    assert config["default-pipeline"] == "hybrid_demucs_df"


def test_project_root():
    from readingroom_audio.utils import get_project_root

    root = get_project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "readingroom_audio").is_dir()


def test_ffmpeg_available():
    from readingroom_audio.utils import check_ffmpeg

    assert check_ffmpeg(), "ffmpeg not found — required for audio processing"


# ── CLI tests ────────────────────────────────────────────────────────


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "readingroom_audio", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "benchmark" in result.stdout
    assert "batch" in result.stdout


def test_cli_batch_help():
    result = subprocess.run(
        [sys.executable, "-m", "readingroom_audio", "batch", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "pipeline" in result.stdout


def test_cli_benchmark_help():
    result = subprocess.run(
        [sys.executable, "-m", "readingroom_audio", "benchmark", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "run-all" in result.stdout


def test_cli_mux_help():
    result = subprocess.run(
        [sys.executable, "-m", "readingroom_audio", "mux", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "verify" in result.stdout
