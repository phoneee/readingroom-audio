"""Systematic audio enhancement benchmark with statistical analysis.

Sub-command CLI for running stratified benchmark across all 161 events:

    python -m readingroom_audio.benchmark select       [--target-n 40] [--seed 42]
    python -m readingroom_audio.benchmark download
    python -m readingroom_audio.benchmark extract      [--duration 45]
    python -m readingroom_audio.benchmark baseline
    python -m readingroom_audio.benchmark enhance      [--pipelines ...]
    python -m readingroom_audio.benchmark analyze
    python -m readingroom_audio.benchmark export       [--output-dir ...] [--n-samples 3]
    python -m readingroom_audio.benchmark preview      [--output-dir ...] [--n-samples 8]
    python -m readingroom_audio.benchmark sensitivity  [--target-n 40] [--seeds ...]
    python -m readingroom_audio.benchmark run-all      [--target-n 40] [--pipelines ...] [--quick]
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from .download import download_audio
from .enhance import PIPELINES, PIPELINE_DESCRIPTIONS, get_available_pipelines
from .sampling import FORMAT_PIPELINE_MAP, load_all_events, stratified_sample
from .score import save_report, score_segment
from .utils import encode_mp3, ensure_wav, get_project_root

# ── Curated defaults ────────────────────────────────────────────────

BENCHMARK_DEFAULT_PIPELINES = [
    "original", "ffmpeg_gentle", "deepfilter_12dB", "deepfilter_full",
    "demucs_vocals", "hybrid_demucs_df",
]

QUICK_PIPELINES = ["original", "ffmpeg_gentle", "hybrid_demucs_df"]

# ── Paths ────────────────────────────────────────────────────────────

ROOT = get_project_root()
EVENTS_DIR = ROOT / "data" / "events"
AUDIO_DIR = ROOT / "data" / "audio"
MANIFEST_PATH = AUDIO_DIR / "benchmark_manifest.json"
RESULTS_PATH = AUDIO_DIR / "benchmark_results.json"
REPORT_PATH = AUDIO_DIR / "benchmark_report.md"
CHARTS_DIR = AUDIO_DIR / "benchmark_charts"
DOWNLOADS_DIR = AUDIO_DIR / "benchmark_downloads"
SEGMENTS_DIR = AUDIO_DIR / "benchmark_segments"
ENHANCED_DIR = AUDIO_DIR / "benchmark_enhanced"


# ── Fail-fast tracker ────────────────────────────────────────────────

def _classify_error(exc: Exception) -> str:
    """Map an exception to an error category for fail-fast grouping."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "import" in name or "modulenotfound" in name:
        return "import_error"
    if "import" in msg or "not installed" in msg or "not available" in msg:
        return "import_error"
    return "runtime_error"


class PipelineFailTracker:
    """Track consecutive failures per pipeline for fail-fast behavior.

    After `threshold` consecutive failures with the same error category,
    the pipeline is disabled for the rest of the run.
    """

    def __init__(self, threshold: int = 2, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled
        self._consecutive: dict[str, int] = {}
        self._last_category: dict[str, str] = {}
        self._disabled: dict[str, str] = {}  # pipeline -> reason

    def should_skip(self, pipeline: str) -> bool:
        if not self.enabled:
            return False
        return pipeline in self._disabled

    def disabled_reason(self, pipeline: str) -> str:
        return self._disabled.get(pipeline, "")

    def record_failure(self, pipeline: str, category: str):
        if not self.enabled:
            return
        prev_cat = self._last_category.get(pipeline)
        if prev_cat == category:
            self._consecutive[pipeline] = self._consecutive.get(pipeline, 1) + 1
        else:
            self._consecutive[pipeline] = 1
            self._last_category[pipeline] = category
        if self._consecutive[pipeline] >= self.threshold:
            self._disabled[pipeline] = (
                f"{self._consecutive[pipeline]}x consecutive {category}"
            )

    def record_success(self, pipeline: str):
        self._consecutive.pop(pipeline, None)
        self._last_category.pop(pipeline, None)

    def summary(self) -> str:
        if not self._disabled:
            return "  Fail-fast: no pipelines disabled"
        lines = ["  Fail-fast disabled pipelines:"]
        for pipe, reason in sorted(self._disabled.items()):
            lines.append(f"    {pipe}: {reason}")
        return "\n".join(lines)


# ── Structured benchmark logger ─────────────────────────────────────

class BenchmarkLogger:
    """JSONL structured logger for benchmark runs.

    Each line records one event (enhance, score, skip) with timing and
    error information. Flushes after each write for crash safety.
    """

    def __init__(self, log_dir: Path | None = None):
        if log_dir is None:
            log_dir = AUDIO_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"benchmark_run_{ts}.jsonl"
        self._fh = None
        self._counts: dict[str, dict[str, int]] = {}  # pipeline -> {ok, error, skipped}
        self._times: dict[str, float] = {}  # pipeline -> total enhance seconds
        self._scores: dict[str, list[float]] = {}  # pipeline -> list of OVRL scores

    def __enter__(self):
        self._fh = open(self.path, "a")
        return self

    def __exit__(self, *exc):
        if self._fh:
            self._fh.close()
            self._fh = None

    def log(
        self,
        phase: str,
        segment_id: str,
        pipeline: str,
        action: str,
        status: str,
        duration_sec: float = 0.0,
        error_message: str = "",
        error_category: str = "",
        scores: dict | None = None,
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "segment_id": segment_id,
            "pipeline": pipeline,
            "action": action,
            "status": status,
            "duration_sec": round(duration_sec, 2),
        }
        if error_message:
            record["error_message"] = error_message
        if error_category:
            record["error_category"] = error_category
        if scores:
            record["scores"] = scores

        if self._fh:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()

        # Track stats for summary
        bucket = self._counts.setdefault(pipeline, {"ok": 0, "error": 0, "skipped": 0})
        if status in ("ok", "cached"):
            bucket["ok"] += 1
        elif status == "error":
            bucket["error"] += 1
        elif status == "skipped":
            bucket["skipped"] += 1

        if action == "enhance" and status in ("ok",) and duration_sec > 0:
            self._times[pipeline] = self._times.get(pipeline, 0) + duration_sec

        if scores and "dnsmos_ovrl" in scores:
            self._scores.setdefault(pipeline, []).append(scores["dnsmos_ovrl"])

    def print_summary(self):
        if not self._counts:
            return
        print(f"\n{'='*70}")
        print(f"BENCHMARK RUN SUMMARY  (log: {self.path.name})")
        print(f"{'='*70}")
        print(f"  {'Pipeline':<30} {'OK':>4} {'Err':>4} {'Skip':>5} {'Time':>7} {'OVRL':>6}")
        print(f"  {'-'*28}  {'-'*4} {'-'*4} {'-'*5} {'-'*7} {'-'*6}")
        for pipe in sorted(self._counts):
            c = self._counts[pipe]
            t = self._times.get(pipe, 0)
            ovrl_list = self._scores.get(pipe, [])
            mean_ovrl = sum(ovrl_list) / len(ovrl_list) if ovrl_list else 0
            ovrl_str = f"{mean_ovrl:.2f}" if ovrl_list else "—"
            time_str = f"{t:.0f}s" if t > 0 else "—"
            print(
                f"  {pipe:<30} {c['ok']:>4} {c['error']:>4} "
                f"{c['skipped']:>5} {time_str:>7} {ovrl_str:>6}"
            )
        print(f"\n  Log file: {self.path}")


# ── Manifest I/O ─────────────────────────────────────────────────────

def _load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def _save_manifest(manifest: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def _save_results(results: dict):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Phase: select ────────────────────────────────────────────────────

def cmd_select(target_n: int = 40, seed: int = 42):
    """Select stratified sample of events for benchmark."""
    print(f"Loading events from {EVENTS_DIR}...")
    events = load_all_events(EVENTS_DIR)
    print(f"  Found {len(events)} events")

    manifest = stratified_sample(events, target_n=target_n, seed=seed)
    print(f"  Selected {len(manifest)} samples")

    # Print distribution
    from collections import Counter
    groups = Counter(e["strata"]["series_group"] for e in manifest)
    eras = Counter(e["strata"]["era"] for e in manifest)
    print(f"\n  Series groups: {dict(sorted(groups.items()))}")
    print(f"  Eras: {dict(sorted(eras.items()))}")

    _save_manifest(manifest)
    print(f"\n  Manifest saved to {MANIFEST_PATH}")
    return manifest


# ── Phase: download ──────────────────────────────────────────────────

def cmd_download():
    """Download audio for all samples in manifest."""
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(manifest)
    cached = 0
    downloaded = 0
    errors = 0

    for i, entry in enumerate(manifest, 1):
        vid = entry["video_id"]
        if not vid:
            print(f"  [{i}/{total}] No video ID for {entry['segment_id']}, skipping")
            errors += 1
            continue

        m4a_path = DOWNLOADS_DIR / f"{vid}.m4a"
        if m4a_path.exists():
            entry["status"]["downloaded"] = True
            cached += 1
            continue

        print(f"  [{i}/{total}] Downloading {vid}...", end=" ", flush=True)
        try:
            download_audio(vid, DOWNLOADS_DIR, format="m4a")
            entry["status"]["downloaded"] = True
            downloaded += 1
            print("done")
        except Exception as e:
            errors += 1
            print(f"FAILED: {e}")

    _save_manifest(manifest)
    print(f"\nDownload summary: {cached} cached, {downloaded} new, {errors} errors")


# ── Phase: extract ───────────────────────────────────────────────────

def cmd_extract(duration: float = 45.0):
    """Extract speech-active segments using Silero VAD."""
    from .segment import extract_segment, find_best_segment

    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(manifest)

    for i, entry in enumerate(manifest, 1):
        sid = entry["segment_id"]
        vid = entry["video_id"]
        segment_path = SEGMENTS_DIR / f"{sid}.wav"

        if segment_path.exists():
            entry["status"]["segment_extracted"] = True
            print(f"  [{i}/{total}] {sid}: cached")
            continue

        # Find source audio (m4a or wav)
        m4a_path = DOWNLOADS_DIR / f"{vid}.m4a"
        if not m4a_path.exists():
            print(f"  [{i}/{total}] {sid}: source not found, skipping")
            continue

        # Convert to WAV for VAD processing
        tmp_wav = str(SEGMENTS_DIR / f"_tmp_{vid}.wav")
        print(f"  [{i}/{total}] {sid}: converting to WAV...", end=" ", flush=True)
        try:
            ensure_wav(str(m4a_path), tmp_wav, sr=48000)
        except Exception as e:
            print(f"conversion FAILED: {e}")
            continue

        # Find best speech segment via VAD
        print("VAD...", end=" ", flush=True)
        try:
            start, end, ratio = find_best_segment(tmp_wav, duration=duration)
            entry["segment"]["start_sec"] = start
            entry["segment"]["end_sec"] = end
            entry["segment"]["speech_ratio"] = ratio
            print(f"[{start:.1f}–{end:.1f}s, speech={ratio:.0%}]", end=" ")
        except Exception as e:
            print(f"VAD FAILED: {e}")
            # Fallback: use 25% offset
            total_dur = entry.get("video_duration_seconds", 300)
            start = total_dur * 0.25
            end = start + duration
            entry["segment"]["start_sec"] = start
            entry["segment"]["end_sec"] = end
            entry["segment"]["speech_ratio"] = 0.0
            print(f"[fallback {start:.1f}–{end:.1f}s]", end=" ")

        # Extract segment
        print("extracting...", end=" ", flush=True)
        try:
            extract_segment(tmp_wav, str(segment_path), start, end, sr=48000)
            entry["status"]["segment_extracted"] = True
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

        # Clean up temp WAV
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    _save_manifest(manifest)
    print(f"\nSegments saved to {SEGMENTS_DIR}")


# ── Phase: baseline ──────────────────────────────────────────────────

def cmd_baseline():
    """Score original (unenhanced) segments."""
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    total = len(manifest)
    scored = 0

    for i, entry in enumerate(manifest, 1):
        sid = entry["segment_id"]
        segment_path = SEGMENTS_DIR / f"{sid}.wav"

        if entry.get("baseline_scores", {}).get("dnsmos_ovrl") is not None:
            print(f"  [{i}/{total}] {sid}: already scored")
            scored += 1
            continue

        if not segment_path.exists():
            print(f"  [{i}/{total}] {sid}: segment not found, skipping")
            continue

        print(f"  [{i}/{total}] {sid}: scoring...", end=" ", flush=True)
        t0 = time.time()
        try:
            scores = score_segment(str(segment_path))
            entry["baseline_scores"] = scores
            entry["status"]["baseline_scored"] = True
            elapsed = time.time() - t0
            ovrl = scores.get("dnsmos_ovrl", "?")
            print(f"done ({elapsed:.1f}s) OVRL={ovrl}")
            scored += 1
        except Exception as e:
            print(f"FAILED: {e}")

    _save_manifest(manifest)
    print(f"\nBaseline scoring complete: {scored}/{total} scored")


# ── Phase: enhance ───────────────────────────────────────────────────

def cmd_enhance(
    pipeline_names: list[str] | None = None,
    metrics: list[str] | None = None,
    fail_fast: bool = True,
):
    """Run enhancement pipelines on all segments and score results.

    Args:
        pipeline_names: Pipelines to run. None = available pipelines only.
        metrics: Metric sets for scoring. None = all (dnsmos, nisqa, utmos).
        fail_fast: Disable pipelines after 2 consecutive same-category failures.
    """
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    if pipeline_names is None:
        pipeline_names = [p for p in get_available_pipelines() if p != "original"]
        print(f"  Auto-detected {len(pipeline_names)} available pipelines")
    else:
        pipeline_names = [p for p in pipeline_names if p != "original"]

    # Skip unavailable pipelines
    available = set(get_available_pipelines())
    skipped = [p for p in pipeline_names if p not in available]
    if skipped:
        print(f"  Skipping unavailable: {', '.join(skipped)}")
    pipeline_names = [p for p in pipeline_names if p in available]

    if not pipeline_names:
        print("  No pipelines to run.")
        return

    tracker = PipelineFailTracker(enabled=fail_fast)
    results = _load_results()

    with BenchmarkLogger() as logger:
        for entry in manifest:
            sid = entry["segment_id"]
            segment_path = SEGMENTS_DIR / f"{sid}.wav"

            if not segment_path.exists():
                print(f"\n[SKIP] {sid}: segment not found")
                continue

            print(f"\n{'='*60}")
            print(f"SEGMENT: {sid} ({entry['strata']['series_group']}/{entry['strata']['era']})")
            print(f"{'='*60}")

            if sid not in results:
                results[sid] = {
                    "strata": entry["strata"],
                    "pipelines": {},
                }

            # Score original inline (replaces separate baseline phase)
            orig_data = results[sid].get("pipelines", {}).get("original", {})
            if orig_data.get("scores", {}).get("dnsmos_ovrl") is None:
                print(f"  [original] Scoring baseline...", end=" ", flush=True)
                t0 = time.time()
                try:
                    orig_scores = score_segment(str(segment_path), metrics=metrics)
                    elapsed = time.time() - t0
                    results[sid].setdefault("pipelines", {})["original"] = {
                        "scores": orig_scores,
                        "processing_time_sec": 0,
                    }
                    ovrl = orig_scores.get("dnsmos_ovrl", "?")
                    print(f"done ({elapsed:.1f}s) OVRL={ovrl}")
                    logger.log("enhance", sid, "original", "score", "ok",
                               duration_sec=elapsed, scores=orig_scores)
                except Exception as e:
                    print(f"FAILED: {e}")
                    results[sid].setdefault("pipelines", {})["original"] = {
                        "scores": {"error": str(e)},
                        "processing_time_sec": 0,
                    }
                    logger.log("enhance", sid, "original", "score", "error",
                               error_message=str(e))
            else:
                ovrl = orig_data["scores"].get("dnsmos_ovrl", "?")
                print(f"  [original] cached (OVRL={ovrl})")
                logger.log("enhance", sid, "original", "score", "cached",
                           scores=orig_data["scores"])

            for pipe_name in pipeline_names:
                pipe_fn = PIPELINES.get(pipe_name)
                if pipe_fn is None:
                    continue

                # Fail-fast: skip disabled pipelines
                if tracker.should_skip(pipe_name):
                    reason = tracker.disabled_reason(pipe_name)
                    print(f"  [{pipe_name}] SKIPPED (fail-fast: {reason})")
                    logger.log("enhance", sid, pipe_name, "skip", "skipped",
                               error_message=reason, error_category="fail_fast")
                    continue

                # Check if already done
                existing = results[sid].get("pipelines", {}).get(pipe_name, {})
                if existing.get("scores", {}).get("dnsmos_ovrl") is not None:
                    print(f"  [{pipe_name}] cached (OVRL={existing['scores']['dnsmos_ovrl']:.2f})")
                    logger.log("enhance", sid, pipe_name, "enhance", "cached",
                               scores=existing["scores"])
                    tracker.record_success(pipe_name)
                    continue

                pipe_dir = ENHANCED_DIR / pipe_name
                pipe_dir.mkdir(parents=True, exist_ok=True)
                enhanced_path = pipe_dir / f"{sid}.wav"

                # Enhance
                if not enhanced_path.exists():
                    print(f"  [{pipe_name}] Enhancing...", end=" ", flush=True)
                    t0 = time.time()
                    try:
                        pipe_fn(str(segment_path), str(enhanced_path))
                        elapsed = time.time() - t0
                        print(f"done ({elapsed:.1f}s)", end=" ")
                        logger.log("enhance", sid, pipe_name, "enhance", "ok",
                                   duration_sec=elapsed)
                    except Exception as e:
                        elapsed = time.time() - t0
                        cat = _classify_error(e)
                        print(f"FAILED: {e}")
                        results[sid].setdefault("pipelines", {})[pipe_name] = {
                            "scores": {"error": str(e)},
                            "processing_time_sec": round(elapsed, 2),
                        }
                        tracker.record_failure(pipe_name, cat)
                        logger.log("enhance", sid, pipe_name, "enhance", "error",
                                   duration_sec=elapsed,
                                   error_message=str(e), error_category=cat)
                        continue
                else:
                    elapsed = 0
                    print(f"  [{pipe_name}] Using cached enhanced...", end=" ")

                if not enhanced_path.exists():
                    print("output not created")
                    results[sid].setdefault("pipelines", {})[pipe_name] = {
                        "scores": {"error": "output file not created"},
                        "processing_time_sec": 0,
                    }
                    tracker.record_failure(pipe_name, "output_missing")
                    logger.log("enhance", sid, pipe_name, "enhance", "error",
                               error_message="output file not created",
                               error_category="output_missing")
                    continue

                # Score
                print("scoring...", end=" ", flush=True)
                t0_score = time.time()
                try:
                    scores = score_segment(str(enhanced_path), metrics=metrics)
                    score_time = time.time() - t0_score
                    results[sid].setdefault("pipelines", {})[pipe_name] = {
                        "scores": scores,
                        "processing_time_sec": round(elapsed, 2),
                    }
                    ovrl = scores.get("dnsmos_ovrl", "?")
                    print(f"done ({score_time:.1f}s) OVRL={ovrl}")
                    tracker.record_success(pipe_name)
                    logger.log("enhance", sid, pipe_name, "score", "ok",
                               duration_sec=score_time, scores=scores)
                except Exception as e:
                    print(f"score FAILED: {e}")
                    results[sid].setdefault("pipelines", {})[pipe_name] = {
                        "scores": {"error": str(e)},
                        "processing_time_sec": round(elapsed, 2),
                    }
                    tracker.record_failure(pipe_name, "score_error")
                    logger.log("enhance", sid, pipe_name, "score", "error",
                               duration_sec=time.time() - t0_score,
                               error_message=str(e), error_category="score_error")

            # Save after each segment (resumability)
            _save_results(results)

        # End-of-run summaries
        print(f"\n{tracker.summary()}")
        logger.print_summary()

    print(f"\nResults saved to {RESULTS_PATH}")


# ── Phase: analyze ───────────────────────────────────────────────────

def cmd_analyze():
    """Run statistical analysis and generate report + charts."""
    results = _load_results()
    if not results:
        print("No results found. Run 'enhance' first.")
        return

    manifest = _load_manifest()

    # Collect data into structured format
    segments, pipeline_names = _collect_analysis_data(results)
    if not segments:
        print("No valid scored segments found.")
        return

    print(f"Analyzing {len(segments)} segments across {len(pipeline_names)} pipelines")

    # Statistical tests
    stats = _run_statistical_tests(segments, pipeline_names)

    # Generate report
    _generate_report(segments, pipeline_names, stats, manifest)

    # Generate Altair charts
    _generate_charts(segments, pipeline_names, stats)

    print(f"\nReport: {REPORT_PATH}")
    print(f"Charts: {CHARTS_DIR}/")


def _collect_analysis_data(
    results: dict, min_coverage: float = 0.8,
) -> tuple[list[dict], list[str]]:
    """Extract structured analysis data from results.

    Only includes pipelines that have scores for at least `min_coverage`
    fraction of segments. This prevents partial pipelines (e.g., run on
    only 30/161 segments) from forcing the statistical tests to drop the
    other 131 segments that lack those pipelines.
    """
    n_total = len(results)

    # Count per-pipeline coverage
    pipe_counts: dict[str, int] = {}
    for sid, data in results.items():
        for pipe_name, pipe_data in data.get("pipelines", {}).items():
            if pipe_data.get("scores", {}).get("dnsmos_ovrl") is not None:
                pipe_counts[pipe_name] = pipe_counts.get(pipe_name, 0) + 1

    # Filter to pipelines with sufficient coverage
    threshold = int(n_total * min_coverage)
    all_pipes = {p for p, c in pipe_counts.items() if c >= threshold}

    pipeline_names = ["original"] + sorted(p for p in all_pipes if p != "original")

    segments = []
    for sid, data in results.items():
        pipelines_data = data.get("pipelines", {})
        # Only include segments that have original + at least 1 enhanced
        if "original" not in pipelines_data:
            continue
        orig_scores = pipelines_data["original"].get("scores", {})
        if orig_scores.get("dnsmos_ovrl") is None:
            continue

        entry = {
            "segment_id": sid,
            "strata": data.get("strata", {}),
            "scores": {},
        }
        for pipe in pipeline_names:
            pipe_data = pipelines_data.get(pipe, {})
            scores = pipe_data.get("scores", {})
            if scores.get("dnsmos_ovrl") is not None:
                entry["scores"][pipe] = scores
        segments.append(entry)

    return segments, pipeline_names


ANALYSIS_METRICS = [
    ("dnsmos_ovrl", "DNSMOS OVRL"),
    ("dnsmos_sig", "DNSMOS SIG"),
    ("dnsmos_bak", "DNSMOS BAK"),
    ("utmos_score", "UTMOS"),
    ("nisqa_mos", "NISQA MOS"),
]

# All sub-metrics for the quality profile table (no full statistical analysis)
_ALL_METRICS = [
    ("dnsmos_ovrl", "OVRL"),
    ("dnsmos_sig", "SIG"),
    ("dnsmos_bak", "BAK"),
    ("dnsmos_p808", "P.808"),
    ("utmos_score", "UTMOS"),
    ("nisqa_mos", "NISQA"),
    ("nisqa_noisiness", "Noise"),
    ("nisqa_coloration", "Color"),
    ("nisqa_discontinuity", "Discont"),
    ("nisqa_loudness", "Loud"),
]


def _bootstrap_ci(data, n_bootstrap: int = 10000, alpha: float = 0.05,
                  seed: int = 42) -> tuple[float, float]:
    """Bootstrap percentile confidence interval for the mean.

    Returns (ci_lo, ci_hi) for the (1-alpha) CI.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(data)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _run_statistical_tests(
    segments: list[dict], pipeline_names: list[str]
) -> dict:
    """Run multi-metric statistical analysis: Friedman + Wilcoxon + CIs."""
    result = {"n_segments": len(segments), "per_metric": {}}

    for metric_key, metric_label in ANALYSIS_METRICS:
        metric_result = _run_tests_for_metric(
            segments, pipeline_names, metric_key, metric_label,
        )
        if metric_result is not None:
            result["per_metric"][metric_key] = metric_result

    # Cross-metric agreement
    result["cross_metric"] = _cross_metric_agreement(segments, pipeline_names)

    # Backward compat: alias primary metric fields at top level
    primary = result["per_metric"].get("dnsmos_ovrl", {})
    result["metric"] = "dnsmos_ovrl"
    result["friedman"] = primary.get("friedman", {})
    result["mean_ranks"] = primary.get("mean_ranks", {})
    result["descriptive"] = primary.get("descriptive", {})
    result["pairwise"] = primary.get("pairwise", {})
    result["alpha_corrected"] = primary.get("alpha_corrected", 0)
    result["per_stratum"] = primary.get("per_stratum", {})
    result["per_format"] = primary.get("per_format", {})

    return result


def _run_tests_for_metric(
    segments: list[dict],
    pipeline_names: list[str],
    metric_key: str,
    metric_label: str,
) -> dict | None:
    """Run Friedman test and post-hoc Wilcoxon for a single metric."""
    import numpy as np
    from scipy import stats as sp_stats
    from scipy.stats import rankdata

    n_pipes = len(pipeline_names)

    # Build score matrix: segments × pipelines (only complete rows)
    score_matrix = []
    valid_segments = []
    for seg in segments:
        row = []
        all_present = True
        for pipe in pipeline_names:
            val = seg["scores"].get(pipe, {}).get(metric_key)
            if val is None:
                all_present = False
                break
            row.append(val)
        if all_present:
            score_matrix.append(row)
            valid_segments.append(seg)

    if len(score_matrix) < 3:
        return None

    matrix = np.array(score_matrix)
    result = {
        "metric_key": metric_key,
        "metric_label": metric_label,
        "n_segments": len(score_matrix),
    }

    # Friedman test
    friedman_stat, friedman_p = sp_stats.friedmanchisquare(
        *[matrix[:, i] for i in range(n_pipes)]
    )
    result["friedman"] = {
        "statistic": float(friedman_stat),
        "p_value": float(friedman_p),
        "significant": friedman_p < 0.05,
    }

    # Mean ranks
    ranks = np.apply_along_axis(rankdata, 1, matrix)
    mean_ranks = ranks.mean(axis=0)
    result["mean_ranks"] = {
        pipe: float(mean_ranks[i]) for i, pipe in enumerate(pipeline_names)
    }

    # Descriptive stats with bootstrap CIs
    result["descriptive"] = {}
    for i, pipe in enumerate(pipeline_names):
        col = matrix[:, i]
        ci_lo, ci_hi = _bootstrap_ci(col)
        result["descriptive"][pipe] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "ci_95": [ci_lo, ci_hi],
        }

    # Post-hoc Wilcoxon signed-rank tests (Bonferroni corrected)
    n_pairs = n_pipes * (n_pipes - 1) // 2
    alpha_corrected = 0.05 / n_pairs if n_pairs > 0 else 0.05
    pairwise = {}

    for (i, pipe_a), (j, pipe_b) in combinations(enumerate(pipeline_names), 2):
        diff = matrix[:, j] - matrix[:, i]
        if np.all(diff == 0):
            pairwise[f"{pipe_a}_vs_{pipe_b}"] = {
                "statistic": 0, "p_value": 1.0,
                "significant": False, "effect_size": 0.0,
            }
            continue

        try:
            stat, p = sp_stats.wilcoxon(matrix[:, i], matrix[:, j])
            n = len(diff[diff != 0])
            r = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0
            pairwise[f"{pipe_a}_vs_{pipe_b}"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "significant": p < alpha_corrected,
                "effect_size": float(r),
            }
        except Exception as e:
            pairwise[f"{pipe_a}_vs_{pipe_b}"] = {"error": str(e)}

    result["pairwise"] = pairwise
    result["alpha_corrected"] = alpha_corrected

    # Per-stratum analysis
    result["per_stratum"] = _per_stratum_analysis(
        valid_segments, matrix, pipeline_names
    )

    # Per-format (content type) analysis
    result["per_format"] = _per_format_analysis(
        valid_segments, matrix, pipeline_names
    )

    return result


def _per_stratum_analysis(
    segments: list[dict],
    matrix,
    pipeline_names: list[str],
) -> dict:
    """Run Friedman test per series_group stratum."""
    import numpy as np
    from scipy import stats as sp_stats

    strata_results = {}
    groups: dict[str, list[int]] = {}
    for idx, seg in enumerate(segments):
        sg = seg["strata"].get("series_group", "unknown")
        groups.setdefault(sg, []).append(idx)

    for sg, indices in sorted(groups.items()):
        sub_matrix = matrix[indices]
        n = len(indices)
        if n < 3:
            strata_results[sg] = {
                "n": n,
                "note": "too few samples for Friedman test",
                "means": {
                    pipe: float(np.mean(sub_matrix[:, i]))
                    for i, pipe in enumerate(pipeline_names)
                },
            }
            continue

        try:
            stat, p = sp_stats.friedmanchisquare(
                *[sub_matrix[:, i] for i in range(len(pipeline_names))]
            )
            strata_results[sg] = {
                "n": n,
                "friedman_statistic": float(stat),
                "friedman_p": float(p),
                "significant": p < 0.05,
                "means": {
                    pipe: float(np.mean(sub_matrix[:, i]))
                    for i, pipe in enumerate(pipeline_names)
                },
            }
        except Exception as e:
            strata_results[sg] = {"n": n, "error": str(e)}

    return strata_results


def _per_format_analysis(
    segments: list[dict],
    matrix,
    pipeline_names: list[str],
) -> dict:
    """Run Friedman test per format_group (content type) stratum.

    Also computes best_pipeline per format group (highest mean OVRL
    excluding original) to support content-type-aware pipeline selection.
    """
    import numpy as np
    from scipy import stats as sp_stats

    format_results = {}
    groups: dict[str, list[int]] = {}
    for idx, seg in enumerate(segments):
        fg = seg["strata"].get("format_group", "lecture")
        groups.setdefault(fg, []).append(idx)

    enhanced_indices = [
        i for i, p in enumerate(pipeline_names) if p != "original"
    ]

    for fg, indices in sorted(groups.items()):
        sub_matrix = matrix[indices]
        n = len(indices)
        means = {
            pipe: float(np.mean(sub_matrix[:, i]))
            for i, pipe in enumerate(pipeline_names)
        }

        # Best pipeline (highest mean, excluding original)
        if enhanced_indices:
            best_idx = max(
                enhanced_indices, key=lambda i: float(np.mean(sub_matrix[:, i]))
            )
            best_pipeline = pipeline_names[best_idx]
            best_mean = float(np.mean(sub_matrix[:, best_idx]))
        else:
            best_pipeline = ""
            best_mean = 0.0

        if n < 3:
            format_results[fg] = {
                "n": n,
                "note": "too few samples for Friedman test",
                "means": means,
                "best_pipeline": best_pipeline,
                "best_mean": best_mean,
            }
            continue

        try:
            stat, p = sp_stats.friedmanchisquare(
                *[sub_matrix[:, i] for i in range(len(pipeline_names))]
            )
            format_results[fg] = {
                "n": n,
                "friedman_statistic": float(stat),
                "friedman_p": float(p),
                "significant": p < 0.05,
                "means": means,
                "best_pipeline": best_pipeline,
                "best_mean": best_mean,
            }
        except Exception as e:
            format_results[fg] = {
                "n": n,
                "error": str(e),
                "means": means,
                "best_pipeline": best_pipeline,
                "best_mean": best_mean,
            }

    return format_results


def _cross_metric_agreement(
    segments: list[dict], pipeline_names: list[str]
) -> dict:
    """Compute Spearman correlations between improvement deltas across metrics.

    For each segment, compute improvement delta (enhanced - original) for each
    metric. Then correlate these deltas across metric pairs. Weak agreement
    (|rho| < 0.4) flags potential metric disagreement.
    """
    import numpy as np
    from scipy import stats as sp_stats

    metric_keys = [m[0] for m in ANALYSIS_METRICS]
    enhanced_pipes = [p for p in pipeline_names if p != "original"]

    # Collect per-metric improvement deltas: {metric: list of deltas}
    metric_deltas: dict[str, list[float]] = {m: [] for m in metric_keys}

    for seg in segments:
        orig_scores = seg["scores"].get("original", {})
        for pipe in enhanced_pipes:
            pipe_scores = seg["scores"].get(pipe, {})
            for mk in metric_keys:
                orig_val = orig_scores.get(mk)
                pipe_val = pipe_scores.get(mk)
                if orig_val is not None and pipe_val is not None:
                    metric_deltas[mk].append(pipe_val - orig_val)
                else:
                    metric_deltas[mk].append(None)

    # Filter to metrics with enough data
    available = {mk: vals for mk, vals in metric_deltas.items()
                 if sum(1 for v in vals if v is not None) >= 5}

    results = {}
    avail_keys = sorted(available.keys())
    for i, mk_a in enumerate(avail_keys):
        for mk_b in avail_keys[i + 1:]:
            vals_a = available[mk_a]
            vals_b = available[mk_b]
            # Pair only where both are non-None
            paired = [(a, b) for a, b in zip(vals_a, vals_b)
                      if a is not None and b is not None]
            if len(paired) < 5:
                continue
            arr_a = np.array([p[0] for p in paired])
            arr_b = np.array([p[1] for p in paired])
            rho, p_val = sp_stats.spearmanr(arr_a, arr_b)
            label_a = next(l for k, l in ANALYSIS_METRICS if k == mk_a)
            label_b = next(l for k, l in ANALYSIS_METRICS if k == mk_b)
            results[f"{mk_a}_vs_{mk_b}"] = {
                "labels": f"{label_a} vs {label_b}",
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "n_pairs": len(paired),
                "weak_agreement": abs(rho) < 0.4,
            }

    return results


def cmd_sensitivity(target_n: int = 40, seeds: list[int] | None = None):
    """Multi-seed sensitivity analysis for stratified sampling.

    Runs stratified_sample() with multiple seeds and reports:
    - Pairwise Jaccard similarity of selected event sets
    - Core overlap (events selected in all seeds)
    """
    if seeds is None:
        seeds = [42, 123, 7, 2024, 999]

    print(f"Loading events from {EVENTS_DIR}...")
    events = load_all_events(EVENTS_DIR)
    print(f"  Found {len(events)} events\n")

    print(f"Running stratified_sample(target_n={target_n}) with {len(seeds)} seeds...")
    seed_sets: dict[int, set[str]] = {}
    for seed in seeds:
        manifest = stratified_sample(events, target_n=target_n, seed=seed)
        sids = {e["segment_id"] for e in manifest}
        seed_sets[seed] = sids
        print(f"  seed={seed}: {len(sids)} samples")

    # Pairwise Jaccard similarity
    print(f"\nPairwise Jaccard similarity:")
    print(f"  {'':>8}", end="")
    for s in seeds:
        print(f"  {s:>6}", end="")
    print()

    for i, sa in enumerate(seeds):
        print(f"  {sa:>8}", end="")
        for j, sb in enumerate(seeds):
            if j <= i:
                set_a, set_b = seed_sets[sa], seed_sets[sb]
                jaccard = len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0
                print(f"  {jaccard:>6.2f}", end="")
            else:
                print(f"  {'':>6}", end="")
        print()

    # Core overlap
    core = set.intersection(*seed_sets.values()) if seed_sets else set()
    all_seen = set.union(*seed_sets.values()) if seed_sets else set()
    print(f"\nCore overlap (in ALL {len(seeds)} seeds): {len(core)}/{target_n} "
          f"({len(core)/target_n:.0%})")
    print(f"Total unique events seen: {len(all_seen)}")

    if core:
        print(f"Core events: {sorted(core)[:10]}{'...' if len(core) > 10 else ''}")


def _report_quality_profile(
    segments: list[dict], pipeline_names: list[str],
) -> list[str]:
    """Generate compact all-metrics summary table."""
    import numpy as np

    lines = ["## Pipeline Quality Profile\n"]
    lines.append("Mean scores across all sub-metrics (higher = better for all except Noise).\n")

    # Header
    header = "| Pipeline |"
    sep = "|---|"
    for _, label in _ALL_METRICS:
        header += f" {label} |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    # Rows
    for pipe in pipeline_names:
        row = f"| {pipe} |"
        for mk, _ in _ALL_METRICS:
            vals = [
                seg["scores"].get(pipe, {}).get(mk)
                for seg in segments
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                row += f" {np.mean(vals):.2f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    lines.append("_OVRL=Overall, SIG=Signal quality, BAK=Background noise, "
                 "P.808=ITU-T P.808, NISQA=MOS, Noise=Noisiness, "
                 "Color=Coloration, Discont=Discontinuity, Loud=Loudness_\n")
    return lines


def _report_improvement_deltas(
    segments: list[dict], pipeline_names: list[str],
) -> list[str]:
    """Generate improvement delta table (each pipeline vs original)."""
    import numpy as np

    enhanced = [p for p in pipeline_names if p != "original"]
    if not enhanced:
        return []

    lines = ["## Improvement Over Original\n"]
    lines.append("Mean improvement delta (pipeline − original). "
                 "Positive = better.\n")

    # Header
    header = "| Pipeline |"
    sep = "|---|"
    for _, label in _ALL_METRICS:
        header += f" Δ{label} |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for pipe in enhanced:
        row = f"| {pipe} |"
        for mk, _ in _ALL_METRICS:
            deltas = []
            for seg in segments:
                orig_val = seg["scores"].get("original", {}).get(mk)
                pipe_val = seg["scores"].get(pipe, {}).get(mk)
                if orig_val is not None and pipe_val is not None:
                    deltas.append(pipe_val - orig_val)
            if deltas:
                mean_d = np.mean(deltas)
                sign = "+" if mean_d >= 0 else ""
                row += f" {sign}{mean_d:.2f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    return lines


def _generate_report(
    segments: list[dict],
    pipeline_names: list[str],
    stats: dict,
    manifest: list[dict],
):
    """Generate Markdown benchmark report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Audio Enhancement Benchmark Report\n"]
    lines.append(f"**Samples**: {stats.get('n_segments', len(segments))}")
    lines.append(f"**Pipelines**: {len(pipeline_names)}")
    n_metrics = len(stats.get("per_metric", {}))
    lines.append(f"**Metrics analyzed**: {n_metrics}\n")

    # Sample distribution
    lines.append("## Sample Distribution\n")
    from collections import Counter
    if manifest:
        groups = Counter(e["strata"]["series_group"] for e in manifest)
        eras = Counter(e["strata"]["era"] for e in manifest)
        lines.append("| Series Group | Count |")
        lines.append("|---|---|")
        for sg, count in sorted(groups.items()):
            lines.append(f"| {sg} | {count} |")
        lines.append("")
        lines.append("| Era | Count |")
        lines.append("|---|---|")
        for era, count in sorted(eras.items()):
            lines.append(f"| {era} | {count} |")
        lines.append("")

    # Pipeline Quality Profile — compact all-metrics overview
    lines.extend(_report_quality_profile(segments, pipeline_names))

    # Improvement over original
    lines.extend(_report_improvement_deltas(segments, pipeline_names))

    # Per-metric analysis sections
    per_metric = stats.get("per_metric", {})
    for metric_key, metric_label in ANALYSIS_METRICS:
        mdata = per_metric.get(metric_key)
        if mdata is None:
            continue

        lines.append(f"## {metric_label} Analysis (n={mdata['n_segments']})\n")

        # Descriptive stats with CIs
        lines.append(f"### Pipeline Scores — {metric_label}\n")
        lines.append("| Pipeline | Mean [95% CI] | Median | Std | Min | Max | Mean Rank |")
        lines.append("|---|---|---|---|---|---|---|")
        desc = mdata.get("descriptive", {})
        ranks = mdata.get("mean_ranks", {})
        for pipe in pipeline_names:
            d = desc.get(pipe, {})
            r = ranks.get(pipe, 0)
            ci = d.get("ci_95", [0, 0])
            lines.append(
                f"| {pipe} | {d.get('mean', 0):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{d.get('median', 0):.3f} | {d.get('std', 0):.3f} | "
                f"{d.get('min', 0):.3f} | {d.get('max', 0):.3f} | {r:.2f} |"
            )
        lines.append("")

        # Friedman test
        friedman = mdata.get("friedman", {})
        lines.append(f"### Friedman Test — {metric_label}\n")
        lines.append(f"- **Statistic**: {friedman.get('statistic', 0):.3f}")
        lines.append(f"- **p-value**: {friedman.get('p_value', 1):.6f}")
        sig = "Yes" if friedman.get("significant") else "No"
        lines.append(f"- **Significant** (alpha=0.05): {sig}\n")

        # Pairwise comparisons
        pairwise = mdata.get("pairwise", {})
        alpha = mdata.get("alpha_corrected", 0.05)
        if pairwise:
            lines.append(f"### Pairwise Wilcoxon — {metric_label} "
                         f"(Bonferroni alpha={alpha:.4f})\n")
            lines.append("| Comparison | Statistic | p-value | Significant | Effect Size (r) |")
            lines.append("|---|---|---|---|---|")
            for pair_key, pair_data in sorted(pairwise.items()):
                if "error" in pair_data:
                    lines.append(f"| {pair_key} | — | — | error | — |")
                    continue
                sig = "**Yes**" if pair_data.get("significant") else "No"
                r_val = pair_data.get("effect_size", 0)
                practical = " *" if abs(r_val) > 0.3 else ""
                lines.append(
                    f"| {pair_key} | {pair_data.get('statistic', 0):.1f} | "
                    f"{pair_data.get('p_value', 1):.6f} | {sig} | "
                    f"{r_val:.3f}{practical} |"
                )
            lines.append("\n_* |r| > 0.3 = practically significant_\n")

    # Cross-metric agreement
    cross = stats.get("cross_metric", {})
    if cross:
        lines.append("## Cross-Metric Agreement\n")
        lines.append("Spearman correlation of improvement deltas between metric pairs.\n")
        lines.append("| Metric Pair | Spearman rho | p-value | n | Agreement |")
        lines.append("|---|---|---|---|---|")
        for pair_key, pair_data in sorted(cross.items()):
            rho = pair_data.get("spearman_rho", 0)
            weak = pair_data.get("weak_agreement", False)
            agreement = "**WEAK**" if weak else "OK"
            lines.append(
                f"| {pair_data.get('labels', pair_key)} | "
                f"{rho:.3f} | {pair_data.get('p_value', 1):.4f} | "
                f"{pair_data.get('n_pairs', 0)} | {agreement} |"
            )
        lines.append("\n_Weak agreement (|rho| < 0.4) suggests metrics may disagree "
                      "on which pipelines improve quality._\n")

    # Per-stratum results (from primary metric)
    per_stratum = stats.get("per_stratum", {})
    if per_stratum:
        lines.append("## Per-Stratum Analysis (DNSMOS OVRL)\n")
        for sg, sg_data in sorted(per_stratum.items()):
            n = sg_data.get("n", 0)
            lines.append(f"### {sg} (n={n})\n")
            if "error" in sg_data or "note" in sg_data:
                lines.append(f"_{sg_data.get('note', sg_data.get('error', ''))}_\n")
            else:
                fp = sg_data.get("friedman_p", 1)
                sig = "Yes" if sg_data.get("significant") else "No"
                lines.append(f"Friedman p={fp:.4f} (significant: {sig})\n")
            means = sg_data.get("means", {})
            if means:
                lines.append("| Pipeline | Mean OVRL |")
                lines.append("|---|---|")
                for pipe in pipeline_names:
                    if pipe in means:
                        lines.append(f"| {pipe} | {means[pipe]:.3f} |")
                lines.append("")

    # Per-format (content type) results
    per_format = stats.get("per_format", {})
    if per_format:
        lines.append("## Per-Content-Type Analysis (DNSMOS OVRL)\n")
        lines.append("| Content Type | N | Best Pipeline | Mean OVRL | Recommended | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for fg, fg_data in sorted(per_format.items()):
            n = fg_data.get("n", 0)
            best = fg_data.get("best_pipeline", "")
            best_mean = fg_data.get("best_mean", 0)
            recommended = FORMAT_PIPELINE_MAP.get(fg, "hybrid_demucs_df")
            note = ""
            if "note" in fg_data:
                note = fg_data["note"]
            elif fg_data.get("significant"):
                note = f"Friedman p={fg_data.get('friedman_p', 1):.4f}"
            lines.append(
                f"| {fg} | {n} | {best} | {best_mean:.3f} | "
                f"{recommended} | {note} |"
            )
        lines.append("")

        for fg, fg_data in sorted(per_format.items()):
            n = fg_data.get("n", 0)
            lines.append(f"### {fg} (n={n})\n")
            if "error" in fg_data or "note" in fg_data:
                lines.append(f"_{fg_data.get('note', fg_data.get('error', ''))}_\n")
            else:
                fp = fg_data.get("friedman_p", 1)
                sig = "Yes" if fg_data.get("significant") else "No"
                lines.append(f"Friedman p={fp:.4f} (significant: {sig})\n")
            means = fg_data.get("means", {})
            if means:
                lines.append("| Pipeline | Mean OVRL |")
                lines.append("|---|---|")
                for pipe in pipeline_names:
                    if pipe in means:
                        lines.append(f"| {pipe} | {means[pipe]:.3f} |")
                lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"  Report saved to {REPORT_PATH}")


def _build_charts(
    segments: list[dict],
    pipeline_names: list[str],
    stats: dict,
) -> dict:
    """Build Altair chart objects for benchmark visualization.

    Returns dict mapping chart name to alt.Chart (or layered chart).
    Callers can save as HTML (.save("x.html")) or PNG (.save("x.png", ppi=150)).
    """
    import altair as alt
    import pandas as pd

    charts = {}

    # Build long-form DataFrame with all metrics
    rows = []
    for seg in segments:
        sid = seg["segment_id"]
        strata = seg["strata"]
        for pipe, scores in seg["scores"].items():
            ovrl = scores.get("dnsmos_ovrl")
            if ovrl is None:
                continue
            rows.append({
                "segment_id": sid,
                "pipeline": pipe,
                "dnsmos_ovrl": ovrl,
                "dnsmos_sig": scores.get("dnsmos_sig"),
                "dnsmos_bak": scores.get("dnsmos_bak"),
                "utmos_score": scores.get("utmos_score"),
                "nisqa_mos": scores.get("nisqa_mos"),
                "nisqa_noisiness": scores.get("nisqa_noisiness"),
                "nisqa_coloration": scores.get("nisqa_coloration"),
                "nisqa_discontinuity": scores.get("nisqa_discontinuity"),
                "nisqa_loudness": scores.get("nisqa_loudness"),
                "series_group": strata.get("series_group", ""),
                "format_group": strata.get("format_group", ""),
                "era": strata.get("era", ""),
            })

    if not rows:
        return charts

    df = pd.DataFrame(rows)

    # 1. DNSMOS OVRL boxplot
    charts["pipeline_boxplot"] = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("pipeline:N", sort=pipeline_names, title="Pipeline"),
            y=alt.Y("dnsmos_ovrl:Q", title="DNSMOS OVRL", scale=alt.Scale(zero=False)),
            color=alt.Color("pipeline:N", legend=None),
        )
        .properties(title="DNSMOS OVRL Distribution by Pipeline", width=600, height=350)
    )

    # 1b. UTMOS boxplot (if data available)
    df_utmos = df.dropna(subset=["utmos_score"])
    if len(df_utmos) > 0:
        charts["utmos_boxplot"] = (
            alt.Chart(df_utmos)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("pipeline:N", sort=pipeline_names, title="Pipeline"),
                y=alt.Y("utmos_score:Q", title="UTMOS", scale=alt.Scale(zero=False)),
                color=alt.Color("pipeline:N", legend=None),
            )
            .properties(title="UTMOS Distribution by Pipeline", width=600, height=350)
        )

    # 2. Series heatmap (pipeline x series_group, mean improvement)
    orig_scores = df[df["pipeline"] == "original"][["segment_id", "dnsmos_ovrl"]].rename(
        columns={"dnsmos_ovrl": "original_ovrl"}
    )
    df_merged = df.merge(orig_scores, on="segment_id", how="left")
    df_merged["improvement"] = df_merged["dnsmos_ovrl"] - df_merged["original_ovrl"]

    heatmap_data = (
        df_merged[df_merged["pipeline"] != "original"]
        .groupby(["pipeline", "series_group"])["improvement"]
        .mean()
        .reset_index()
    )

    if not heatmap_data.empty:
        charts["series_heatmap"] = (
            alt.Chart(heatmap_data)
            .mark_rect()
            .encode(
                x=alt.X("series_group:N", title="Series Group"),
                y=alt.Y("pipeline:N", title="Pipeline"),
                color=alt.Color(
                    "improvement:Q",
                    title="Mean OVRL Improvement",
                    scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                ),
                tooltip=["pipeline", "series_group",
                          alt.Tooltip("improvement:Q", format=".3f")],
            )
            .properties(
                title="Mean DNSMOS OVRL Improvement over Original",
                width=450, height=300,
            )
        )

    # 2b. Format heatmap (pipeline x format_group, mean improvement)
    format_heatmap_data = (
        df_merged[df_merged["pipeline"] != "original"]
        .groupby(["pipeline", "format_group"])["improvement"]
        .mean()
        .reset_index()
    )

    if not format_heatmap_data.empty:
        charts["format_heatmap"] = (
            alt.Chart(format_heatmap_data)
            .mark_rect()
            .encode(
                x=alt.X("format_group:N", title="Content Type"),
                y=alt.Y("pipeline:N", title="Pipeline"),
                color=alt.Color(
                    "improvement:Q",
                    title="Mean OVRL Improvement",
                    scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                ),
                tooltip=["pipeline", "format_group",
                          alt.Tooltip("improvement:Q", format=".3f")],
            )
            .properties(
                title="Mean DNSMOS OVRL Improvement by Content Type",
                width=450, height=300,
            )
        )

    # 3. Baseline quality by era
    orig_df = df[df["pipeline"] == "original"]
    if not orig_df.empty:
        charts["baseline_histogram"] = (
            alt.Chart(orig_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("dnsmos_ovrl:Q", bin=alt.Bin(maxbins=15),
                         title="DNSMOS OVRL (Original)"),
                y=alt.Y("count()", title="Count"),
                color=alt.Color("era:N", title="Era"),
            )
            .properties(title="Baseline Audio Quality by Era", width=500, height=250)
        )

    # 4. Pairwise significance matrix
    pairwise = stats.get("pairwise", {})
    if pairwise:
        pw_rows = []
        for pair_key, pair_data in pairwise.items():
            parts = pair_key.split("_vs_")
            if len(parts) != 2:
                continue
            p_val = pair_data.get("p_value", 1.0)
            pw_rows.append({"pipe_a": parts[0], "pipe_b": parts[1], "p_value": p_val})
            pw_rows.append({"pipe_a": parts[1], "pipe_b": parts[0], "p_value": p_val})

        if pw_rows:
            pw_df = pd.DataFrame(pw_rows)
            charts["pairwise_significance"] = (
                alt.Chart(pw_df)
                .mark_rect()
                .encode(
                    x=alt.X("pipe_a:N", title="Pipeline A", sort=pipeline_names),
                    y=alt.Y("pipe_b:N", title="Pipeline B", sort=pipeline_names),
                    color=alt.Color(
                        "p_value:Q",
                        title="Wilcoxon p-value",
                        scale=alt.Scale(
                            scheme="redyellowgreen",
                            reverse=True,
                            type="log",
                            clamp=True,
                        ),
                    ),
                    tooltip=["pipe_a", "pipe_b",
                              alt.Tooltip("p_value:Q", format=".6f")],
                )
                .properties(
                    title="Pairwise Wilcoxon p-values",
                    width=400, height=400,
                )
            )

    # 5. Cross-metric scatter (DNSMOS OVRL vs UTMOS)
    df_cross = df.dropna(subset=["utmos_score"])
    if len(df_cross) > 0:
        points = (
            alt.Chart(df_cross)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("dnsmos_ovrl:Q", title="DNSMOS OVRL",
                         scale=alt.Scale(zero=False)),
                y=alt.Y("utmos_score:Q", title="UTMOS",
                         scale=alt.Scale(zero=False)),
                color=alt.Color("pipeline:N", sort=pipeline_names,
                                legend=alt.Legend(title="Pipeline")),
                tooltip=["segment_id", "pipeline",
                          alt.Tooltip("dnsmos_ovrl:Q", format=".3f"),
                          alt.Tooltip("utmos_score:Q", format=".3f")],
            )
        )
        regression = points.transform_regression(
            "dnsmos_ovrl", "utmos_score",
        ).mark_line(strokeDash=[4, 4], color="gray")

        charts["cross_metric_scatter"] = (points + regression).properties(
            title="Cross-Metric: DNSMOS OVRL vs UTMOS",
            width=500, height=400,
        )

    # 6. SIG vs BAK tradeoff scatter (speech quality vs noise suppression)
    df_sigbak = df.dropna(subset=["dnsmos_sig", "dnsmos_bak"])
    if len(df_sigbak) > 0:
        charts["sig_bak_tradeoff"] = (
            alt.Chart(df_sigbak)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("dnsmos_bak:Q", title="Background Noise (BAK) →",
                         scale=alt.Scale(zero=False)),
                y=alt.Y("dnsmos_sig:Q", title="Signal Quality (SIG) →",
                         scale=alt.Scale(zero=False)),
                color=alt.Color("pipeline:N", sort=pipeline_names,
                                legend=alt.Legend(title="Pipeline")),
                tooltip=["segment_id", "pipeline",
                          alt.Tooltip("dnsmos_sig:Q", format=".2f"),
                          alt.Tooltip("dnsmos_bak:Q", format=".2f"),
                          alt.Tooltip("dnsmos_ovrl:Q", format=".2f")],
            )
            .properties(
                title="Signal Quality vs Background Noise Tradeoff",
                width=500, height=400,
            )
        )

    # 7. NISQA sub-dimensions radar-style grouped bar chart
    nisqa_cols = ["nisqa_noisiness", "nisqa_coloration",
                  "nisqa_discontinuity", "nisqa_loudness"]
    df_nisqa = df.dropna(subset=nisqa_cols)
    if len(df_nisqa) > 0:
        nisqa_means = (
            df_nisqa.groupby("pipeline")[nisqa_cols]
            .mean()
            .reset_index()
            .melt(id_vars="pipeline", var_name="dimension", value_name="score")
        )
        nisqa_means["dimension"] = nisqa_means["dimension"].str.replace("nisqa_", "")
        charts["nisqa_subdimensions"] = (
            alt.Chart(nisqa_means)
            .mark_bar()
            .encode(
                x=alt.X("dimension:N", title=None),
                y=alt.Y("score:Q", title="NISQA Score"),
                color=alt.Color("pipeline:N", sort=pipeline_names,
                                legend=alt.Legend(title="Pipeline")),
                xOffset="pipeline:N",
            )
            .properties(
                title="NISQA Sub-dimensions by Pipeline",
                width=500, height=350,
            )
        )

    # 8. CI forest plot (point + error bar per pipeline)
    per_metric = stats.get("per_metric", {})
    ci_rows = []
    for metric_key, metric_label in ANALYSIS_METRICS:
        mdata = per_metric.get(metric_key, {})
        desc = mdata.get("descriptive", {})
        for pipe in pipeline_names:
            d = desc.get(pipe)
            if d is None or "ci_95" not in d:
                continue
            ci_rows.append({
                "pipeline": pipe,
                "metric": metric_label,
                "mean": d["mean"],
                "ci_lo": d["ci_95"][0],
                "ci_hi": d["ci_95"][1],
            })

    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        points = (
            alt.Chart(ci_df)
            .mark_point(filled=True, size=60)
            .encode(
                x=alt.X("mean:Q", title="Score", scale=alt.Scale(zero=False)),
                y=alt.Y("pipeline:N", sort=pipeline_names, title=None),
                color=alt.Color("metric:N", title="Metric"),
            )
        )
        errorbars = (
            alt.Chart(ci_df)
            .mark_errorbar()
            .encode(
                x=alt.X("ci_lo:Q", title="Score"),
                x2="ci_hi:Q",
                y=alt.Y("pipeline:N", sort=pipeline_names),
                color=alt.Color("metric:N", title="Metric"),
            )
        )
        charts["ci_forest_plot"] = (errorbars + points).properties(
            title="Pipeline Means with 95% Bootstrap CI",
            width=500, height=350,
        )

    return charts


def _generate_charts(
    segments: list[dict],
    pipeline_names: list[str],
    stats: dict,
):
    """Generate Altair HTML charts."""
    try:
        import altair as alt  # noqa: F401
        import pandas as pd  # noqa: F401
    except ImportError:
        print("  [WARN] altair/pandas not available, skipping charts")
        return

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    charts = _build_charts(segments, pipeline_names, stats)
    if not charts:
        print("  No data for charts")
        return

    for name, chart in charts.items():
        chart.save(str(CHARTS_DIR / f"{name}.html"))
        print(f"  Saved {name}.html")


# ── Phase: export ───────────────────────────────────────────────────

def _select_representative_segments(
    segments: list[dict], n: int = 3,
) -> list[str]:
    """Pick representative segments for audio comparison.

    Selects segments with diversity across original quality:
    lowest, highest, and median DNSMOS OVRL baselines.
    """
    scored = []
    for seg in segments:
        ovrl = seg["scores"].get("original", {}).get("dnsmos_ovrl")
        if ovrl is not None:
            scored.append((seg["segment_id"], ovrl))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1])

    if len(scored) <= n:
        return [s[0] for s in scored]

    # Pick lowest, highest, median
    selected = [scored[0][0], scored[-1][0]]
    mid_idx = len(scored) // 2
    if scored[mid_idx][0] not in selected:
        selected.insert(1, scored[mid_idx][0])

    # Fill remaining slots if n > 3
    for s in scored:
        if len(selected) >= n:
            break
        if s[0] not in selected:
            selected.append(s[0])

    return selected[:n]


def _select_preview_segments(
    segments: list[dict],
    n: int = 20,
    include_sids: list[str] | None = None,
) -> list[str]:
    """Select diverse segments for audio preview across quality and content.

    Strategy: include anchor segments first (e.g. from export), then fill
    remaining slots with evenly spaced samples across the quality range,
    swapping duplicates to ensure series_group diversity.
    """
    scored = []
    for seg in segments:
        ovrl = seg["scores"].get("original", {}).get("dnsmos_ovrl")
        if ovrl is not None:
            scored.append({
                "sid": seg["segment_id"],
                "ovrl": ovrl,
                "series_group": seg.get("strata", {}).get("series_group", "other"),
                "era": seg.get("strata", {}).get("era", ""),
            })

    if not scored:
        return []

    scored.sort(key=lambda x: x["ovrl"])

    # Start with anchor segments (from export)
    used_sids: set[str] = set()
    if include_sids:
        for sid in include_sids:
            if any(s["sid"] == sid for s in scored):
                used_sids.add(sid)

    if len(scored) <= n:
        return [s["sid"] for s in scored]

    # Fill remaining slots with evenly spaced picks
    remaining = n - len(used_sids)
    available = [s for s in scored if s["sid"] not in used_sids]
    if remaining > 0 and available:
        step = (len(available) - 1) / max(remaining - 1, 1)
        indices = [round(i * step) for i in range(remaining)]
        selected = [available[i] for i in indices]
    else:
        selected = []

    # Ensure series_group diversity: swap duplicates with nearest unused
    seen_groups: set[str] = set()
    # Count groups from anchor segments
    scored_by_sid = {s["sid"]: s for s in scored}
    for sid in used_sids:
        if sid in scored_by_sid:
            seen_groups.add(scored_by_sid[sid]["series_group"])

    for sid in [s["sid"] for s in selected]:
        used_sids.add(sid)

    for i, entry in enumerate(selected):
        group = entry["series_group"]
        if group not in seen_groups:
            seen_groups.add(group)
            continue
        # Find nearest unused segment with a different group
        idx = scored.index(entry)
        for offset in range(1, len(scored)):
            for candidate_idx in [idx + offset, idx - offset]:
                if candidate_idx < 0 or candidate_idx >= len(scored):
                    continue
                candidate = scored[candidate_idx]
                if (candidate["sid"] not in used_sids
                        and candidate["series_group"] not in seen_groups):
                    used_sids.discard(entry["sid"])
                    used_sids.add(candidate["sid"])
                    selected[i] = candidate
                    seen_groups.add(candidate["series_group"])
                    break
            else:
                continue
            break
        else:
            seen_groups.add(group)

    # Combine anchors + selected, sort by OVRL
    all_sids = used_sids
    final = [s for s in scored if s["sid"] in all_sids]
    final.sort(key=lambda x: x["ovrl"])
    return [s["sid"] for s in final]


def _export_audio_samples(
    segment_ids: list[str],
    pipeline_names: list[str],
    output_dir: Path,
    bitrate: str = "192k",
):
    """Encode representative segments as MP3 for the export report."""
    audio_dir = output_dir / "audio"
    total = 0

    for sid in segment_ids:
        sid_dir = audio_dir / sid
        sid_dir.mkdir(parents=True, exist_ok=True)

        # Original
        orig_wav = SEGMENTS_DIR / f"{sid}.wav"
        if orig_wav.exists():
            mp3_path = sid_dir / "original.mp3"
            if not mp3_path.exists():
                print(f"  {sid}/original.mp3...", end=" ", flush=True)
                encode_mp3(str(orig_wav), str(mp3_path), bitrate=bitrate)
                print("done")
            total += 1

        # Enhanced pipelines
        for pipe in pipeline_names:
            if pipe == "original":
                continue
            enhanced_wav = ENHANCED_DIR / pipe / f"{sid}.wav"
            mp3_path = sid_dir / f"{pipe}.mp3"
            if enhanced_wav.exists() and not mp3_path.exists():
                print(f"  {sid}/{pipe}.mp3...", end=" ", flush=True)
                encode_mp3(str(enhanced_wav), str(mp3_path), bitrate=bitrate)
                print("done")
            if mp3_path.exists():
                total += 1

    print(f"  Encoded {total} audio files")


def _generate_export_markdown(
    segments: list[dict],
    pipeline_names: list[str],
    stats: dict,
    manifest: list[dict],
    representative_sids: list[str],
    output_dir: Path,
):
    """Generate README.md for the export report directory."""
    import numpy as np

    lines = ["# Audio Enhancement Benchmark Report\n"]

    # Overview
    lines.append("## Overview\n")
    n_seg = stats.get("n_segments", len(segments))
    n_pipes = len(pipeline_names)
    now = datetime.now().strftime("%Y-%m-%d")
    lines.append(
        f"Benchmark of **{n_pipes} audio enhancement pipelines** across "
        f"**{n_seg} stratified segments** (45-second speech-active excerpts "
        f"selected via Silero VAD) from 429 YouTube recordings of "
        f"The Reading Room BKK (2011-2019).\n"
    )
    lines.append(f"Generated: {now}\n")

    # Quality profile table
    lines.extend(_report_quality_profile(segments, pipeline_names))

    # Improvement deltas table
    lines.extend(_report_improvement_deltas(segments, pipeline_names))

    # Charts — reference the 6 key charts from the plan
    chart_sections = [
        ("pipeline_boxplot", "Score Distribution",
         "Distribution of DNSMOS OVRL scores across all segments for each pipeline."),
        ("sig_bak_tradeoff", "Signal vs Background Tradeoff",
         "Each point is one segment. Upper-right corner = best (high signal quality + high background suppression)."),
        ("nisqa_subdimensions", "NISQA Sub-dimensions",
         "NISQA decomposes speech quality into noisiness, coloration, discontinuity, and loudness."),
        ("cross_metric_scatter", "Cross-Metric Correlation",
         "DNSMOS OVRL vs UTMOS — assessing agreement between two independent quality metrics."),
        ("series_heatmap", "Pipeline Scores by Series Group",
         "Mean DNSMOS OVRL improvement over original, broken down by content series."),
        ("format_heatmap", "Pipeline Scores by Content Type",
         "Mean DNSMOS OVRL improvement over original, broken down by content type (lecture, screening, performance, etc.)."),
        ("ci_forest_plot", "Confidence Intervals",
         "Pipeline means with 95% bootstrap confidence intervals across all metrics."),
    ]

    for chart_name, title, description in chart_sections:
        lines.append(f"## {title}\n")
        lines.append(f"{description}\n")
        lines.append(f"![{title}](images/{chart_name}.png)\n")

    # Audio comparison
    lines.append("## Audio Comparison\n")
    lines.append(
        "Representative segments selected for diversity: "
        "lowest, median, and highest original DNSMOS OVRL.\n"
    )

    # Build lookup for segment metadata
    seg_by_id = {s["segment_id"]: s for s in segments}
    manifest_by_sid = {}
    for entry in manifest:
        manifest_by_sid[entry["segment_id"]] = entry

    for sid in representative_sids:
        seg = seg_by_id.get(sid)
        if not seg:
            continue

        strata = seg.get("strata", {})
        series_group = strata.get("series_group", "")
        era = strata.get("era", "")
        orig_ovrl = seg["scores"].get("original", {}).get("dnsmos_ovrl", 0)

        # Use event title + YouTube link if available
        m_entry = manifest_by_sid.get(sid, {})
        event_title = m_entry.get("event_key", sid)
        video_id = m_entry.get("video_id", "")
        if video_id:
            yt_url = f"https://www.youtube.com/watch?v={video_id}"
            title_part = f"[{event_title}]({yt_url})"
        else:
            title_part = f"`{sid}`"

        lines.append(
            f"### {title_part} ({series_group}/{era}) "
            f"— Baseline OVRL: {orig_ovrl:.2f}\n"
        )

        lines.append("| Pipeline | OVRL | SIG | BAK | Audio |")
        lines.append("|----------|------|-----|-----|-------|")

        for pipe in pipeline_names:
            scores = seg["scores"].get(pipe, {})
            ovrl = scores.get("dnsmos_ovrl")
            sig = scores.get("dnsmos_sig")
            bak = scores.get("dnsmos_bak")
            ovrl_s = f"{ovrl:.2f}" if ovrl is not None else "—"
            sig_s = f"{sig:.2f}" if sig is not None else "—"
            bak_s = f"{bak:.2f}" if bak is not None else "—"
            audio_tag = (
                f'<audio controls src="audio/{sid}/{pipe}.mp3"></audio>'
            )
            lines.append(f"| {pipe} | {ovrl_s} | {sig_s} | {bak_s} | {audio_tag} |")

        lines.append("")

    # Statistical analysis
    lines.append("## Statistical Analysis\n")

    # Friedman test summary (condensed)
    lines.append("### Friedman Test\n")
    lines.append("| Metric | Statistic | p-value | Significant |")
    lines.append("|--------|-----------|---------|-------------|")
    per_metric = stats.get("per_metric", {})
    for metric_key, metric_label in ANALYSIS_METRICS:
        mdata = per_metric.get(metric_key, {})
        friedman = mdata.get("friedman", {})
        if friedman:
            stat = friedman.get("statistic", 0)
            p_val = friedman.get("p_value", 1)
            sig = "Yes" if friedman.get("significant") else "No"
            lines.append(f"| {metric_label} | {stat:.3f} | {p_val:.6f} | {sig} |")
    lines.append("")

    # Key findings from cross-metric agreement
    cross = stats.get("cross_metric", {})
    if cross:
        lines.append("### Key Findings\n")
        weak_pairs = []
        strong_pairs = []
        for pair_key, pair_data in sorted(cross.items()):
            rho = pair_data.get("spearman_rho", 0)
            labels = pair_data.get("labels", pair_key)
            if pair_data.get("weak_agreement"):
                weak_pairs.append(f"- **Weak agreement** between {labels} "
                                  f"(rho={rho:.3f})")
            elif abs(rho) >= 0.7:
                strong_pairs.append(f"- **Strong agreement** between {labels} "
                                    f"(rho={rho:.3f})")

        for line in strong_pairs + weak_pairs:
            lines.append(line)
        if not strong_pairs and not weak_pairs:
            lines.append("- All metric pairs show moderate agreement.")
        lines.append("")

    # Recommendation
    lines.append("## Recommendation\n")
    lines.append(
        "**`hybrid_demucs_df`** (Demucs vocal separation + DeepFilterNet 12dB "
        "+ loudnorm) is recommended for batch processing. While not the highest-"
        "scoring pipeline on DNSMOS OVRL, it provides meaningful improvement "
        "over the original while preserving ambient character (laughter, room "
        "atmosphere, audience reactions) that gives these archival recordings "
        "their documentary value.\n"
    )
    lines.append(
        "More aggressive pipelines (e.g., `deepfilter_full`) score higher on "
        "objective metrics but risk over-suppressing the ambient soundscape. "
        "The SIG vs BAK tradeoff chart above illustrates this tension.\n"
    )

    # Pipeline by Content Type
    per_format = stats.get("per_format", {})
    if per_format:
        lines.append("## Pipeline by Content Type\n")
        lines.append(
            "Not all content types benefit from the same pipeline. Demucs-based "
            "pipelines use source separation that strips non-speech audio — "
            "destructive for screenings (film audio) and performances (music).\n"
        )
        lines.append("| Content Type | N | Default Pipeline | Rationale |")
        lines.append("|---|---|---|---|")
        rationale = {
            "lecture": "Speech-dominant; Demucs isolates voice cleanly",
            "panel": "Multi-speaker speech; same approach as lecture",
            "book_club": "Reading/discussion; same approach as lecture",
            "screening": "Film audio through speakers; Demucs strips it. Mild filtering preserves film sound",
            "performance": "Music/sound art IS the content; minimal processing only",
        }
        for fg in sorted(per_format.keys()):
            fg_data = per_format[fg]
            n = fg_data.get("n", 0)
            rec = FORMAT_PIPELINE_MAP.get(fg, "hybrid_demucs_df")
            rat = rationale.get(fg, "")
            lines.append(f"| {fg} | {n} | `{rec}` | {rat} |")
        lines.append("")
        lines.append("![Content Type Heatmap](images/format_heatmap.png)\n")

    # Methodology
    lines.append("## Methodology\n")
    lines.append(
        f"- **Sampling**: N={n_seg} segments (one representative clip per event, "
        f"from 429 total clips), stratified by series group, content type, and era\n"
        f"- **Segment extraction**: 45-second speech-active windows via Silero VAD\n"
        f"- **Metrics**: Non-intrusive quality (DNSMOS P.835, UTMOS, NISQA) "
        f"— 10 sub-metrics from 3 independent model families\n"
        f"- **Statistics**: Friedman test (omnibus) + Wilcoxon signed-rank "
        f"(pairwise, Bonferroni corrected) + bootstrap 95% CIs\n"
        f"- **Source**: 429 YouTube recordings (128kbps AAC) from "
        f"The Reading Room BKK (2011-2019)\n"
        f"- **Metric sufficiency**: Intrusive metrics (PESQ, POLQA) require "
        f"clean reference audio, unavailable for archival material. "
        f"Cross-metric agreement analysis confirms DNSMOS and NISQA provide "
        f"consistent assessments; UTMOS shows saturation on this quality level.\n"
    )

    # Write as index.md with Jekyll front matter for GitHub Pages
    index_path = output_dir / "index.md"
    front_matter = "---\nlayout: default\ntitle: Benchmark Report\n---\n\n"
    with open(index_path, "w") as f:
        f.write(front_matter + "\n".join(lines))

    print(f"  Report written to {index_path}")


def cmd_export(output_dir: str | None = None, n_samples: int = 3):
    """Export benchmark report with PNG charts and audio samples."""
    results = _load_results()
    if not results:
        print("No results found. Run 'enhance' first.")
        return

    manifest = _load_manifest()
    segments, pipeline_names = _collect_analysis_data(results)
    if not segments:
        print("No valid scored segments found.")
        return

    stats = _run_statistical_tests(segments, pipeline_names)
    out = Path(output_dir) if output_dir else (ROOT / "docs" / "benchmark-report")

    print(f"Exporting benchmark report to {out}")
    print(f"  {len(segments)} segments, {len(pipeline_names)} pipelines\n")

    # 1. Build and save charts as PNG
    try:
        import altair as alt  # noqa: F401
        import pandas as pd  # noqa: F401
    except ImportError:
        print("  [WARN] altair/pandas not available, skipping charts")
        charts = {}
    else:
        charts = _build_charts(segments, pipeline_names, stats)

    if charts:
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for name, chart in charts.items():
            png_path = img_dir / f"{name}.png"
            print(f"  Saving {name}.png...", end=" ", flush=True)
            chart.save(str(png_path), ppi=150)
            print("done")
        print(f"  Saved {len(charts)} chart images\n")

    # 2. Select & export audio samples
    rep_sids = _select_representative_segments(segments, n=n_samples)
    if rep_sids:
        print(f"  Representative segments: {rep_sids}")
        _export_audio_samples(rep_sids, pipeline_names, out)
        print()

    # 3. Generate markdown report
    _generate_export_markdown(
        segments, pipeline_names, stats, manifest, rep_sids, out,
    )

    print(f"\nExport complete: {out}")
    print(f"  Open: {out / 'index.md'}")


# ── Phase: preview ──────────────────────────────────────────────────

def cmd_preview(output_dir: str | None = None, n_samples: int = 0):
    """Generate HTML audio preview page for benchmark results.

    n_samples=0 (default) means include all benchmarked segments.
    """
    results = _load_results()
    if not results:
        print("No results found. Run 'enhance' first.")
        return

    segments, pipeline_names = _collect_analysis_data(results)
    if not segments:
        print("No valid scored segments found.")
        return

    out = Path(output_dir) if output_dir else (ROOT / "docs" / "audio-preview")

    # Build segment metadata from manifest (titles + video URLs)
    manifest = _load_manifest()
    seg_meta = {}
    for entry in manifest:
        sid = entry["segment_id"]
        seg_meta[sid] = {
            "title": entry.get("event_key", sid),
            "date": entry.get("event_date", ""),
            "video_id": entry.get("video_id", ""),
        }

    # Select and sort segments
    if n_samples <= 0:
        # All segments, sorted by event date
        preview_sids = sorted(
            [s["segment_id"] for s in segments],
            key=lambda sid: seg_meta.get(sid, {}).get("date", ""),
        )
    else:
        anchor_sids = _select_representative_segments(segments, n=3)
        preview_sids = _select_preview_segments(
            segments, n=n_samples, include_sids=anchor_sids,
        )
    if not preview_sids:
        print("No segments with scores available.")
        return

    print(f"Preview: {len(preview_sids)} segments, {len(pipeline_names)} pipelines")
    print(f"  Output:   {out}\n")

    # 3. Encode WAV→MP3 at 64kbps for web preview (skip cached)
    _export_audio_samples(preview_sids, pipeline_names, out, bitrate="64k")

    # 4. Generate HTML
    html_content = _generate_preview_html(
        segments, pipeline_names, preview_sids, results, seg_meta=seg_meta,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html_content, encoding="utf-8")

    print(f"\nPreview page: {out / 'index.html'}")
    print(f"  Open: open {out / 'index.html'}")


def _generate_preview_html(
    segments: list[dict],
    pipeline_names: list[str],
    preview_sids: list[str],
    results: dict,
    seg_meta: dict[str, dict] | None = None,
) -> str:
    """Generate self-contained HTML audio preview page."""
    import html as html_mod

    if seg_meta is None:
        seg_meta = {}

    seg_by_id = {s["segment_id"]: s for s in segments}

    # Compute summary stats per pipeline
    summary: dict[str, dict] = {}
    for pipe_name in pipeline_names:
        ovrl_vals: list[float] = []
        sig_vals: list[float] = []
        bak_vals: list[float] = []
        utmos_vals: list[float] = []
        for seg in segments:
            ps = seg["scores"].get(pipe_name, {})
            if (v := ps.get("dnsmos_ovrl")) is not None:
                ovrl_vals.append(v)
            if (v := ps.get("dnsmos_sig")) is not None:
                sig_vals.append(v)
            if (v := ps.get("dnsmos_bak")) is not None:
                bak_vals.append(v)
            if (v := ps.get("utmos_score")) is not None:
                utmos_vals.append(v)
        if ovrl_vals:
            summary[pipe_name] = {
                "mean_ovrl": sum(ovrl_vals) / len(ovrl_vals),
                "mean_sig": sum(sig_vals) / len(sig_vals) if sig_vals else None,
                "mean_bak": sum(bak_vals) / len(bak_vals) if bak_vals else None,
                "mean_utmos": sum(utmos_vals) / len(utmos_vals) if utmos_vals else None,
                "n": len(ovrl_vals),
            }

    # Find best mean per metric for summary highlighting
    summary_metrics = ["mean_ovrl", "mean_sig", "mean_bak", "mean_utmos"]
    summary_best: dict[str, float] = {}
    non_orig_pipes = [p for p in pipeline_names if p != "original"]
    for mk in summary_metrics:
        vals = [summary.get(p, {}).get(mk) for p in non_orig_pipes]
        valid = [v for v in vals if v is not None]
        summary_best[mk] = max(valid) if valid else 0

    # Summary table rows (sorted by mean OVRL desc)
    sorted_pipes = sorted(
        pipeline_names,
        key=lambda p: summary.get(p, {}).get("mean_ovrl", 0),
        reverse=True,
    )

    summary_rows = []
    for pipe_name in sorted_pipes:
        s = summary.get(pipe_name, {})
        mean_ovrl = s.get("mean_ovrl", 0)
        mean_sig = s.get("mean_sig")
        mean_bak = s.get("mean_bak")
        mean_utmos = s.get("mean_utmos")
        n = s.get("n", 0)
        bar_w = (mean_ovrl / 5.0 * 100) if mean_ovrl else 0
        desc = html_mod.escape(PIPELINE_DESCRIPTIONS.get(pipe_name, ""))

        is_orig = pipe_name == "original"
        # Highlight best-in-column for non-original pipelines
        def _best_cls(val, key):
            if is_orig or val is None:
                return ""
            return " best-score" if abs(val - summary_best.get(key, -1)) < 0.001 else ""

        ovrl_cls = _best_cls(mean_ovrl, "mean_ovrl")
        sig_cls = _best_cls(mean_sig, "mean_sig")
        bak_cls = _best_cls(mean_bak, "mean_bak")
        utmos_cls = _best_cls(mean_utmos, "mean_utmos")

        sig_s = f"{mean_sig:.3f}" if mean_sig is not None else "—"
        bak_s = f"{mean_bak:.3f}" if mean_bak is not None else "—"
        utmos_s = f"{mean_utmos:.3f}" if mean_utmos is not None else "—"

        summary_rows.append(
            f'      <tr title="{desc}">'
            f'<td class="pipe-name">{html_mod.escape(pipe_name)}</td>'
            f'<td class="score{ovrl_cls}"><strong>{mean_ovrl:.3f}</strong></td>'
            f'<td class="score{sig_cls}">{sig_s}</td>'
            f'<td class="score{bak_cls}">{bak_s}</td>'
            f'<td class="score{utmos_cls}">{utmos_s}</td>'
            f'<td class="score">{n}</td>'
            f'<td class="bar-cell"><div class="bar summary-bar" style="width:{bar_w:.0f}%"></div></td>'
            f"</tr>"
        )

    # Per-metric win count (across all benchmark segments, not just preview)
    win_metrics = [
        ("dnsmos_ovrl", "OVRL"), ("dnsmos_sig", "SIG"),
        ("dnsmos_bak", "BAK"), ("utmos_score", "UTMOS"),
        ("nisqa_mos", "NISQA"),
    ]
    from collections import Counter
    win_counts: dict[str, Counter] = {mk: Counter() for mk, _ in win_metrics}
    for seg in segments:
        for mk, _ in win_metrics:
            best_pipe = None
            best_val = -1.0
            for p in pipeline_names:
                if p == "original":
                    continue
                val = seg["scores"].get(p, {}).get(mk)
                if val is not None and val > best_val:
                    best_val = val
                    best_pipe = p
            if best_pipe:
                win_counts[mk][best_pipe] += 1

    # Build win-count table rows
    win_table_rows = []
    for pipe_name in sorted_pipes:
        if pipe_name == "original":
            continue
        cells = [f'<td class="pipe-name">{html_mod.escape(pipe_name)}</td>']
        for mk, _ in win_metrics:
            cnt = win_counts[mk].get(pipe_name, 0)
            pct = cnt / len(segments) * 100 if segments else 0
            is_top = cnt == win_counts[mk].most_common(1)[0][1] if cnt > 0 else False
            cls = " best-score" if is_top else ""
            cells.append(f'<td class="score{cls}">{cnt} ({pct:.0f}%)</td>')
        win_table_rows.append(f'      <tr>{"".join(cells)}</tr>')

    # Pipeline descriptions
    legend_rows = []
    for pipe_name in pipeline_names:
        desc = html_mod.escape(PIPELINE_DESCRIPTIONS.get(pipe_name, ""))
        legend_rows.append(
            f"      <tr><td>{html_mod.escape(pipe_name)}</td><td>{desc}</td></tr>"
        )

    # Sample cards
    score_keys = ["dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "utmos_score"]
    sample_cards = []
    for sid in preview_sids:
        seg = seg_by_id.get(sid)
        if not seg:
            continue
        strata = seg.get("strata", {})
        series_group = html_mod.escape(strata.get("series_group", ""))
        era = html_mod.escape(strata.get("era", ""))
        orig_scores = seg["scores"].get("original", {})
        orig_ovrl = orig_scores.get("dnsmos_ovrl", 0)

        # Find best score per metric (excluding original)
        seg_best: dict[str, float] = {}
        for mk in score_keys:
            vals = []
            for p in pipeline_names:
                if p == "original":
                    continue
                v = seg["scores"].get(p, {}).get(mk)
                if v is not None:
                    vals.append(v)
            seg_best[mk] = max(vals) if vals else 0

        # Player rows
        player_rows = []
        for pipe_name in pipeline_names:
            ps = seg["scores"].get(pipe_name, {})
            ovrl = ps.get("dnsmos_ovrl")
            sig = ps.get("dnsmos_sig")
            bak = ps.get("dnsmos_bak")
            utmos = ps.get("utmos_score")

            is_orig = pipe_name == "original"

            def _fmt(val, key):
                if val is None:
                    return "—", ""
                s = f"{val:.2f}"
                if not is_orig and abs(val - seg_best.get(key, -1)) < 0.001:
                    return s, " best-score"
                return s, ""

            ovrl_s, ovrl_cls = _fmt(ovrl, "dnsmos_ovrl")
            sig_s, sig_cls = _fmt(sig, "dnsmos_sig")
            bak_s, bak_cls = _fmt(bak, "dnsmos_bak")
            utmos_s, utmos_cls = _fmt(utmos, "utmos_score")

            # Delta from original
            orig_ovrl_v = orig_scores.get("dnsmos_ovrl")
            if not is_orig and ovrl is not None and orig_ovrl_v is not None:
                delta = ovrl - orig_ovrl_v
                delta_s = f'<span class="delta {"pos" if delta >= 0 else "neg"}">{"+" if delta >= 0 else ""}{delta:.2f}</span>'
            else:
                delta_s = ""

            bar_w = (ovrl / 5.0 * 100) if ovrl else 0

            player_rows.append(
                f'      <tr class="player-row{" orig-row" if is_orig else ""}">'
                f'<td class="pipe-name">{html_mod.escape(pipe_name)}</td>'
                f'<td class="player-cell">'
                f'<audio controls preload="none" src="audio/{sid}/{pipe_name}.mp3"></audio>'
                f"</td>"
                f'<td class="score{ovrl_cls}">{ovrl_s} {delta_s}</td>'
                f'<td class="score{sig_cls}">{sig_s}</td>'
                f'<td class="score{bak_cls}">{bak_s}</td>'
                f'<td class="score{utmos_cls}">{utmos_s}</td>'
                f'<td class="bar-cell"><div class="bar" style="width:{bar_w:.0f}%"></div></td>'
                f"</tr>"
            )

        # Event title and YouTube link
        meta = seg_meta.get(sid, {})
        event_title = html_mod.escape(meta.get("title", sid))
        video_id = meta.get("video_id", "")
        event_date = html_mod.escape(meta.get("date", ""))
        if video_id:
            yt_url = f"https://www.youtube.com/watch?v={video_id}"
            title_html = f'<a href="{yt_url}" target="_blank" rel="noopener">{event_title}</a>'
        else:
            title_html = event_title

        sample_cards.append(f"""
  <div class="sample-card" id="{sid}">
    <div class="card-header">
      <h3>{title_html}</h3>
      <div class="card-meta">
        <span class="meta-date">{event_date}</span>
        <span class="meta-series">{series_group}</span>
        <span class="meta-era">{era}</span>
        <span class="meta-ovrl">Baseline OVRL: {orig_ovrl:.2f}</span>
      </div>
    </div>
    <table class="player-table">
      <thead>
        <tr>
          <th>Pipeline</th>
          <th>Audio</th>
          <th>OVRL</th>
          <th>SIG</th>
          <th>BAK</th>
          <th>UTMOS</th>
          <th class="bar-cell">Score</th>
        </tr>
      </thead>
      <tbody>
{"".join(player_rows)}
      </tbody>
    </table>
    <div class="card-actions">
      <button class="stop-all-btn" onclick="stopAllInCard(this)">Stop All</button>
    </div>
  </div>""")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Reading Room BKK — Audio Enhancement Preview</title>
<style>
{_preview_css()}
</style>
</head>
<body>

<header>
  <h1>The Reading Room BKK<br><span class="subtitle">Audio Enhancement Preview</span></h1>
  <p class="header-desc">
    Comparing {len(pipeline_names)} audio enhancement pipelines across
    {len(preview_sids)} diverse segments (from {len(segments)} benchmarked).
    <br>
    Each segment is a 45-second speech-active excerpt selected via Silero VAD
    from 429 YouTube recordings (2011–2019).
  </p>
</header>

<section id="summary">
  <h2>Summary — Mean Scores ({len(segments)} segments)</h2>
  <table class="summary-table">
    <thead>
      <tr>
        <th>Pipeline</th>
        <th>OVRL</th>
        <th>SIG</th>
        <th>BAK</th>
        <th>UTMOS</th>
        <th>n</th>
        <th class="bar-cell">Distribution</th>
      </tr>
    </thead>
    <tbody>
{"".join(summary_rows)}
    </tbody>
  </table>
</section>

<section id="metric-wins">
  <h2>Per-Metric Best Pipeline ({len(segments)} segments)</h2>
  <p class="section-desc">
    How often each pipeline achieves the best score for a given metric.
    Different pipelines excel on different dimensions &mdash; no single pipeline dominates all metrics.
  </p>
  <table class="summary-table">
    <thead>
      <tr>
        <th>Pipeline</th>
        <th>OVRL</th>
        <th>SIG</th>
        <th>BAK</th>
        <th>UTMOS</th>
        <th>NISQA</th>
      </tr>
    </thead>
    <tbody>
{"".join(win_table_rows)}
    </tbody>
  </table>
</section>

<section id="pipelines">
  <h2>Pipeline Descriptions</h2>
  <table class="legend-table">
    <thead><tr><th>Pipeline</th><th>Description</th></tr></thead>
    <tbody>
{"".join(legend_rows)}
    </tbody>
  </table>
</section>

<section id="samples">
  <h2>Audio Samples</h2>
  <p class="section-desc">
    {len(preview_sids)} segments selected for diversity across quality range
    and content types. Sorted by event date (oldest first).
    <br>
    <span class="best-score-legend">Green</span> = best score in column for this segment.
    OVRL column shows delta (&Delta;) from original.
  </p>
{"".join(sample_cards)}
</section>

<footer>
  <p>Generated {now} by <code>readingroom_audio.benchmark preview</code>
  — The Reading Room BKK Retrospective Project</p>
</footer>

<script>
{_preview_js()}
</script>
</body>
</html>"""


def _preview_css() -> str:
    """Inline CSS for the audio preview page."""
    return """\
:root {
  --bg: #fafafa;
  --card-bg: #fff;
  --border: #e0e0e0;
  --text: #1a1a1a;
  --text-muted: #666;
  --accent: #2563eb;
  --accent-light: #dbeafe;
  --bar-color: #3b82f6;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans Thai', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

header {
  text-align: center;
  margin-bottom: 3rem;
  padding-bottom: 2rem;
  border-bottom: 2px solid var(--border);
}

header h1 {
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

header .subtitle {
  font-weight: 400;
  font-size: 1.1rem;
  color: var(--text-muted);
}

.header-desc, .section-desc {
  color: var(--text-muted);
  font-size: 0.95rem;
  margin-top: 0.75rem;
}

h2 {
  font-size: 1.3rem;
  margin: 2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

.summary-table, .legend-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  margin-bottom: 2rem;
}

.summary-table th, .summary-table td,
.legend-table th, .legend-table td {
  padding: 0.5rem 0.6rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

.summary-table th, .legend-table th {
  background: #f5f5f5;
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.best-score {
  background: #ecfdf5;
  font-weight: 600;
}

.best-score-legend {
  display: inline-block;
  background: #ecfdf5;
  padding: 0.1rem 0.5rem;
  border-radius: 3px;
  font-weight: 500;
  font-size: 0.85rem;
}

.delta {
  font-size: 0.72rem;
  font-weight: 500;
  margin-left: 0.15rem;
}

.delta.pos { color: #16a34a; }
.delta.neg { color: #dc2626; }

.orig-row { background: #f9fafb; color: var(--text-muted); }

.sample-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 1.5rem;
  overflow: hidden;
}

.card-header {
  padding: 1rem 1.2rem;
  border-bottom: 1px solid var(--border);
}

.card-header h3 {
  font-size: 1.05rem;
  margin-bottom: 0.3rem;
}

.card-header h3 a {
  color: var(--accent);
  text-decoration: none;
}

.card-header h3 a:hover {
  text-decoration: underline;
}

.card-meta {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  font-size: 0.8rem;
  color: var(--text-muted);
}

.meta-date {
  font-weight: 500;
}

.meta-series {
  background: var(--accent-light);
  color: var(--accent);
  padding: 0.1rem 0.5rem;
  border-radius: 3px;
  font-weight: 500;
}

.meta-era {
  background: #f3f4f6;
  padding: 0.1rem 0.5rem;
  border-radius: 3px;
}

.meta-ovrl {
  font-family: monospace;
  font-weight: 500;
}

.player-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.player-table th, .player-table td {
  padding: 0.4rem 0.6rem;
  border-bottom: 1px solid #f0f0f0;
}

.player-table th {
  background: #fafafa;
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.pipe-name {
  font-family: monospace;
  font-size: 0.8rem;
  white-space: nowrap;
}

.player-cell { min-width: 200px; }

.player-cell audio {
  width: 100%;
  height: 32px;
}

.score {
  text-align: right;
  font-family: monospace;
  font-size: 0.82rem;
  white-space: nowrap;
}

.bar-cell { width: 120px; }

.bar {
  height: 14px;
  background: var(--bar-color);
  border-radius: 2px;
  min-width: 2px;
  transition: width 0.3s;
}

.summary-bar { height: 18px; }

.card-actions {
  padding: 0.5rem 1.2rem;
  border-top: 1px solid #f0f0f0;
  text-align: right;
}

.stop-all-btn {
  background: #f3f4f6;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.3rem 0.8rem;
  cursor: pointer;
  font-size: 0.8rem;
  color: var(--text-muted);
}

.stop-all-btn:hover { background: #e5e7eb; }

footer {
  text-align: center;
  color: var(--text-muted);
  font-size: 0.8rem;
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
}

@media (max-width: 768px) {
  body { padding: 1rem 0.5rem; }
  .bar-cell { display: none; }
  .card-meta { flex-direction: column; gap: 0.3rem; }
}"""


def _preview_js() -> str:
    """Inline JavaScript for singleton audio playback with native controls."""
    return """\
// Singleton playback: only one audio plays at a time
document.addEventListener('DOMContentLoaded', function() {
  const allAudio = document.querySelectorAll('audio');
  allAudio.forEach(function(audio) {
    audio.addEventListener('play', function() {
      allAudio.forEach(function(other) {
        if (other !== audio && !other.paused) {
          other.pause();
          other.currentTime = 0;
        }
      });
    });
  });
});

function stopAllInCard(stopBtn) {
  const card = stopBtn.closest('.sample-card');
  if (!card) return;
  card.querySelectorAll('audio').forEach(function(a) {
    a.pause();
    a.currentTime = 0;
  });
}"""


# ── Phase: run-all ───────────────────────────────────────────────────

def _generate_og_image(output_path: Path):
    """Generate OG image (1200x630) from benchmark results."""
    from collections import defaultdict

    from PIL import Image, ImageDraw, ImageFont

    results = _load_results()
    if not results:
        print("  No results for OG image, skipping.")
        return

    # Collect pipeline mean OVRL (only pipelines with >=80% coverage)
    pipe_scores = defaultdict(list)
    for sid, data in results.items():
        for pipe, pdata in data.get("pipelines", {}).items():
            ovrl = pdata.get("scores", {}).get("dnsmos_ovrl")
            if ovrl is not None:
                pipe_scores[pipe].append(ovrl)

    n_total = len(results)
    threshold = int(n_total * 0.8)
    ranked = sorted(
        [(p, sum(v) / len(v)) for p, v in pipe_scores.items() if len(v) >= threshold],
        key=lambda x: x[1],
        reverse=True,
    )

    n_pipelines = len(ranked)
    default_pipe = "hybrid_demucs_df"

    # --- Drawing ---
    W, H = 1200, 630
    bg = (22, 22, 42)
    accent = (37, 99, 235)
    white = (240, 240, 240)
    muted = (140, 140, 160)
    bar_colors = [
        (239, 68, 68), (249, 115, 22), (234, 179, 8),
        (6, 182, 212), (34, 197, 94), (168, 85, 247),
        (236, 72, 153), (99, 102, 241), (156, 163, 175),
    ]

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 44)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
        stat_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 48)
        label_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
        small_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        subtitle_font = title_font
        stat_font = title_font
        label_font = title_font
        small_font = title_font

    # Title
    draw.text((60, 40), "The Reading Room BKK", fill=white, font=title_font)
    draw.text((60, 95), "Audio Enhancement Project", fill=muted, font=subtitle_font)
    draw.rectangle([60, 135, W - 60, 137], fill=accent)

    # Stats row
    stats = [
        (str(n_total), "Events"),
        ("429", "Videos"),
        ("341h", "Audio"),
        (str(n_pipelines), "Pipelines"),
    ]
    stat_x = 60
    for value, label in stats:
        draw.text((stat_x, 155), value, fill=accent, font=stat_font)
        draw.text((stat_x, 210), label, fill=muted, font=label_font)
        stat_x += 220

    # Bar chart — top 5 pipelines + original
    show_pipes = [p for p in ranked if p[0] != "original"][:5]
    original = next((p for p in ranked if p[0] == "original"), ("original", 1.0))
    show_pipes.append(original)
    max_score = max(s for _, s in show_pipes) if show_pipes else 3.0

    bar_y = 260
    bar_h = 30
    bar_gap = 12
    bar_left = 310
    bar_max_w = 400

    for i, (pipe_name, mean_ovrl) in enumerate(show_pipes):
        y = bar_y + i * (bar_h + bar_gap)
        color = bar_colors[i % len(bar_colors)] if pipe_name != "original" else muted

        draw.text((60, y + 4), pipe_name, fill=white, font=label_font)
        bar_w = int((mean_ovrl / max_score) * bar_max_w)
        draw.rectangle([bar_left, y, bar_left + bar_w, y + bar_h], fill=color)
        draw.text((bar_left + bar_w + 15, y + 4), f"{mean_ovrl:.2f}", fill=white, font=label_font)

        if pipe_name == default_pipe:
            tag_x = bar_left + bar_w + 80
            draw.rounded_rectangle(
                [tag_x, y + 2, tag_x + 90, y + bar_h - 2],
                radius=4, fill=(34, 197, 94),
            )
            draw.text((tag_x + 10, y + 5), "default", fill=bg, font=small_font)

    # Footer
    draw.text(
        (60, H - 45),
        "DNSMOS / NISQA / UTMOS quality metrics  |  2010\u20132019 archival recordings",
        fill=muted, font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG", optimize=True)
    print(f"  OG image saved: {output_path}")


def cmd_publish(output_dir: str | None = None, n_samples: int = 3):
    """Regenerate all outputs: analyze → export → preview → OG image.

    One-shot command to refresh docs after model improvements.
    """
    docs_dir = Path(output_dir) if output_dir else (ROOT / "docs")

    print("=" * 70)
    print("PUBLISH: ANALYZE")
    print("=" * 70)
    cmd_analyze()

    print("\n" + "=" * 70)
    print("PUBLISH: EXPORT")
    print("=" * 70)
    export_dir = str(docs_dir / "benchmark-report") if output_dir else None
    cmd_export(output_dir=export_dir, n_samples=n_samples)

    print("\n" + "=" * 70)
    print("PUBLISH: PREVIEW")
    print("=" * 70)
    preview_dir = str(docs_dir / "audio-preview") if output_dir else None
    cmd_preview(output_dir=preview_dir)

    print("\n" + "=" * 70)
    print("PUBLISH: OG IMAGE")
    print("=" * 70)
    _generate_og_image(docs_dir / "og-image.png")

    print("\n" + "=" * 70)
    print("PUBLISH COMPLETE — docs/ ready for deployment")
    print("=" * 70)


def cmd_run_all(
    target_n: int = 40,
    seed: int = 42,
    pipeline_names: list[str] | None = None,
    duration: float = 45.0,
    quick: bool = False,
    fail_fast: bool = True,
):
    """Run all benchmark phases sequentially.

    Args:
        quick: Fast iteration mode — 5 segments, 3 pipelines, DNSMOS only, 15s segments.
        fail_fast: Disable pipelines after 2 consecutive same-category failures.
    """
    if quick:
        target_n = 5
        duration = 15.0
        pipeline_names = QUICK_PIPELINES
        metrics = ["dnsmos"]
        print(">>> QUICK MODE: 5 segments, 3 pipelines, DNSMOS only, 15s")
    else:
        metrics = None
        if pipeline_names is None:
            pipeline_names = BENCHMARK_DEFAULT_PIPELINES
            print(f">>> Using curated defaults: {', '.join(pipeline_names)}")

    # Strip "original" for enhance phase (it's scored inline)
    enhance_pipelines = [p for p in pipeline_names if p != "original"]

    print("=" * 70)
    print("PHASE 1: SELECT")
    print("=" * 70)
    cmd_select(target_n=target_n, seed=seed)

    print("\n" + "=" * 70)
    print("PHASE 2: DOWNLOAD")
    print("=" * 70)
    cmd_download()

    print("\n" + "=" * 70)
    print("PHASE 3: EXTRACT SEGMENTS")
    print("=" * 70)
    cmd_extract(duration=duration)

    print("\n" + "=" * 70)
    print("PHASE 4: ENHANCE + SCORE (baseline scored inline)")
    print("=" * 70)
    cmd_enhance(pipeline_names=enhance_pipelines, metrics=metrics,
                fail_fast=fail_fast)

    print("\n" + "=" * 70)
    print("PHASE 5: ANALYZE")
    print("=" * 70)
    cmd_analyze()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Systematic audio enhancement benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Sub-commands:
  select       Select stratified sample of events
  download     Download audio for selected samples
  extract      Extract speech-active segments using VAD
  baseline     Score original segments
  enhance      Run enhancement pipelines and score
  analyze      Statistical analysis + report + charts
  export       Export report with PNG charts + audio samples
  preview      Generate HTML audio preview page
  publish      Regenerate all outputs (analyze+export+preview)
  sensitivity  Multi-seed sensitivity analysis for sampling
  run-all      Run all phases sequentially

Available pipelines:
{chr(10).join(f'  {k:<22} {v}' for k, v in PIPELINE_DESCRIPTIONS.items())}
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # select
    p_select = subparsers.add_parser("select", help="Select stratified sample")
    p_select.add_argument("--target-n", type=int, default=40)
    p_select.add_argument("--seed", type=int, default=42)

    # download
    subparsers.add_parser("download", help="Download audio for selected samples")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract speech segments via VAD")
    p_extract.add_argument("--duration", type=float, default=45.0)

    # baseline
    subparsers.add_parser("baseline", help="Score original segments")

    # enhance
    p_enhance = subparsers.add_parser("enhance", help="Enhance + score")
    p_enhance.add_argument(
        "--pipelines", nargs="+", default=None,
        choices=list(PIPELINES.keys()),
        help="Pipelines to run (default: all except original)",
    )
    p_enhance.add_argument(
        "--no-fail-fast", action="store_true",
        help="Disable fail-fast (don't skip pipelines after repeated failures)",
    )

    # analyze
    subparsers.add_parser("analyze", help="Statistical analysis + charts")

    # export
    p_export = subparsers.add_parser("export", help="Export report with charts + audio")
    p_export.add_argument("--output-dir", type=str, default=None,
                          help="Output directory (default: docs/benchmark-report)")
    p_export.add_argument("--n-samples", type=int, default=3,
                          help="Number of representative audio samples (default: 3)")

    # preview
    p_preview = subparsers.add_parser("preview", help="Generate HTML audio preview page")
    p_preview.add_argument("--output-dir", type=str, default=None,
                           help="Output directory (default: docs/audio-preview)")
    p_preview.add_argument("--n-samples", type=int, default=0,
                           help="Number of preview segments (0=all)")

    # sensitivity
    p_sens = subparsers.add_parser("sensitivity", help="Multi-seed sensitivity analysis")
    p_sens.add_argument("--target-n", type=int, default=40)
    p_sens.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to test (default: 42 123 7 2024 999)")

    # publish (analyze + export + preview)
    p_pub = subparsers.add_parser("publish", help="Regenerate all outputs (analyze+export+preview)")
    p_pub.add_argument("--output-dir", type=str, default=None,
                       help="Output base directory (default: docs/)")
    p_pub.add_argument("--n-samples", type=int, default=3,
                       help="Number of representative audio samples (default: 3)")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run all phases")
    p_all.add_argument("--target-n", type=int, default=40)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--duration", type=float, default=45.0)
    p_all.add_argument(
        "--pipelines", nargs="+", default=None,
        choices=list(PIPELINES.keys()),
        help=f"Pipelines to run (default: {', '.join(BENCHMARK_DEFAULT_PIPELINES)})",
    )
    p_all.add_argument(
        "--quick", action="store_true",
        help="Fast iteration: 5 segments, 3 pipelines, DNSMOS only, 15s",
    )
    p_all.add_argument(
        "--no-fail-fast", action="store_true",
        help="Disable fail-fast (don't skip pipelines after repeated failures)",
    )

    args = parser.parse_args()

    if args.command == "select":
        cmd_select(target_n=args.target_n, seed=args.seed)
    elif args.command == "download":
        cmd_download()
    elif args.command == "extract":
        cmd_extract(duration=args.duration)
    elif args.command == "baseline":
        cmd_baseline()
    elif args.command == "enhance":
        cmd_enhance(pipeline_names=args.pipelines,
                    fail_fast=not args.no_fail_fast)
    elif args.command == "analyze":
        cmd_analyze()
    elif args.command == "export":
        cmd_export(output_dir=args.output_dir, n_samples=args.n_samples)
    elif args.command == "preview":
        cmd_preview(output_dir=args.output_dir, n_samples=args.n_samples)
    elif args.command == "sensitivity":
        cmd_sensitivity(target_n=args.target_n, seeds=args.seeds)
    elif args.command == "publish":
        cmd_publish(output_dir=args.output_dir, n_samples=args.n_samples)
    elif args.command == "run-all":
        cmd_run_all(
            target_n=args.target_n,
            seed=args.seed,
            pipeline_names=args.pipelines,
            duration=args.duration,
            quick=args.quick,
            fail_fast=not args.no_fail_fast,
        )


if __name__ == "__main__":
    main()
