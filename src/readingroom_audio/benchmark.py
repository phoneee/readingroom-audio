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
import time
from datetime import UTC, datetime
from pathlib import Path

from .download import download_audio
from .enhance import PIPELINE_DESCRIPTIONS, PIPELINES, get_available_pipelines
from .reporting import (
    build_charts,
    export_audio_samples,
    generate_charts,
    generate_export_markdown,
    generate_og_image,
    generate_preview_html,
    generate_report,
    select_preview_segments,
    select_representative_segments,
)
from .sampling import load_all_events, stratified_sample
from .score import score_segment
from .stats import (
    cross_metric_agreement,  # noqa: F401
    per_format_analysis,  # noqa: F401
    per_stratum_analysis,  # noqa: F401
    run_statistical_tests,
    run_tests_for_metric,  # noqa: F401
)
from .utils import ensure_wav, get_project_root

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
            "timestamp": datetime.now(UTC).isoformat(),
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
                print("  [original] Scoring baseline...", end=" ", flush=True)
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
    stats = run_statistical_tests(segments, pipeline_names)

    # Generate report
    generate_report(segments, pipeline_names, stats, manifest)

    # Generate Altair charts
    generate_charts(segments, pipeline_names, stats)

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
    for _sid, data in results.items():
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
    print("\nPairwise Jaccard similarity:")
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


# ── Phase: export ───────────────────────────────────────────────────

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

    stats = run_statistical_tests(segments, pipeline_names)
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
        charts = build_charts(segments, pipeline_names, stats)

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
    rep_sids = select_representative_segments(segments, n=n_samples)
    if rep_sids:
        print(f"  Representative segments: {rep_sids}")
        export_audio_samples(rep_sids, pipeline_names, out)
        print()

    # 3. Generate markdown report
    generate_export_markdown(
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
        anchor_sids = select_representative_segments(segments, n=3)
        preview_sids = select_preview_segments(
            segments, n=n_samples, include_sids=anchor_sids,
        )
    if not preview_sids:
        print("No segments with scores available.")
        return

    print(f"Preview: {len(preview_sids)} segments, {len(pipeline_names)} pipelines")
    print(f"  Output:   {out}\n")

    # 3. Encode WAV→MP3 at 64kbps for web preview (skip cached)
    export_audio_samples(preview_sids, pipeline_names, out, bitrate="64k")

    # 4. Generate HTML
    html_content = generate_preview_html(
        segments, pipeline_names, preview_sids, results, seg_meta=seg_meta,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html_content, encoding="utf-8")

    print(f"\nPreview page: {out / 'index.html'}")
    print(f"  Open: open {out / 'index.html'}")



# ── Phase: publish ──────────────────────────────────────────────────

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
    generate_og_image(docs_dir / "og-image.png")

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
