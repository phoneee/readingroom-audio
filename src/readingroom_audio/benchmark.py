"""Systematic audio enhancement benchmark with statistical analysis.

Sub-command CLI for running stratified benchmark across ~40 samples:

    python -m readingroom_audio.benchmark select    [--target-n 40] [--seed 42]
    python -m readingroom_audio.benchmark download
    python -m readingroom_audio.benchmark extract   [--duration 45]
    python -m readingroom_audio.benchmark baseline
    python -m readingroom_audio.benchmark enhance   [--pipelines ...]
    python -m readingroom_audio.benchmark analyze
    python -m readingroom_audio.benchmark run-all   [--target-n 40] [--pipelines ...]
"""

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

from .download import download_audio
from .enhance import PIPELINES, PIPELINE_DESCRIPTIONS
from .sampling import load_all_events, stratified_sample
from .score import save_report, score_segment
from .utils import ensure_wav, get_project_root

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

def cmd_enhance(pipeline_names: list[str] | None = None):
    """Run enhancement pipelines on all segments and score results."""
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    if pipeline_names is None:
        pipeline_names = [p for p in PIPELINES if p != "original"]
    else:
        pipeline_names = [p for p in pipeline_names if p != "original"]

    results = _load_results()

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
                "pipelines": {
                    "original": {
                        "scores": entry.get("baseline_scores", {}),
                        "processing_time_sec": 0,
                    }
                },
            }

        for pipe_name in pipeline_names:
            pipe_fn = PIPELINES.get(pipe_name)
            if pipe_fn is None:
                continue

            # Check if already done
            existing = results[sid].get("pipelines", {}).get(pipe_name, {})
            if existing.get("scores", {}).get("dnsmos_ovrl") is not None:
                print(f"  [{pipe_name}] cached (OVRL={existing['scores']['dnsmos_ovrl']:.2f})")
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
                except Exception as e:
                    print(f"FAILED: {e}")
                    results[sid].setdefault("pipelines", {})[pipe_name] = {
                        "scores": {"error": str(e)},
                        "processing_time_sec": 0,
                    }
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
                continue

            # Score
            print("scoring...", end=" ", flush=True)
            t0_score = time.time()
            try:
                scores = score_segment(str(enhanced_path))
                score_time = time.time() - t0_score
                results[sid].setdefault("pipelines", {})[pipe_name] = {
                    "scores": scores,
                    "processing_time_sec": round(elapsed, 2),
                }
                ovrl = scores.get("dnsmos_ovrl", "?")
                print(f"done ({score_time:.1f}s) OVRL={ovrl}")
            except Exception as e:
                print(f"score FAILED: {e}")
                results[sid].setdefault("pipelines", {})[pipe_name] = {
                    "scores": {"error": str(e)},
                    "processing_time_sec": round(elapsed, 2),
                }

        # Save after each segment (resumability)
        _save_results(results)

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


def _collect_analysis_data(results: dict) -> tuple[list[dict], list[str]]:
    """Extract structured analysis data from results."""
    # Find all pipelines that have at least some scores
    all_pipes = set()
    for sid, data in results.items():
        for pipe_name, pipe_data in data.get("pipelines", {}).items():
            if pipe_data.get("scores", {}).get("dnsmos_ovrl") is not None:
                all_pipes.add(pipe_name)

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


def _run_statistical_tests(
    segments: list[dict], pipeline_names: list[str]
) -> dict:
    """Run Friedman test and post-hoc Wilcoxon with Bonferroni."""
    import numpy as np
    from scipy import stats as sp_stats

    metric = "dnsmos_ovrl"
    n_pipes = len(pipeline_names)

    # Build score matrix: segments × pipelines
    score_matrix = []
    valid_segments = []
    for seg in segments:
        row = []
        all_present = True
        for pipe in pipeline_names:
            val = seg["scores"].get(pipe, {}).get(metric)
            if val is None:
                all_present = False
                break
            row.append(val)
        if all_present:
            score_matrix.append(row)
            valid_segments.append(seg)

    if len(score_matrix) < 3:
        return {"error": f"Need at least 3 complete segments, got {len(score_matrix)}"}

    matrix = np.array(score_matrix)
    result = {"n_segments": len(score_matrix), "metric": metric}

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
    from scipy.stats import rankdata
    ranks = np.apply_along_axis(rankdata, 1, matrix)
    mean_ranks = ranks.mean(axis=0)
    result["mean_ranks"] = {
        pipe: float(mean_ranks[i]) for i, pipe in enumerate(pipeline_names)
    }

    # Descriptive stats
    result["descriptive"] = {}
    for i, pipe in enumerate(pipeline_names):
        col = matrix[:, i]
        result["descriptive"][pipe] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    # Post-hoc Wilcoxon signed-rank tests (Bonferroni corrected)
    n_pairs = n_pipes * (n_pipes - 1) // 2
    alpha_corrected = 0.05 / n_pairs
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
            # Rank-biserial correlation as effect size
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
    # Group segment indices by series_group
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
    lines.append(f"**Primary metric**: {stats.get('metric', 'dnsmos_ovrl')}\n")

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

    # Descriptive stats
    lines.append("## Pipeline Scores (DNSMOS OVRL)\n")
    lines.append("| Pipeline | Mean | Median | Std | Min | Max | Mean Rank |")
    lines.append("|---|---|---|---|---|---|---|")
    desc = stats.get("descriptive", {})
    ranks = stats.get("mean_ranks", {})
    for pipe in pipeline_names:
        d = desc.get(pipe, {})
        r = ranks.get(pipe, 0)
        lines.append(
            f"| {pipe} | {d.get('mean', 0):.3f} | {d.get('median', 0):.3f} | "
            f"{d.get('std', 0):.3f} | {d.get('min', 0):.3f} | {d.get('max', 0):.3f} | "
            f"{r:.2f} |"
        )
    lines.append("")

    # Friedman test
    friedman = stats.get("friedman", {})
    lines.append("## Friedman Test\n")
    lines.append(f"- **Statistic**: {friedman.get('statistic', 0):.3f}")
    lines.append(f"- **p-value**: {friedman.get('p_value', 1):.6f}")
    sig = "Yes" if friedman.get("significant") else "No"
    lines.append(f"- **Significant** (α=0.05): {sig}\n")

    # Pairwise comparisons
    lines.append("## Pairwise Wilcoxon Tests (Bonferroni corrected)\n")
    alpha = stats.get("alpha_corrected", 0.0018)
    lines.append(f"Corrected α = {alpha:.4f}\n")
    pairwise = stats.get("pairwise", {})
    if pairwise:
        lines.append("| Comparison | Statistic | p-value | Significant | Effect Size |")
        lines.append("|---|---|---|---|---|")
        for pair_key, pair_data in sorted(pairwise.items()):
            if "error" in pair_data:
                lines.append(f"| {pair_key} | — | — | error | — |")
                continue
            sig = "**Yes**" if pair_data.get("significant") else "No"
            lines.append(
                f"| {pair_key} | {pair_data.get('statistic', 0):.1f} | "
                f"{pair_data.get('p_value', 1):.6f} | {sig} | "
                f"{pair_data.get('effect_size', 0):.3f} |"
            )
        lines.append("")

    # Per-stratum results
    lines.append("## Per-Stratum Analysis\n")
    per_stratum = stats.get("per_stratum", {})
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

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"  Report saved to {REPORT_PATH}")


def _generate_charts(
    segments: list[dict],
    pipeline_names: list[str],
    stats: dict,
):
    """Generate Altair HTML charts."""
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        print("  [WARN] altair/pandas not available, skipping charts")
        return

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build long-form DataFrame
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
                "dnsmos_sig": scores.get("dnsmos_sig", 0),
                "dnsmos_bak": scores.get("dnsmos_bak", 0),
                "series_group": strata.get("series_group", ""),
                "era": strata.get("era", ""),
            })

    if not rows:
        print("  No data for charts")
        return

    df = pd.DataFrame(rows)

    # 1. Pipeline boxplot
    chart = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("pipeline:N", sort=pipeline_names, title="Pipeline"),
            y=alt.Y("dnsmos_ovrl:Q", title="DNSMOS OVRL", scale=alt.Scale(zero=False)),
            color=alt.Color("pipeline:N", legend=None),
        )
        .properties(title="DNSMOS OVRL Distribution by Pipeline", width=600, height=350)
    )
    chart.save(str(CHARTS_DIR / "pipeline_boxplot.html"))
    print("  Saved pipeline_boxplot.html")

    # 2. Series heatmap (pipeline × series_group, mean improvement)
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
        chart = (
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
        chart.save(str(CHARTS_DIR / "series_heatmap.html"))
        print("  Saved series_heatmap.html")

    # 3. Baseline quality by era
    orig_df = df[df["pipeline"] == "original"]
    if not orig_df.empty:
        chart = (
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
        chart.save(str(CHARTS_DIR / "baseline_histogram.html"))
        print("  Saved baseline_histogram.html")

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
            chart = (
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
            chart.save(str(CHARTS_DIR / "pairwise_significance.html"))
            print("  Saved pairwise_significance.html")


# ── Phase: run-all ───────────────────────────────────────────────────

def cmd_run_all(
    target_n: int = 40,
    seed: int = 42,
    pipeline_names: list[str] | None = None,
    duration: float = 45.0,
):
    """Run all benchmark phases sequentially."""
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
    print("PHASE 4: BASELINE SCORING")
    print("=" * 70)
    cmd_baseline()

    print("\n" + "=" * 70)
    print("PHASE 5: ENHANCE + SCORE")
    print("=" * 70)
    cmd_enhance(pipeline_names=pipeline_names)

    print("\n" + "=" * 70)
    print("PHASE 6: ANALYZE")
    print("=" * 70)
    cmd_analyze()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Systematic audio enhancement benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Sub-commands:
  select    Select stratified sample of events
  download  Download audio for selected samples
  extract   Extract speech-active segments using VAD
  baseline  Score original segments
  enhance   Run enhancement pipelines and score
  analyze   Statistical analysis + report + charts
  run-all   Run all phases sequentially

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

    # analyze
    subparsers.add_parser("analyze", help="Statistical analysis + charts")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run all phases")
    p_all.add_argument("--target-n", type=int, default=40)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--duration", type=float, default=45.0)
    p_all.add_argument(
        "--pipelines", nargs="+", default=None,
        choices=list(PIPELINES.keys()),
        help="Pipelines to run (default: all)",
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
        cmd_enhance(pipeline_names=args.pipelines)
    elif args.command == "analyze":
        cmd_analyze()
    elif args.command == "run-all":
        cmd_run_all(
            target_n=args.target_n,
            seed=args.seed,
            pipeline_names=args.pipelines,
            duration=args.duration,
        )


if __name__ == "__main__":
    main()
