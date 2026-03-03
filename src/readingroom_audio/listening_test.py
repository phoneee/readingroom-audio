"""Listening test generator — audio enhancement comparison for GitHub Pages.

Generates a self-contained HTML page with embedded audio players for
side-by-side comparison of enhancement pipelines on diverse talk samples.

Six resumable phases:

    python -m readingroom_audio.listening_test select    [--target-n 10] [--seed 7]
    python -m readingroom_audio.listening_test download
    python -m readingroom_audio.listening_test extract   [--duration 30]
    python -m readingroom_audio.listening_test enhance   [--pipelines ...]
    python -m readingroom_audio.listening_test score
    python -m readingroom_audio.listening_test build
    python -m readingroom_audio.listening_test run-all   [--target-n 10] [--pipelines ...]
"""

import argparse
import html
import json
import os
import subprocess
import time
from pathlib import Path

from .download import download_audio
from .enhance import PIPELINES, PIPELINE_DESCRIPTIONS
from .sampling import load_all_events, stratified_sample
from .score import score_segment
from .utils import ensure_wav, get_project_root

# ── Config ──────────────────────────────────────────────────────────

LISTENING_PIPELINES = [
    "original",
    "deepfilter_full",
    "deepfilter_12dB",
    "mossformer2_48k",
    "frcrn_16k",
    "demucs_vocals",
    "ffmpeg_gentle",
    "hybrid_demucs_df",
]

SEGMENT_DURATION = 30.0  # seconds (shorter than benchmark for quick listening)

# ── Paths ───────────────────────────────────────────────────────────

ROOT = get_project_root()
EVENTS_DIR = ROOT / "data" / "events"
WORK_DIR = ROOT / "data" / "audio" / "listening-test"
MANIFEST_PATH = WORK_DIR / "manifest.json"
SEGMENTS_DIR = WORK_DIR / "segments"
ENHANCED_DIR = WORK_DIR / "enhanced"
SCORES_PATH = WORK_DIR / "scores.json"

# Shared raw download cache (same as benchmark/batch)
RAW_DIR = ROOT / "data" / "audio" / "raw"

# Output for GitHub Pages
OUTPUT_DIR = ROOT / "docs" / "listening-test"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"


# ── Manifest I/O ────────────────────────────────────────────────────

def _load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def _save_manifest(manifest: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _load_scores() -> dict:
    if SCORES_PATH.exists():
        with open(SCORES_PATH) as f:
            return json.load(f)
    return {}


def _save_scores(scores: dict):
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


# ── Phase: select ───────────────────────────────────────────────────

def _enrich_with_display(entry: dict, event: dict):
    """Add event display metadata to a manifest entry."""
    be = event.get("baseline_extraction", {})
    bilingual = event.get("bilingual", {})
    video = next(
        (v for v in event.get("videos", []) if v.get("id") == entry["video_id"]),
        event.get("videos", [{}])[0] if event.get("videos") else {},
    )

    entry["display"] = {
        "title": event.get("display_title", event.get("key", "")),
        "series": event.get("series", "Other"),
        "speakers": be.get("display_speakers", []),
        "description_th": bilingual.get("th", ""),
        "description_en": bilingual.get("en", ""),
        "video_url": video.get("url", ""),
        "event_date": event.get("event_date", ""),
        "formats": be.get("formats", []),
    }


def cmd_select(target_n: int = 10, seed: int = 7):
    """Select stratified sample enriched with event display metadata."""
    print(f"Loading events from {EVENTS_DIR}...")
    events = load_all_events(EVENTS_DIR)
    print(f"  Found {len(events)} events")

    manifest = stratified_sample(events, target_n=target_n, seed=seed)
    print(f"  Selected {len(manifest)} samples")

    # Build lookup for enrichment
    events_by_num = {e.get("event_number", 0): e for e in events}

    for entry in manifest:
        event = events_by_num.get(entry["event_number"])
        if event:
            _enrich_with_display(entry, event)

    # Print summary
    from collections import Counter
    groups = Counter(e["strata"]["series_group"] for e in manifest)
    eras = Counter(e["strata"]["era"] for e in manifest)
    print(f"\n  Series groups: {dict(sorted(groups.items()))}")
    print(f"  Eras: {dict(sorted(eras.items()))}")

    _save_manifest(manifest)
    print(f"\n  Manifest saved to {MANIFEST_PATH}")
    return manifest


# ── Phase: download ─────────────────────────────────────────────────

def cmd_download():
    """Download audio for all samples (uses shared raw/ cache)."""
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
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

        m4a_path = RAW_DIR / f"{vid}.m4a"
        if m4a_path.exists():
            entry["status"]["downloaded"] = True
            cached += 1
            continue

        print(f"  [{i}/{total}] Downloading {vid}...", end=" ", flush=True)
        try:
            download_audio(vid, RAW_DIR, format="m4a")
            entry["status"]["downloaded"] = True
            downloaded += 1
            print("done")
        except Exception as e:
            errors += 1
            print(f"FAILED: {e}")

    _save_manifest(manifest)
    print(f"\nDownload summary: {cached} cached, {downloaded} new, {errors} errors")


# ── Phase: extract ──────────────────────────────────────────────────

def cmd_extract(duration: float = SEGMENT_DURATION):
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

        m4a_path = RAW_DIR / f"{vid}.m4a"
        if not m4a_path.exists():
            print(f"  [{i}/{total}] {sid}: source not found, skipping")
            continue

        tmp_wav = str(SEGMENTS_DIR / f"_tmp_{vid}.wav")
        print(f"  [{i}/{total}] {sid}: converting to WAV...", end=" ", flush=True)
        try:
            ensure_wav(str(m4a_path), tmp_wav, sr=48000)
        except Exception as e:
            print(f"conversion FAILED: {e}")
            continue

        print("VAD...", end=" ", flush=True)
        try:
            start, end, ratio = find_best_segment(tmp_wav, duration=duration)
            entry["segment"]["start_sec"] = start
            entry["segment"]["end_sec"] = end
            entry["segment"]["speech_ratio"] = ratio
            print(f"[{start:.1f}–{end:.1f}s, speech={ratio:.0%}]", end=" ")
        except Exception as e:
            print(f"VAD FAILED: {e}")
            total_dur = entry.get("video_duration_seconds", 300)
            start = total_dur * 0.25
            end = start + duration
            entry["segment"]["start_sec"] = start
            entry["segment"]["end_sec"] = end
            entry["segment"]["speech_ratio"] = 0.0
            print(f"[fallback {start:.1f}–{end:.1f}s]", end=" ")

        print("extracting...", end=" ", flush=True)
        try:
            extract_segment(tmp_wav, str(segment_path), start, end, sr=48000)
            entry["status"]["segment_extracted"] = True
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    _save_manifest(manifest)
    print(f"\nSegments saved to {SEGMENTS_DIR}")


# ── Phase: enhance ──────────────────────────────────────────────────

def cmd_enhance(pipeline_names: list[str] | None = None):
    """Run enhancement pipelines on all segments."""
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    if pipeline_names is None:
        pipeline_names = list(LISTENING_PIPELINES)
    # Filter out original — no processing needed
    pipeline_names = [p for p in pipeline_names if p != "original"]

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    total = len(manifest)

    for i, entry in enumerate(manifest, 1):
        sid = entry["segment_id"]
        segment_path = SEGMENTS_DIR / f"{sid}.wav"

        if not segment_path.exists():
            print(f"\n[SKIP] {sid}: segment not found")
            continue

        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {sid}")
        print(f"{'='*60}")

        for pipe_name in pipeline_names:
            pipe_fn = PIPELINES.get(pipe_name)
            if pipe_fn is None:
                continue

            pipe_dir = ENHANCED_DIR / pipe_name
            pipe_dir.mkdir(parents=True, exist_ok=True)
            enhanced_path = pipe_dir / f"{sid}.wav"

            if enhanced_path.exists():
                print(f"  [{pipe_name}] cached")
                continue

            print(f"  [{pipe_name}] Enhancing...", end=" ", flush=True)
            t0 = time.time()
            try:
                pipe_fn(str(segment_path), str(enhanced_path))
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")

    _save_manifest(manifest)
    print("\nEnhancement complete")


# ── Phase: score ────────────────────────────────────────────────────

def cmd_score():
    """Score all segments with DNSMOS (fast, no NISQA/UTMOS)."""
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    scores = _load_scores()
    total = len(manifest)

    for i, entry in enumerate(manifest, 1):
        sid = entry["segment_id"]
        segment_path = SEGMENTS_DIR / f"{sid}.wav"

        if not segment_path.exists():
            print(f"  [{i}/{total}] {sid}: segment not found, skipping")
            continue

        if sid not in scores:
            scores[sid] = {}

        # Score original
        if "original" not in scores[sid]:
            print(f"  [{i}/{total}] {sid} [original] scoring...", end=" ", flush=True)
            try:
                s = score_segment(str(segment_path), metrics=["dnsmos"])
                scores[sid]["original"] = s
                print(f"OVRL={s.get('dnsmos_ovrl', '?'):.2f}")
            except Exception as e:
                print(f"FAILED: {e}")
                scores[sid]["original"] = {"error": str(e)}

        # Score each pipeline
        for pipe_name in LISTENING_PIPELINES:
            if pipe_name == "original":
                continue

            if pipe_name in scores[sid]:
                continue

            enhanced_path = ENHANCED_DIR / pipe_name / f"{sid}.wav"
            if not enhanced_path.exists():
                continue

            print(f"  [{i}/{total}] {sid} [{pipe_name}] scoring...", end=" ", flush=True)
            try:
                s = score_segment(str(enhanced_path), metrics=["dnsmos"])
                scores[sid][pipe_name] = s
                print(f"OVRL={s.get('dnsmos_ovrl', '?'):.2f}")
            except Exception as e:
                print(f"FAILED: {e}")
                scores[sid][pipe_name] = {"error": str(e)}

        _save_scores(scores)

    print(f"\nScores saved to {SCORES_PATH}")


# ── Phase: build ────────────────────────────────────────────────────

def _encode_mp3(input_wav: str, output_mp3: str, bitrate: str = "192k"):
    """Encode WAV to MP3 using ffmpeg."""
    Path(output_mp3).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-i", input_wav,
        "-c:a", "libmp3lame", "-b:a", bitrate,
        output_mp3,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)


def cmd_build():
    """Encode MP3s and generate static HTML listening test page."""
    manifest = _load_manifest()
    if not manifest:
        print("No manifest found. Run 'select' first.")
        return

    scores = _load_scores()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which pipelines have actual enhanced files
    active_pipelines = ["original"]
    for pipe_name in LISTENING_PIPELINES:
        if pipe_name == "original":
            continue
        # Check if at least one segment has this pipeline
        has_any = any(
            (ENHANCED_DIR / pipe_name / f"{e['segment_id']}.wav").exists()
            for e in manifest
        )
        if has_any:
            active_pipelines.append(pipe_name)

    print(f"Active pipelines: {', '.join(active_pipelines)}")
    total_files = 0

    # Encode MP3s
    for entry in manifest:
        sid = entry["segment_id"]

        # Original
        segment_path = SEGMENTS_DIR / f"{sid}.wav"
        if not segment_path.exists():
            print(f"  [SKIP] {sid}: no segment")
            continue

        mp3_dir = AUDIO_OUTPUT_DIR / sid
        mp3_dir.mkdir(parents=True, exist_ok=True)

        mp3_path = mp3_dir / "original.mp3"
        if not mp3_path.exists():
            print(f"  {sid}/original.mp3...", end=" ", flush=True)
            _encode_mp3(str(segment_path), str(mp3_path))
            print("done")
        total_files += 1

        # Enhanced pipelines
        for pipe_name in active_pipelines:
            if pipe_name == "original":
                continue
            enhanced_wav = ENHANCED_DIR / pipe_name / f"{sid}.wav"
            mp3_path = mp3_dir / f"{pipe_name}.mp3"
            if enhanced_wav.exists() and not mp3_path.exists():
                print(f"  {sid}/{pipe_name}.mp3...", end=" ", flush=True)
                _encode_mp3(str(enhanced_wav), str(mp3_path))
                print("done")
            if mp3_path.exists():
                total_files += 1

    print(f"\nEncoded {total_files} MP3 files")

    # Generate HTML
    html_path = OUTPUT_DIR / "index.html"
    html_content = _generate_html(manifest, scores, active_pipelines)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML written to {html_path}")


def _generate_html(
    manifest: list[dict],
    scores: dict,
    active_pipelines: list[str],
) -> str:
    """Generate self-contained HTML listening test page."""
    # Compute summary stats
    summary = {}
    for pipe_name in active_pipelines:
        values = []
        for entry in manifest:
            sid = entry["segment_id"]
            seg_scores = scores.get(sid, {}).get(pipe_name, {})
            ovrl = seg_scores.get("dnsmos_ovrl")
            if ovrl is not None and "error" not in seg_scores:
                values.append(ovrl)
        if values:
            summary[pipe_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }

    # Build sample cards
    sample_cards = []
    for entry in manifest:
        sid = entry["segment_id"]
        segment_path = SEGMENTS_DIR / f"{sid}.wav"
        if not segment_path.exists():
            continue

        display = entry.get("display", {})
        seg_scores = scores.get(sid, {})

        card = _render_sample_card(sid, display, entry, seg_scores, active_pipelines)
        sample_cards.append(card)

    # Summary table
    summary_rows = _render_summary_table(summary, active_pipelines)

    # Pipeline legend
    legend_rows = _render_pipeline_legend(active_pipelines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reading Room BKK — Audio Enhancement Listening Test</title>
<style>
{_get_css()}
</style>
</head>
<body>

<header>
  <h1>The Reading Room BKK<br><span class="subtitle">Audio Enhancement Listening Test</span></h1>
  <p class="header-desc">
    Comparing {len(active_pipelines)} audio enhancement pipelines across
    {len(sample_cards)} diverse talks from 429 YouTube recordings (2011–2019).
    <br>
    Each sample is a {int(SEGMENT_DURATION)}-second speech-active segment selected via Silero VAD.
  </p>
  <p class="header-desc-th">
    เปรียบเทียบ {len(active_pipelines)} ไปป์ไลน์ปรับปรุงเสียงจากบันทึกเสวนา YouTube
    ของ The Reading Room Bangkok (2554–2562) จำนวน 429 รายการ
  </p>
</header>

<section id="summary">
  <h2>Summary — Mean DNSMOS Scores</h2>
  <table class="summary-table">
    <thead>
      <tr>
        <th>Pipeline</th>
        <th>Mean OVRL</th>
        <th>SIG</th>
        <th>BAK</th>
        <th>Range</th>
        <th>n</th>
        <th class="bar-cell">Distribution</th>
      </tr>
    </thead>
    <tbody>
{summary_rows}
    </tbody>
  </table>
</section>

<section id="pipelines">
  <h2>Pipeline Descriptions</h2>
  <table class="legend-table">
    <thead><tr><th>Pipeline</th><th>Description</th></tr></thead>
    <tbody>
{legend_rows}
    </tbody>
  </table>
</section>

<section id="samples">
  <h2>Samples</h2>
{''.join(sample_cards)}
</section>

<footer>
  <p>Generated by <code>readingroom_audio/listening_test.py</code>
  — The Reading Room BKK Retrospective Project</p>
</footer>

<script>
{_get_js()}
</script>
</body>
</html>"""


def _render_sample_card(
    sid: str,
    display: dict,
    entry: dict,
    seg_scores: dict,
    active_pipelines: list[str],
) -> str:
    """Render HTML card for one sample."""
    title = html.escape(display.get("title", sid))
    series = html.escape(display.get("series", ""))
    date = html.escape(display.get("event_date", ""))
    speakers = display.get("speakers", [])
    speakers_str = html.escape(", ".join(speakers)) if speakers else ""
    desc_en = html.escape(display.get("description_en", ""))
    desc_th = html.escape(display.get("description_th", ""))
    video_url = display.get("video_url", "")
    segment = entry.get("segment", {})
    start = segment.get("start_sec", 0)
    end = segment.get("end_sec", 0)
    ratio = segment.get("speech_ratio", 0)

    # Build audio player rows
    player_rows = []
    for pipe_name in active_pipelines:
        ps = seg_scores.get(pipe_name, {})
        ovrl = ps.get("dnsmos_ovrl")
        sig = ps.get("dnsmos_sig")
        bak = ps.get("dnsmos_bak")

        ovrl_str = f"{ovrl:.2f}" if ovrl is not None else "—"
        sig_str = f"{sig:.2f}" if sig is not None else "—"
        bak_str = f"{bak:.2f}" if bak is not None else "—"
        bar_width = (ovrl / 5.0 * 100) if ovrl else 0

        pipe_label = html.escape(pipe_name)
        is_best = pipe_name == "hybrid_demucs_df"

        player_rows.append(f"""      <tr class="player-row{' recommended' if is_best else ''}" data-sample="{sid}">
        <td class="pipe-name">{pipe_label}{'<span class="star">★</span>' if is_best else ''}</td>
        <td class="player-cell">
          <audio preload="none" src="audio/{sid}/{pipe_name}.mp3"></audio>
          <button class="play-btn" onclick="togglePlay(this)">&#9654;</button>
        </td>
        <td class="score">{ovrl_str}</td>
        <td class="score">{sig_str}</td>
        <td class="score">{bak_str}</td>
        <td class="bar-cell"><div class="bar" style="width:{bar_width:.0f}%"></div></td>
      </tr>""")

    video_link = ""
    if video_url:
        video_link = f' <a href="{html.escape(video_url)}" target="_blank" class="yt-link">▶ YouTube</a>'

    return f"""
  <div class="sample-card" id="{sid}">
    <div class="card-header">
      <h3>{title}{video_link}</h3>
      <div class="card-meta">
        <span class="meta-series">{series}</span>
        <span class="meta-date">{date}</span>
        {f'<span class="meta-speakers">{speakers_str}</span>' if speakers_str else ''}
      </div>
      {f'<p class="desc-en">{desc_en}</p>' if desc_en else ''}
      {f'<p class="desc-th">{desc_th}</p>' if desc_th else ''}
      <p class="segment-info">Segment: {start:.1f}–{end:.1f}s (speech ratio: {ratio:.0%})</p>
    </div>
    <table class="player-table">
      <thead>
        <tr>
          <th>Pipeline</th>
          <th>Play</th>
          <th>OVRL</th>
          <th>SIG</th>
          <th>BAK</th>
          <th class="bar-cell">Score</th>
        </tr>
      </thead>
      <tbody>
{''.join(player_rows)}
      </tbody>
    </table>
    <div class="card-actions">
      <button class="stop-all-btn" onclick="stopAllInCard(this)">Stop All</button>
    </div>
  </div>
"""


def _render_summary_table(summary: dict, active_pipelines: list[str]) -> str:
    """Render summary table rows."""
    # Compute mean SIG and BAK per pipeline from scores file
    # We already have OVRL in summary; also need SIG/BAK
    rows = []
    # Sort by mean OVRL descending
    sorted_pipes = sorted(
        active_pipelines,
        key=lambda p: summary.get(p, {}).get("mean", 0),
        reverse=True,
    )
    best_mean = max((summary.get(p, {}).get("mean", 0) for p in active_pipelines), default=0)

    for pipe_name in sorted_pipes:
        s = summary.get(pipe_name, {})
        mean_ovrl = s.get("mean", 0)
        min_ovrl = s.get("min", 0)
        max_ovrl = s.get("max", 0)
        n = s.get("n", 0)
        bar_width = (mean_ovrl / 5.0 * 100) if mean_ovrl else 0
        is_best = mean_ovrl == best_mean and mean_ovrl > 0
        desc = html.escape(PIPELINE_DESCRIPTIONS.get(pipe_name, ""))
        pipe_label = html.escape(pipe_name)

        rows.append(f"""      <tr class="{'best-row' if is_best else ''}" title="{desc}">
        <td class="pipe-name">{pipe_label}</td>
        <td class="score"><strong>{mean_ovrl:.3f}</strong></td>
        <td class="score">—</td>
        <td class="score">—</td>
        <td class="score">{min_ovrl:.2f}–{max_ovrl:.2f}</td>
        <td class="score">{n}</td>
        <td class="bar-cell"><div class="bar summary-bar" style="width:{bar_width:.0f}%"></div></td>
      </tr>""")

    return "\n".join(rows)


def _render_pipeline_legend(active_pipelines: list[str]) -> str:
    """Render pipeline description table rows."""
    rows = []
    for pipe_name in active_pipelines:
        desc = html.escape(PIPELINE_DESCRIPTIONS.get(pipe_name, ""))
        rows.append(f"      <tr><td>{html.escape(pipe_name)}</td><td>{desc}</td></tr>")
    return "\n".join(rows)


def _get_css() -> str:
    """Return inline CSS for the listening test page."""
    return """
:root {
  --bg: #fafafa;
  --card-bg: #fff;
  --border: #e0e0e0;
  --text: #1a1a1a;
  --text-muted: #666;
  --accent: #2563eb;
  --accent-light: #dbeafe;
  --bar-color: #3b82f6;
  --bar-bg: #e5e7eb;
  --best-bg: #f0fdf4;
  --best-border: #86efac;
  --recommended-bg: #eff6ff;
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

.header-desc {
  color: var(--text-muted);
  font-size: 0.95rem;
  margin-top: 0.75rem;
}

.header-desc-th {
  color: var(--text-muted);
  font-size: 0.9rem;
  margin-top: 0.25rem;
  font-style: italic;
}

h2 {
  font-size: 1.3rem;
  margin: 2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

/* Summary & Legend Tables */
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

.best-row { background: var(--best-bg); }

/* Sample Cards */
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

.yt-link {
  font-size: 0.8rem;
  color: var(--accent);
  text-decoration: none;
  margin-left: 0.5rem;
}

.card-meta {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 0.3rem;
}

.meta-series {
  background: var(--accent-light);
  color: var(--accent);
  padding: 0.1rem 0.5rem;
  border-radius: 3px;
  font-weight: 500;
}

.desc-en, .desc-th {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-top: 0.3rem;
}

.desc-th { font-style: italic; }

.segment-info {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 0.4rem;
  font-family: monospace;
}

/* Player Table */
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

.player-row.recommended {
  background: var(--recommended-bg);
}

.pipe-name {
  font-family: monospace;
  font-size: 0.8rem;
  white-space: nowrap;
}

.star {
  color: #f59e0b;
  margin-left: 0.3rem;
}

.player-cell {
  width: 50px;
  text-align: center;
}

.play-btn {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 50%;
  width: 28px;
  height: 28px;
  cursor: pointer;
  font-size: 0.7rem;
  line-height: 28px;
  padding: 0;
}

.play-btn:hover { opacity: 0.85; }
.play-btn.playing { background: #dc2626; }

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
}
"""


def _get_js() -> str:
    """Return inline JavaScript for audio playback control."""
    return """
// Track currently playing audio globally
let currentlyPlaying = null;

function togglePlay(btn) {
  const audio = btn.parentElement.querySelector('audio');
  if (!audio) return;

  if (audio.paused) {
    // Stop any currently playing audio
    if (currentlyPlaying && currentlyPlaying !== audio) {
      currentlyPlaying.pause();
      currentlyPlaying.currentTime = 0;
      const prevBtn = currentlyPlaying.parentElement.querySelector('.play-btn');
      if (prevBtn) {
        prevBtn.classList.remove('playing');
        prevBtn.innerHTML = '\\u25B6';
      }
    }
    audio.play();
    currentlyPlaying = audio;
    btn.classList.add('playing');
    btn.innerHTML = '\\u25A0';

    audio.onended = function() {
      btn.classList.remove('playing');
      btn.innerHTML = '\\u25B6';
      currentlyPlaying = null;
    };
  } else {
    audio.pause();
    audio.currentTime = 0;
    btn.classList.remove('playing');
    btn.innerHTML = '\\u25B6';
    currentlyPlaying = null;
  }
}

function stopAllInCard(stopBtn) {
  const card = stopBtn.closest('.sample-card');
  if (!card) return;
  card.querySelectorAll('audio').forEach(function(audio) {
    audio.pause();
    audio.currentTime = 0;
  });
  card.querySelectorAll('.play-btn').forEach(function(btn) {
    btn.classList.remove('playing');
    btn.innerHTML = '\\u25B6';
  });
  currentlyPlaying = null;
}
"""


# ── Phase: run-all ──────────────────────────────────────────────────

def cmd_run_all(
    target_n: int = 10,
    seed: int = 7,
    pipeline_names: list[str] | None = None,
    duration: float = SEGMENT_DURATION,
):
    """Run all phases sequentially."""
    print("=" * 70)
    print("PHASE 1: SELECT")
    print("=" * 70)
    cmd_select(target_n=target_n, seed=seed)

    print("\n" + "=" * 70)
    print("PHASE 2: DOWNLOAD")
    print("=" * 70)
    cmd_download()

    print("\n" + "=" * 70)
    print("PHASE 3: EXTRACT")
    print("=" * 70)
    cmd_extract(duration=duration)

    print("\n" + "=" * 70)
    print("PHASE 4: ENHANCE")
    print("=" * 70)
    cmd_enhance(pipeline_names=pipeline_names)

    print("\n" + "=" * 70)
    print("PHASE 5: SCORE")
    print("=" * 70)
    cmd_score()

    print("\n" + "=" * 70)
    print("PHASE 6: BUILD")
    print("=" * 70)
    cmd_build()


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Listening test generator for audio enhancement comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Sub-commands:
  select    Select stratified sample of events
  download  Download audio for selected samples
  extract   Extract speech-active segments using VAD
  enhance   Run enhancement pipelines
  score     Score all segments with DNSMOS
  build     Encode MP3s + generate HTML page
  run-all   Run all phases sequentially

Default pipelines:
{chr(10).join(f'  {k:<22} {PIPELINE_DESCRIPTIONS.get(k, "")}' for k in LISTENING_PIPELINES)}
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # select
    p_select = subparsers.add_parser("select", help="Select stratified sample")
    p_select.add_argument("--target-n", type=int, default=10)
    p_select.add_argument("--seed", type=int, default=7)

    # download
    subparsers.add_parser("download", help="Download audio for selected samples")

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract speech segments via VAD")
    p_extract.add_argument("--duration", type=float, default=SEGMENT_DURATION)

    # enhance
    p_enhance = subparsers.add_parser("enhance", help="Enhance segments")
    p_enhance.add_argument(
        "--pipelines", nargs="+", default=None,
        choices=list(PIPELINES.keys()),
        help=f"Pipelines to run (default: {', '.join(LISTENING_PIPELINES[1:])})",
    )

    # score
    subparsers.add_parser("score", help="Score all segments with DNSMOS")

    # build
    subparsers.add_parser("build", help="Encode MP3s + generate HTML")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Run all phases")
    p_all.add_argument("--target-n", type=int, default=10)
    p_all.add_argument("--seed", type=int, default=7)
    p_all.add_argument("--duration", type=float, default=SEGMENT_DURATION)
    p_all.add_argument(
        "--pipelines", nargs="+", default=None,
        choices=list(PIPELINES.keys()),
        help="Pipelines to run (default: all listening test pipelines)",
    )

    args = parser.parse_args()

    if args.command == "select":
        cmd_select(target_n=args.target_n, seed=args.seed)
    elif args.command == "download":
        cmd_download()
    elif args.command == "extract":
        cmd_extract(duration=args.duration)
    elif args.command == "enhance":
        cmd_enhance(pipeline_names=args.pipelines)
    elif args.command == "score":
        cmd_score()
    elif args.command == "build":
        cmd_build()
    elif args.command == "run-all":
        cmd_run_all(
            target_n=args.target_n,
            seed=args.seed,
            pipeline_names=args.pipelines,
            duration=args.duration,
        )


if __name__ == "__main__":
    main()
