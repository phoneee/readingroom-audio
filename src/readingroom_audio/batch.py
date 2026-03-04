"""Batch audio enhancement for all 429 Reading Room BKK YouTube videos.

Processes every video through a chosen enhancement pipeline, outputting FLAC.
Two-phase parallel architecture: bulk download → parallel enhance.

Usage:
    python -m readingroom_audio.batch run --pipeline hybrid_demucs_df
    python -m readingroom_audio.batch run --pipeline hybrid_demucs_df --limit 10
    python -m readingroom_audio.batch run --pipeline ffmpeg_gentle --workers 4
    python -m readingroom_audio.batch run --resume
    python -m readingroom_audio.batch status
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .download import batch_download_parallel, download_audio
from .enhance import PIPELINES, PIPELINE_DESCRIPTIONS
from .sampling import FORMAT_PIPELINE_MAP, load_all_events
from .utils import encode_flac, ensure_wav, get_project_root

# ── Paths ────────────────────────────────────────────────────────────

ROOT = get_project_root()
EVENTS_DIR = ROOT / "data" / "events"
AUDIO_DIR = ROOT / "data" / "audio"
RAW_DIR = AUDIO_DIR / "raw"
ENHANCED_DIR = AUDIO_DIR / "enhanced_final"
STATUS_PATH = AUDIO_DIR / "batch_status.json"


# ── Status I/O (thread-safe, batched writes) ──────────────────────

_status_lock = threading.Lock()
_status_pending: dict = {}  # accumulated updates not yet flushed
_STATUS_FLUSH_INTERVAL = 10  # flush to disk every N updates


def _load_status() -> dict:
    with _status_lock:
        if STATUS_PATH.exists():
            with open(STATUS_PATH) as f:
                return json.load(f)
        return {}


def _save_status(status: dict):
    with _status_lock:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)


def _flush_pending_status():
    """Write accumulated pending updates to disk."""
    with _status_lock:
        if not _status_pending:
            return
        if STATUS_PATH.exists():
            with open(STATUS_PATH) as f:
                status = json.load(f)
        else:
            status = {}
        for pipeline_name, videos in _status_pending.items():
            status.setdefault(pipeline_name, {}).update(videos)
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        _status_pending.clear()


def _update_video_status(pipeline_name: str, video_id: str, result: dict):
    """Accumulate status update; flush to disk every N updates."""
    with _status_lock:
        _status_pending.setdefault(pipeline_name, {})[video_id] = result
        total_pending = sum(len(v) for v in _status_pending.values())
    if total_pending >= _STATUS_FLUSH_INTERVAL:
        _flush_pending_status()


# ── Video loading ────────────────────────────────────────────────────

def load_all_videos() -> list[dict]:
    """Flatten 161 events into a list of individual video dicts.

    Each dict has: video_id, event_number, event_key, duration_seconds.
    """
    events = load_all_events(EVENTS_DIR)
    videos = []
    seen_ids = set()

    for event in events:
        event_num = event.get("event_number", 0)
        event_key = event.get("key", "")

        for video in event.get("videos", []):
            vid = video.get("id", "")
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)
            videos.append({
                "video_id": vid,
                "event_number": event_num,
                "event_key": event_key,
                "duration_seconds": video.get("duration_seconds", 0),
                "format_group": event.get("_format_group", "lecture"),
            })

    return videos


# ── Per-video enhancement (no download) ─────────────────────────────

def enhance_video(
    video: dict,
    pipeline_name: str,
    pipeline_fn,
    output_dir: Path,
    worker_id: int = 0,
) -> dict:
    """Enhance a single video: WAV convert → enhance → FLAC → cleanup.

    Assumes m4a is already downloaded in RAW_DIR.
    Each worker uses an isolated temp directory to avoid file collisions.

    Returns status dict with timing info.
    """
    vid = video["video_id"]
    event_num = video["event_number"]
    output_flac = output_dir / f"E{event_num:03d}_{vid}.flac"

    if output_flac.exists():
        return {"status": "completed", "note": "already exists"}

    m4a_path = RAW_DIR / f"{vid}.m4a"
    if not m4a_path.exists():
        return {"status": "failed", "error": "m4a not found (download failed?)"}

    result = {"status": "pending"}
    t0 = time.time()

    # Isolated temp dir per worker to avoid collisions
    tmp_dir = output_dir / f"_tmp_{worker_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_wav_src = str(tmp_dir / f"{vid}_src.wav")
    tmp_wav_enh = str(tmp_dir / f"{vid}_enhanced.wav")

    try:
        # Stage 1: Convert to WAV 48kHz
        t1 = time.time()
        ensure_wav(str(m4a_path), tmp_wav_src, sr=48000)
        result["convert_sec"] = round(time.time() - t1, 1)

        # Stage 2: Enhance
        t2 = time.time()
        pipeline_fn(tmp_wav_src, tmp_wav_enh)
        if not os.path.exists(tmp_wav_enh):
            return {"status": "failed", "error": "enhancement produced no output"}
        result["enhance_sec"] = round(time.time() - t2, 1)

        # Stage 3: Encode to FLAC
        t3 = time.time()
        encode_flac(tmp_wav_enh, str(output_flac))
        result["encode_sec"] = round(time.time() - t3, 1)

        result["status"] = "completed"
        result["total_sec"] = round(time.time() - t0, 1)
        result["flac_path"] = str(output_flac)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["total_sec"] = round(time.time() - t0, 1)

    finally:
        for f in [tmp_wav_src, tmp_wav_enh]:
            if os.path.exists(f):
                os.remove(f)

    return result


# Legacy single-video function (download + enhance, for backward compat)

def process_video(
    video: dict,
    pipeline_name: str,
    pipeline_fn,
    output_dir: Path,
) -> dict:
    """Process a single video: download → WAV → enhance → FLAC → cleanup."""
    vid = video["video_id"]
    event_num = video["event_number"]
    output_flac = output_dir / f"E{event_num:03d}_{vid}.flac"

    if output_flac.exists():
        return {"status": "completed", "note": "already exists"}

    result = {"status": "pending"}
    t0 = time.time()

    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_wav_src = str(tmp_dir / f"{vid}_src.wav")
    tmp_wav_enh = str(tmp_dir / f"{vid}_enhanced.wav")

    try:
        # Stage 1: Download m4a
        m4a_path = download_audio(vid, RAW_DIR, format="m4a")
        if not m4a_path.exists():
            return {"status": "failed", "error": "download failed"}
        result["download_sec"] = round(time.time() - t0, 1)

        # Stage 2: Convert to WAV 48kHz
        t1 = time.time()
        ensure_wav(str(m4a_path), tmp_wav_src, sr=48000)
        result["convert_sec"] = round(time.time() - t1, 1)

        # Stage 3: Enhance
        t2 = time.time()
        pipeline_fn(tmp_wav_src, tmp_wav_enh)
        if not os.path.exists(tmp_wav_enh):
            return {"status": "failed", "error": "enhancement produced no output"}
        result["enhance_sec"] = round(time.time() - t2, 1)

        # Stage 4: Encode to FLAC
        t3 = time.time()
        encode_flac(tmp_wav_enh, str(output_flac))
        result["encode_sec"] = round(time.time() - t3, 1)

        result["status"] = "completed"
        result["total_sec"] = round(time.time() - t0, 1)
        result["flac_path"] = str(output_flac)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["total_sec"] = round(time.time() - t0, 1)

    finally:
        for f in [tmp_wav_src, tmp_wav_enh]:
            if os.path.exists(f):
                os.remove(f)

    return result


# ── Commands ─────────────────────────────────────────────────────────

def cmd_run(
    pipeline_name: str,
    limit: int | None = None,
    resume: bool = False,
    workers: int = 2,
    download_workers: int = 4,
    _videos_override: list[dict] | None = None,
):
    """Main batch processing loop with parallel download + parallel enhance.

    When _videos_override is provided, uses that list instead of loading all
    videos. This allows cmd_run_auto() to reuse all download/enhance logic.
    """
    if pipeline_name not in PIPELINES:
        print(f"Unknown pipeline: {pipeline_name}")
        print(f"Available: {', '.join(PIPELINES.keys())}")
        return

    if pipeline_name == "original":
        print("Cannot batch-process with 'original' (no-op pipeline)")
        return

    pipeline_fn = PIPELINES[pipeline_name]
    output_dir = ENHANCED_DIR / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline: {pipeline_name}")
    print(f"  {PIPELINE_DESCRIPTIONS.get(pipeline_name, '')}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers} enhance, {download_workers} download")

    # Load videos
    videos = _videos_override if _videos_override is not None else load_all_videos()
    print(f"Total videos: {len(videos)}")

    # Load status
    status = _load_status()
    pipeline_status = status.setdefault(pipeline_name, {})

    # Filter based on resume / limit
    if resume:
        pending = [
            v for v in videos
            if pipeline_status.get(v["video_id"], {}).get("status") != "completed"
        ]
        print(f"Resuming: {len(pending)} pending/failed of {len(videos)}")
    else:
        pending = videos

    if limit:
        pending = pending[:limit]
        print(f"Limited to: {limit} videos")

    if not pending:
        print("Nothing to process.")
        return

    # ── Phase A: Pre-download all pending m4a files in parallel ──────
    need_download = [
        v for v in pending
        if not (RAW_DIR / f"{v['video_id']}.m4a").exists()
    ]

    if need_download:
        print(f"\n{'='*60}")
        print(f"Phase A: Downloading {len(need_download)} audio files "
              f"({download_workers} workers)")
        print(f"{'='*60}")

        dl_ids = [v["video_id"] for v in need_download]
        dl_results = batch_download_parallel(
            dl_ids, RAW_DIR, format="m4a", workers=download_workers,
        )

        dl_errors = sum(1 for v in dl_results.values() if v.startswith("error"))
        if dl_errors:
            print(f"  Warning: {dl_errors} downloads failed")
    else:
        print(f"\nAll {len(pending)} audio files already downloaded")

    # ── Phase B: Enhance in parallel ─────────────────────────────────
    # Re-filter: only videos with m4a available
    enhanceable = [
        v for v in pending
        if (RAW_DIR / f"{v['video_id']}.m4a").exists()
    ]

    # Skip already-completed FLAC files
    to_enhance = []
    for v in enhanceable:
        output_flac = output_dir / f"E{v['event_number']:03d}_{v['video_id']}.flac"
        if not output_flac.exists():
            to_enhance.append(v)

    skipped = len(enhanceable) - len(to_enhance)
    if skipped:
        print(f"\n{skipped} already enhanced (FLAC exists)")

    if not to_enhance:
        print("Nothing to enhance.")
        return

    print(f"\n{'='*60}")
    print(f"Phase B: Enhancing {len(to_enhance)} videos ({workers} workers)")
    print(f"{'='*60}")

    total = len(to_enhance)
    completed = 0
    failed = 0
    elapsed_times = []
    _print_lock = threading.Lock()

    if workers <= 1:
        # Sequential mode
        for i, video in enumerate(to_enhance, 1):
            vid = video["video_id"]
            event_num = video["event_number"]
            dur = video["duration_seconds"]

            if elapsed_times:
                avg_time = sum(elapsed_times) / len(elapsed_times)
                eta_sec = avg_time * (total - i + 1)
                eta_str = _format_duration(eta_sec)
            else:
                eta_str = "calculating..."

            print(
                f"\n[{i}/{total}] E{event_num:03d} {vid} "
                f"({_format_duration(dur)}) | ETA: {eta_str}"
            )

            result = enhance_video(video, pipeline_name, pipeline_fn,
                                   output_dir, worker_id=0)

            _update_video_status(pipeline_name, vid, result)

            if result["status"] == "completed":
                completed += 1
                elapsed_times.append(result.get("total_sec", 0))
                print(
                    f"  Done in {result.get('total_sec', 0):.0f}s "
                    f"(cvt:{result.get('convert_sec', 0):.0f} "
                    f"enh:{result.get('enhance_sec', 0):.0f} "
                    f"enc:{result.get('encode_sec', 0):.0f})"
                )
            else:
                failed += 1
                print(f"  FAILED: {result.get('error', 'unknown')}")
    else:
        # Parallel mode
        def _enhance_one(args: tuple) -> tuple[dict, dict]:
            video, wid = args
            r = enhance_video(video, pipeline_name, pipeline_fn,
                              output_dir, worker_id=wid)
            _update_video_status(pipeline_name, video["video_id"], r)
            return video, r

        # Assign worker IDs round-robin
        work_items = [(v, i % workers) for i, v in enumerate(to_enhance)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_enhance_one, item): item[0]
                for item in work_items
            }

            for future in as_completed(futures):
                video, result = future.result()
                vid = video["video_id"]
                event_num = video["event_number"]

                with _print_lock:
                    if result["status"] == "completed":
                        completed += 1
                        elapsed_times.append(result.get("total_sec", 0))
                        avg = sum(elapsed_times) / len(elapsed_times)
                        remaining = total - completed - failed
                        eta_str = _format_duration(avg * remaining) if remaining else "done"
                        print(
                            f"  [{completed + failed}/{total}] "
                            f"E{event_num:03d} {vid} done in "
                            f"{result.get('total_sec', 0):.0f}s "
                            f"(cvt:{result.get('convert_sec', 0):.0f} "
                            f"enh:{result.get('enhance_sec', 0):.0f} "
                            f"enc:{result.get('encode_sec', 0):.0f}) "
                            f"| ETA: {eta_str}",
                            flush=True,
                        )
                    else:
                        failed += 1
                        print(
                            f"  [{completed + failed}/{total}] "
                            f"E{event_num:03d} {vid} FAILED: "
                            f"{result.get('error', 'unknown')}",
                            flush=True,
                        )

    # Flush any remaining pending status updates
    _flush_pending_status()

    # Cleanup empty temp dirs
    for tmp in output_dir.glob("_tmp_*"):
        if tmp.is_dir() and not any(tmp.iterdir()):
            tmp.rmdir()

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch complete: {completed} completed, {failed} failed, "
          f"{skipped} skipped")
    if elapsed_times:
        total_time = sum(elapsed_times)
        print(f"Total enhance time: {_format_duration(total_time)} "
              f"(avg {_format_duration(total_time / len(elapsed_times))}/video)")
    print(f"Output: {output_dir}")


def cmd_run_auto(
    limit: int | None = None,
    resume: bool = False,
    workers: int = 2,
    download_workers: int = 4,
):
    """Auto-select pipeline per content type and run batch enhancement.

    Groups videos by format_group → maps to pipeline via FORMAT_PIPELINE_MAP
    → calls cmd_run() per pipeline group.
    """
    from collections import Counter

    videos = load_all_videos()
    print(f"Total videos: {len(videos)}")

    if limit:
        videos = videos[:limit]
        print(f"Limited to: {limit} videos")

    # Group by format → pipeline
    groups: dict[str, list[dict]] = {}
    for v in videos:
        fg = v.get("format_group", "lecture")
        pipeline = FORMAT_PIPELINE_MAP.get(fg, "hybrid_demucs_df")
        groups.setdefault(pipeline, []).append(v)

    # Summary
    print(f"\nAuto-pipeline assignment:")
    format_counts = Counter(v.get("format_group", "lecture") for v in videos)
    for fg, count in sorted(format_counts.items()):
        pipeline = FORMAT_PIPELINE_MAP.get(fg, "hybrid_demucs_df")
        print(f"  {fg}: {count} videos → {pipeline}")
    print()

    # Process each pipeline group
    for pipeline_name, group_videos in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Processing {len(group_videos)} videos with {pipeline_name}")
        print(f"{'='*60}")
        cmd_run(
            pipeline_name=pipeline_name,
            limit=None,  # already limited above
            resume=resume,
            workers=workers,
            download_workers=download_workers,
            _videos_override=group_videos,
        )


def cmd_status():
    """Print batch processing status summary."""
    status = _load_status()
    if not status:
        print("No batch status found. Run 'batch run' first.")
        return

    videos = load_all_videos()
    total_videos = len(videos)

    print(f"Total videos in corpus: {total_videos}")
    print()

    for pipeline_name, pipeline_status in sorted(status.items()):
        completed = sum(
            1 for v in pipeline_status.values()
            if isinstance(v, dict) and v.get("status") == "completed"
        )
        failed = sum(
            1 for v in pipeline_status.values()
            if isinstance(v, dict) and v.get("status") == "failed"
        )
        pending = total_videos - completed - failed

        total_time = sum(
            v.get("total_sec", 0) for v in pipeline_status.values()
            if isinstance(v, dict) and v.get("status") == "completed"
        )

        print(f"Pipeline: {pipeline_name}")
        print(f"  {PIPELINE_DESCRIPTIONS.get(pipeline_name, '')}")
        print(f"  Completed: {completed}/{total_videos} "
              f"({completed/total_videos*100:.1f}%)")
        if failed:
            print(f"  Failed: {failed}")
        if pending > 0:
            print(f"  Pending: {pending}")
        if total_time > 0:
            print(f"  Total time: {_format_duration(total_time)}")
            if completed > 0:
                avg = total_time / completed
                print(f"  Avg per video: {_format_duration(avg)}")
                if pending > 0:
                    print(f"  Est remaining: {_format_duration(avg * pending)}")
        print()


# ── Helpers ──────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m:02d}m"


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch audio enhancement for all 429 videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available pipelines:
{chr(10).join(f'  {k:<30} {v}' for k, v in PIPELINE_DESCRIPTIONS.items() if k != 'original')}

Worker guidelines:
  ffmpeg_gentle:     4-6 workers (CPU-only, lightweight)
  deepfilter_*:      2-3 workers (each model ~2GB RAM)
  demucs_* / hybrid: 1-2 workers (large models, ~4GB RAM each)
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run batch enhancement")
    pipeline_group = p_run.add_mutually_exclusive_group(required=True)
    pipeline_group.add_argument(
        "--pipeline",
        choices=[k for k in PIPELINES if k != "original"],
        help="Enhancement pipeline to use",
    )
    pipeline_group.add_argument(
        "--auto-pipeline", action="store_true",
        help="Auto-select pipeline per content type (lecture→hybrid_demucs_df, "
             "screening→deepfilter_12dB, performance→ffmpeg_gentle)",
    )
    p_run.add_argument(
        "--limit", type=int, default=None,
        help="Process only N videos (for testing)",
    )
    p_run.add_argument(
        "--resume", action="store_true",
        help="Resume: skip completed videos, re-process failed",
    )
    p_run.add_argument(
        "--workers", type=int, default=2,
        help="Parallel enhance workers (default: 2, see guidelines above)",
    )
    p_run.add_argument(
        "--download-workers", type=int, default=4,
        help="Parallel download workers (default: 4)",
    )

    # status
    subparsers.add_parser("status", help="Show batch processing status")

    args = parser.parse_args()

    if args.command == "run":
        if args.auto_pipeline:
            cmd_run_auto(
                limit=args.limit,
                resume=args.resume,
                workers=args.workers,
                download_workers=args.download_workers,
            )
        else:
            cmd_run(
                pipeline_name=args.pipeline,
                limit=args.limit,
                resume=args.resume,
                workers=args.workers,
                download_workers=args.download_workers,
            )
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
