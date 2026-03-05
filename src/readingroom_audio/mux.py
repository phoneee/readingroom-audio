"""Video mux pipeline: combine enhanced audio with original YouTube video.

Downloads video-only streams from YouTube, verifies duration alignment with
enhanced FLAC audio, and muxes into MP4 with AAC audio encoding.

Three-phase architecture:
  Phase A: Identify enhanced FLACs available for chosen pipeline
  Phase B: Download video-only streams in parallel
  Phase C: Verify duration + mux in parallel

Usage:
    python -m readingroom_audio.mux run --pipeline hybrid_demucs_df
    python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --limit 2
    python -m readingroom_audio.mux run --pipeline ffmpeg_gentle --resume
    python -m readingroom_audio.mux verify --pipeline ffmpeg_gentle
    python -m readingroom_audio.mux status
"""

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .batch import load_all_videos
from .download import _has_aria2c
from .utils import get_audio_duration, get_project_root

# ── Paths ────────────────────────────────────────────────────────────

ROOT = get_project_root()
AUDIO_DIR = ROOT / "data" / "audio"
RAW_AUDIO_DIR = AUDIO_DIR / "raw"
ENHANCED_DIR = AUDIO_DIR / "enhanced_final"

VIDEO_DIR = ROOT / "data" / "video"
VIDEO_RAW_DIR = VIDEO_DIR / "raw"
VIDEO_MUXED_DIR = VIDEO_DIR / "muxed"
MUX_STATUS_PATH = VIDEO_DIR / "mux_status.json"

DURATION_TOLERANCE_SEC = 0.5
AAC_BITRATE = "192k"

# ── Status I/O (thread-safe) ────────────────────────────────────────

_status_lock = threading.Lock()


def _load_status() -> dict:
    with _status_lock:
        if MUX_STATUS_PATH.exists():
            with open(MUX_STATUS_PATH) as f:
                return json.load(f)
        return {}


def _save_status(status: dict):
    with _status_lock:
        MUX_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MUX_STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)


def _update_video_status(pipeline_name: str, video_id: str, result: dict):
    """Atomically update a single video's mux status."""
    with _status_lock:
        if MUX_STATUS_PATH.exists():
            with open(MUX_STATUS_PATH) as f:
                status = json.load(f)
        else:
            status = {}
        status.setdefault(pipeline_name, {})[video_id] = result
        MUX_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MUX_STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)


# ── Core functions ───────────────────────────────────────────────────

def download_video(video_id: str, output_dir: Path) -> Path:
    """Download video-only stream from YouTube.

    Prefers MP4 (H.264). If source is WebM, re-encodes to H.264 after download.
    Uses aria2c for acceleration when available.

    Returns path to downloaded MP4 video file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.mp4"

    if output_path.exists():
        return output_path

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Download best video (prefer mp4/h264, fall back to any)
    tmp_output = output_dir / f"{video_id}_tmp.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]/bestvideo",
        "--concurrent-fragments", "4",
        "--socket-timeout", "30",
        "--retries", "3",
        "-o", str(tmp_output),
        url,
    ]

    if _has_aria2c():
        cmd[1:1] = ["--external-downloader", "aria2c",
                     "--external-downloader-args", "-x 8 -k 1M"]

    subprocess.run(cmd, capture_output=True, check=True, timeout=1800)

    # Find the downloaded file (could be .mp4 or .webm)
    downloaded = None
    for ext in [".mp4", ".webm", ".mkv"]:
        candidate = output_dir / f"{video_id}_tmp{ext}"
        if candidate.exists():
            downloaded = candidate
            break

    if downloaded is None:
        raise FileNotFoundError(f"yt-dlp produced no output for {video_id}")

    # Re-encode WebM/MKV to H.264 MP4 if needed
    if downloaded.suffix != ".mp4":
        cmd_reencode = [
            "ffmpeg", "-y", "-threads", "0",
            "-i", str(downloaded),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-an",  # no audio
            str(output_path),
        ]
        subprocess.run(cmd_reencode, capture_output=True, check=True, timeout=3600)
        downloaded.unlink()
    else:
        downloaded.rename(output_path)

    return output_path


def verify_duration(
    video_path: str | Path,
    enhanced_path: str | Path,
    original_m4a_path: str | Path,
    tolerance: float = DURATION_TOLERANCE_SEC,
) -> dict:
    """Compare durations of video, enhanced audio, and original audio.

    Returns dict with validation result and drift measurements.
    """
    video_dur = get_audio_duration(str(video_path))
    enhanced_dur = get_audio_duration(str(enhanced_path))
    original_dur = get_audio_duration(str(original_m4a_path))

    drift_enhanced_vs_original = abs(enhanced_dur - original_dur)
    drift_video_vs_original = abs(video_dur - original_dur)

    valid = (drift_enhanced_vs_original <= tolerance
             and drift_video_vs_original <= tolerance)

    return {
        "valid": valid,
        "video_duration": round(video_dur, 2),
        "enhanced_duration": round(enhanced_dur, 2),
        "original_duration": round(original_dur, 2),
        "drift_enhanced_vs_original": round(drift_enhanced_vs_original, 3),
        "drift_video_vs_original": round(drift_video_vs_original, 3),
    }


def mux_video(video_path: str | Path, enhanced_audio_path: str | Path,
              output_path: str | Path):
    """Mux video with enhanced audio into MP4 with AAC encoding.

    Video stream is copied (no re-encode). Audio is encoded to AAC.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-threads", "0",
        "-i", str(video_path),
        "-i", str(enhanced_audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", AAC_BITRATE,
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=1800)


def process_one_video(
    video: dict,
    pipeline_name: str,
    skip_verify: bool = False,
) -> dict:
    """Full pipeline for one video: check FLAC → download video → verify → mux.

    Returns result dict with timing info and duration check.
    """
    vid = video["video_id"]
    event_num = video["event_number"]
    result = {"status": "pending"}
    t0 = time.time()

    # Locate enhanced FLAC
    flac_dir = ENHANCED_DIR / pipeline_name
    flac_path = flac_dir / f"E{event_num:03d}_{vid}.flac"
    if not flac_path.exists():
        return {"status": "failed", "error": "enhanced FLAC not found"}

    # Check output
    output_dir = VIDEO_MUXED_DIR / pipeline_name
    output_path = output_dir / f"E{event_num:03d}_{vid}.mp4"
    if output_path.exists():
        return {"status": "completed", "note": "already exists"}

    try:
        # Phase B: Download video-only
        t1 = time.time()
        video_path = download_video(vid, VIDEO_RAW_DIR)
        result["download_sec"] = round(time.time() - t1, 1)

        # Phase C: Verify duration alignment
        if not skip_verify:
            t2 = time.time()
            m4a_path = RAW_AUDIO_DIR / f"{vid}.m4a"
            if not m4a_path.exists():
                return {"status": "failed", "error": "original m4a not found for verification"}

            duration_check = verify_duration(video_path, flac_path, m4a_path)
            result["verify_sec"] = round(time.time() - t2, 1)
            result["duration_check"] = duration_check

            if not duration_check["valid"]:
                result["status"] = "failed"
                result["error"] = (
                    f"duration mismatch: enhanced drift "
                    f"{duration_check['drift_enhanced_vs_original']:.3f}s, "
                    f"video drift {duration_check['drift_video_vs_original']:.3f}s"
                )
                result["total_sec"] = round(time.time() - t0, 1)
                return result

        # Mux
        t3 = time.time()
        mux_video(video_path, flac_path, output_path)
        result["mux_sec"] = round(time.time() - t3, 1)

        result["status"] = "completed"
        result["total_sec"] = round(time.time() - t0, 1)
        result["output_path"] = str(output_path)
        result["output_size_mb"] = round(output_path.stat().st_size / (1024 * 1024), 1)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["total_sec"] = round(time.time() - t0, 1)

    return result


# ── Commands ─────────────────────────────────────────────────────────

def cmd_run(
    pipeline_name: str,
    limit: int | None = None,
    resume: bool = False,
    workers: int = 2,
    download_workers: int = 4,
    skip_verify: bool = False,
):
    """Main mux pipeline: download video → verify → mux with enhanced audio."""
    output_dir = VIDEO_MUXED_DIR / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mux pipeline: {pipeline_name}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers} mux, {download_workers} download")
    if skip_verify:
        print("  Duration verification: SKIPPED")
    print()

    # Load all videos
    videos = load_all_videos()
    print(f"Total videos in corpus: {len(videos)}")

    # Phase A: Filter to videos with enhanced FLACs
    flac_dir = ENHANCED_DIR / pipeline_name
    if not flac_dir.exists():
        print(f"No enhanced FLACs found at {flac_dir}")
        print(f"Run: uv run python -m readingroom_audio.batch run --pipeline {pipeline_name}")
        return

    available = []
    for v in videos:
        flac_path = flac_dir / f"E{v['event_number']:03d}_{v['video_id']}.flac"
        if flac_path.exists():
            available.append(v)

    print(f"Enhanced FLACs available: {len(available)}/{len(videos)}")

    if not available:
        print("Nothing to mux.")
        return

    # Load status for resume filtering
    status = _load_status()
    pipeline_status = status.get(pipeline_name, {})

    if resume:
        pending = [
            v for v in available
            if pipeline_status.get(v["video_id"], {}).get("status") != "completed"
        ]
        print(f"Resuming: {len(pending)} pending/failed")
    else:
        pending = available

    if limit:
        pending = pending[:limit]
        print(f"Limited to: {limit} videos")

    if not pending:
        print("Nothing to process.")
        return

    # Phase B: Pre-download video-only streams
    need_download = [
        v for v in pending
        if not (VIDEO_RAW_DIR / f"{v['video_id']}.mp4").exists()
    ]

    if need_download:
        print(f"\n{'='*60}")
        print(f"Phase B: Downloading {len(need_download)} video streams "
              f"({download_workers} workers)")
        print(f"{'='*60}")

        dl_completed = 0
        dl_errors = 0
        _print_lock = threading.Lock()

        def _download_one(v: dict) -> tuple[str, str]:
            try:
                download_video(v["video_id"], VIDEO_RAW_DIR)
                return v["video_id"], "downloaded"
            except Exception as e:
                return v["video_id"], f"error: {e}"

        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = {executor.submit(_download_one, v): v for v in need_download}
            for future in as_completed(futures):
                vid_id, dl_status = future.result()
                with _print_lock:
                    if dl_status == "downloaded":
                        dl_completed += 1
                    else:
                        dl_errors += 1
                    print(
                        f"  [{dl_completed + dl_errors}/{len(need_download)}] "
                        f"{vid_id}: {dl_status}",
                        flush=True,
                    )

        print(f"\nDownload summary: {dl_completed} downloaded, {dl_errors} errors")
    else:
        print(f"\nAll {len(pending)} video files already downloaded")

    # Phase C: Verify + Mux
    # Re-filter to videos with downloaded video files
    muxable = [
        v for v in pending
        if (VIDEO_RAW_DIR / f"{v['video_id']}.mp4").exists()
    ]

    # Skip already-completed
    to_mux = []
    for v in muxable:
        out = output_dir / f"E{v['event_number']:03d}_{v['video_id']}.mp4"
        if not out.exists():
            to_mux.append(v)

    skipped = len(muxable) - len(to_mux)
    if skipped:
        print(f"\n{skipped} already muxed (output exists)")

    if not to_mux:
        print("Nothing to mux.")
        return

    print(f"\n{'='*60}")
    print(f"Phase C: Verify + Mux {len(to_mux)} videos ({workers} workers)")
    print(f"{'='*60}")

    total = len(to_mux)
    completed = 0
    failed = 0
    elapsed_times = []
    _print_lock = threading.Lock()

    def _mux_one(v: dict) -> tuple[dict, dict]:
        r = process_one_video(v, pipeline_name, skip_verify=skip_verify)
        _update_video_status(pipeline_name, v["video_id"], r)
        return v, r

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_mux_one, v): v for v in to_mux}
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
                    size_str = f"{result.get('output_size_mb', 0):.0f}MB"
                    print(
                        f"  [{completed + failed}/{total}] "
                        f"E{event_num:03d} {vid} done in "
                        f"{result.get('total_sec', 0):.0f}s "
                        f"({size_str}) | ETA: {eta_str}",
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

    # Summary
    print(f"\n{'='*60}")
    print(f"Mux complete: {completed} completed, {failed} failed, {skipped} skipped")
    if elapsed_times:
        total_time = sum(elapsed_times)
        print(f"Total mux time: {_format_duration(total_time)} "
              f"(avg {_format_duration(total_time / len(elapsed_times))}/video)")
    print(f"Output: {output_dir}")


def cmd_verify(pipeline_name: str, limit: int | None = None):
    """Dry-run duration verification: compare m4a vs FLAC durations.

    No video download needed — uses existing m4a and FLAC files only.
    """
    videos = load_all_videos()
    flac_dir = ENHANCED_DIR / pipeline_name

    if not flac_dir.exists():
        print(f"No enhanced FLACs found at {flac_dir}")
        return

    # Find videos with both m4a and FLAC
    checkable = []
    for v in videos:
        flac_path = flac_dir / f"E{v['event_number']:03d}_{v['video_id']}.flac"
        m4a_path = RAW_AUDIO_DIR / f"{v['video_id']}.m4a"
        if flac_path.exists() and m4a_path.exists():
            checkable.append(v)

    if limit:
        checkable = checkable[:limit]

    print(f"Verifying {len(checkable)} files for pipeline: {pipeline_name}")
    print(f"Tolerance: {DURATION_TOLERANCE_SEC}s\n")

    mismatches = []
    for i, v in enumerate(checkable, 1):
        vid = v["video_id"]
        event_num = v["event_number"]
        flac_path = flac_dir / f"E{event_num:03d}_{vid}.flac"
        m4a_path = RAW_AUDIO_DIR / f"{vid}.m4a"

        try:
            enhanced_dur = get_audio_duration(str(flac_path))
            original_dur = get_audio_duration(str(m4a_path))
            drift = abs(enhanced_dur - original_dur)

            if drift > DURATION_TOLERANCE_SEC:
                mismatches.append({
                    "video_id": vid,
                    "event_number": event_num,
                    "original_duration": round(original_dur, 2),
                    "enhanced_duration": round(enhanced_dur, 2),
                    "drift": round(drift, 3),
                })
                print(
                    f"  [{i}/{len(checkable)}] E{event_num:03d} {vid} "
                    f"MISMATCH drift={drift:.3f}s "
                    f"(orig={original_dur:.2f}s enh={enhanced_dur:.2f}s)"
                )
            else:
                if i % 50 == 0 or i == len(checkable):
                    print(f"  [{i}/{len(checkable)}] checked...", flush=True)

        except Exception as e:
            print(f"  [{i}/{len(checkable)}] E{event_num:03d} {vid} ERROR: {e}")

    print(f"\nResults: {len(checkable)} checked, {len(mismatches)} mismatches")
    if mismatches:
        print("\nMismatched files:")
        for m in mismatches:
            print(f"  E{m['event_number']:03d} {m['video_id']}: "
                  f"drift={m['drift']:.3f}s")
    else:
        print("All files within tolerance.")


def cmd_dry_run(
    pipeline_name: str,
    limit: int | None = None,
    resume: bool = False,
):
    """Preview what would be muxed without running."""
    videos = load_all_videos()
    flac_dir = ENHANCED_DIR / pipeline_name
    output_dir = VIDEO_MUXED_DIR / pipeline_name

    if not flac_dir.exists():
        print(f"No enhanced FLACs found at {flac_dir}")
        return

    status = _load_status()
    pipeline_status = status.get(pipeline_name, {})

    available = []
    for v in videos:
        flac_path = flac_dir / f"E{v['event_number']:03d}_{v['video_id']}.flac"
        if flac_path.exists():
            available.append(v)

    if resume:
        pending = [
            v for v in available
            if pipeline_status.get(v["video_id"], {}).get("status") != "completed"
        ]
    else:
        pending = available

    if limit:
        pending = pending[:limit]

    need_dl = [v for v in pending if not (VIDEO_RAW_DIR / f"{v['video_id']}.mp4").exists()]
    already_muxed = [
        v for v in pending
        if (output_dir / f"E{v['event_number']:03d}_{v['video_id']}.mp4").exists()
    ]

    print(f"Mux dry-run: {pipeline_name}")
    print(f"  Enhanced FLACs available: {len(available)}/{len(videos)}")
    print(f"  Would mux: {len(pending) - len(already_muxed)}")
    print(f"  Would download video: {len(need_dl)}")
    print(f"  Already muxed: {len(already_muxed)}")


def cmd_status():
    """Print mux pipeline status summary."""
    status = _load_status()
    if not status:
        print("No mux status found. Run 'mux run' first.")
        return

    videos = load_all_videos()
    total_videos = len(videos)

    # Count available enhanced FLACs per pipeline
    print(f"Total videos in corpus: {total_videos}\n")

    for pipeline_name, pipeline_status in sorted(status.items()):
        completed = sum(
            1 for v in pipeline_status.values()
            if isinstance(v, dict) and v.get("status") == "completed"
        )
        failed = sum(
            1 for v in pipeline_status.values()
            if isinstance(v, dict) and v.get("status") == "failed"
        )

        # Count available FLACs
        flac_dir = ENHANCED_DIR / pipeline_name
        flac_count = len(list(flac_dir.glob("*.flac"))) if flac_dir.exists() else 0

        total_time = sum(
            v.get("total_sec", 0) for v in pipeline_status.values()
            if isinstance(v, dict) and v.get("status") == "completed"
        )

        total_size_mb = sum(
            v.get("output_size_mb", 0) for v in pipeline_status.values()
            if isinstance(v, dict) and v.get("status") == "completed"
        )

        # Count muxed output files on disk
        muxed_dir = VIDEO_MUXED_DIR / pipeline_name
        muxed_on_disk = len(list(muxed_dir.glob("*.mp4"))) if muxed_dir.exists() else 0

        pending = flac_count - completed - failed

        print(f"Pipeline: {pipeline_name}")
        print(f"  Enhanced FLACs available: {flac_count}")
        print(f"  Muxed: {completed}/{flac_count} "
              f"({completed/flac_count*100:.1f}%)" if flac_count else
              "  Muxed: 0")
        if muxed_on_disk != completed:
            print(f"  Muxed on disk: {muxed_on_disk}")
        if failed:
            print(f"  Failed: {failed}")
            # Show failure reasons
            fail_reasons: dict[str, int] = {}
            for v in pipeline_status.values():
                if isinstance(v, dict) and v.get("status") == "failed":
                    reason = v.get("error", "unknown")[:60]
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                print(f"    {count}x {reason}")
        if pending > 0:
            print(f"  Pending: {pending}")
        if total_size_mb > 0:
            size_str = (f"{total_size_mb / 1024:.1f}GB"
                        if total_size_mb > 1024 else f"{total_size_mb:.0f}MB")
            print(f"  Total output size: {size_str}")
        if total_time > 0:
            print(f"  Total mux time: {_format_duration(total_time)}")
            if completed > 0:
                avg = total_time / completed
                print(f"  Avg per video: {_format_duration(avg)}")
                if pending > 0:
                    print(f"  Est remaining: {_format_duration(avg * pending)}")

        # Raw video cache info
        raw_videos = list(VIDEO_RAW_DIR.glob("*.mp4")) if VIDEO_RAW_DIR.exists() else []
        if raw_videos:
            raw_size_mb = sum(f.stat().st_size for f in raw_videos) / (1024 * 1024)
            raw_str = (f"{raw_size_mb / 1024:.1f}GB"
                       if raw_size_mb > 1024 else f"{raw_size_mb:.0f}MB")
            print(f"  Raw video cache: {len(raw_videos)} files ({raw_str})")

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
        description="Mux enhanced audio with original YouTube video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify duration alignment (dry-run, no downloads)
  python -m readingroom_audio.mux verify --pipeline ffmpeg_gentle

  # Mux 2 videos for testing
  python -m readingroom_audio.mux run --pipeline ffmpeg_gentle --limit 2

  # Full batch mux with resume
  python -m readingroom_audio.mux run --pipeline hybrid_demucs_df --resume

  # Check status
  python -m readingroom_audio.mux status
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Download video + mux with enhanced audio")
    p_run.add_argument(
        "--pipeline", required=True,
        help="Enhancement pipeline name (must have FLACs in enhanced_final/)",
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
        help="Parallel verify+mux workers (default: 2)",
    )
    p_run.add_argument(
        "--download-workers", type=int, default=4,
        help="Parallel video download workers (default: 4)",
    )
    p_run.add_argument(
        "--skip-verify", action="store_true",
        help="Skip duration verification before mux",
    )
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without actually running",
    )

    # verify
    p_verify = subparsers.add_parser("verify", help="Dry-run duration verification")
    p_verify.add_argument(
        "--pipeline", required=True,
        help="Enhancement pipeline name",
    )
    p_verify.add_argument(
        "--limit", type=int, default=None,
        help="Check only N videos",
    )

    # status
    subparsers.add_parser("status", help="Show mux pipeline status")

    args = parser.parse_args()

    if args.command == "run":
        if args.dry_run:
            cmd_dry_run(
                pipeline_name=args.pipeline,
                limit=args.limit,
                resume=args.resume,
            )
        else:
            cmd_run(
                pipeline_name=args.pipeline,
                limit=args.limit,
                resume=args.resume,
                workers=args.workers,
                download_workers=args.download_workers,
                skip_verify=args.skip_verify,
            )
    elif args.command == "verify":
        cmd_verify(
            pipeline_name=args.pipeline,
            limit=args.limit,
        )
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
