"""yt-dlp batch download with resume support for Reading Room BKK audio.

Downloads audio from YouTube videos listed in the event data,
saving as m4a (original AAC) or converting to WAV for enhancement.
"""

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .utils import get_project_root


def _has_aria2c() -> bool:
    """Check if aria2c is available on the system."""
    return shutil.which("aria2c") is not None


def download_audio(video_id: str, output_dir: Path, format: str = "m4a") -> Path:
    """Download audio from a single YouTube video.

    Uses aria2c as external downloader when available for faster downloads.

    Args:
        video_id: YouTube video ID.
        output_dir: Directory to save downloaded audio.
        format: Output format — 'm4a' (original) or 'wav'.

    Returns:
        Path to downloaded audio file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.{format}"

    if output_path.exists():
        return output_path

    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp", "-x",
        "--audio-format", format,
        "--concurrent-fragments", "4",
        "--socket-timeout", "30",
        "--retries", "3",
        "-o", str(output_path),
        url,
    ]

    # Use aria2c for multi-connection download acceleration
    if _has_aria2c():
        cmd[1:1] = ["--external-downloader", "aria2c",
                     "--external-downloader-args", "-x 8 -k 1M"]

    subprocess.run(cmd, capture_output=True, check=True, timeout=600)
    return output_path


def get_video_ids(events_dir: Path | None = None) -> list[str]:
    """Extract all video IDs from event JSON files.

    Reads video IDs from data/events/*.json (161 committed event files)
    instead of the large metadata JSON.

    Args:
        events_dir: Path to events directory. Defaults to data/events/.

    Returns:
        List of YouTube video IDs.
    """
    if events_dir is None:
        events_dir = get_project_root() / "data" / "events"

    video_ids, seen = [], set()
    for f in sorted(events_dir.glob("*.json")):
        with open(f) as fh:
            event = json.load(fh)
        for video in event.get("videos", []):
            vid = video.get("id", "")
            if vid and vid not in seen:
                seen.add(vid)
                video_ids.append(vid)
    return video_ids


def batch_download(video_ids: list[str], output_dir: Path,
                   format: str = "m4a") -> dict[str, str]:
    """Download audio for multiple videos sequentially with resume support.

    Skips already-downloaded files. Returns status dict.
    """
    results = {}
    total = len(video_ids)

    for i, vid in enumerate(video_ids, 1):
        output_path = output_dir / f"{vid}.{format}"
        if output_path.exists():
            results[vid] = "cached"
            continue

        print(f"  [{i}/{total}] Downloading {vid}...", end=" ", flush=True)
        try:
            download_audio(vid, output_dir, format)
            results[vid] = "downloaded"
            print("done")
        except Exception as e:
            results[vid] = f"error: {e}"
            print(f"FAILED: {e}")

    cached = sum(1 for v in results.values() if v == "cached")
    downloaded = sum(1 for v in results.values() if v == "downloaded")
    errors = sum(1 for v in results.values() if v.startswith("error"))
    print(f"\nSummary: {cached} cached, {downloaded} downloaded, {errors} errors")

    return results


def batch_download_parallel(
    video_ids: list[str],
    output_dir: Path,
    format: str = "m4a",
    workers: int = 4,
) -> dict[str, str]:
    """Download audio for multiple videos in parallel with resume support.

    Uses ThreadPoolExecutor — yt-dlp runs as subprocess so GIL is not a bottleneck.

    Args:
        video_ids: List of YouTube video IDs.
        output_dir: Directory to save downloaded audio.
        format: Output format — 'm4a' or 'wav'.
        workers: Number of parallel download workers (default: 4).

    Returns:
        Dict mapping video_id to status string.
    """
    results = {}
    pending = []

    # Pre-filter cached
    for vid in video_ids:
        output_path = output_dir / f"{vid}.{format}"
        if output_path.exists():
            results[vid] = "cached"
        else:
            pending.append(vid)

    cached = len(results)
    if cached:
        print(f"  {cached} already downloaded (cached)")

    if not pending:
        return results

    total = len(pending)
    completed = 0
    errors = 0

    def _download_one(vid: str) -> tuple[str, str]:
        try:
            download_audio(vid, output_dir, format)
            return vid, "downloaded"
        except Exception as e:
            return vid, f"error: {e}"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, vid): vid for vid in pending}

        for future in as_completed(futures):
            vid, status = future.result()
            results[vid] = status
            if status == "downloaded":
                completed += 1
            else:
                errors += 1
            print(
                f"  [{completed + errors}/{total}] {vid}: {status}",
                flush=True,
            )

    print(
        f"\nSummary: {cached} cached, {completed} downloaded, {errors} errors"
    )
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download audio from Reading Room BKK YouTube videos")
    parser.add_argument("--output-dir", type=Path, default=get_project_root() / "data" / "audio" / "raw",
                        help="Output directory for downloaded audio")
    parser.add_argument("--format", choices=["m4a", "wav"], default="m4a",
                        help="Audio format to download (default: m4a)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of downloads (for testing)")
    parser.add_argument("--video-id", type=str, default=None,
                        help="Download a single video by ID")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    args = parser.parse_args()

    if args.video_id:
        print(f"Downloading {args.video_id}...")
        path = download_audio(args.video_id, args.output_dir, args.format)
        print(f"Saved to: {path}")
    else:
        video_ids = get_video_ids()
        if args.limit:
            video_ids = video_ids[:args.limit]
        print(f"Downloading {len(video_ids)} videos to {args.output_dir}")
        if args.workers > 1:
            batch_download_parallel(video_ids, args.output_dir, args.format,
                                    workers=args.workers)
        else:
            batch_download(video_ids, args.output_dir, args.format)


if __name__ == "__main__":
    main()
