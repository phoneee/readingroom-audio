"""Run enhancement pipeline comparison across multiple audio files.

Orchestrates enhancement and scoring, producing a quality report JSON.
"""

import json
import os
import time
from pathlib import Path

from .enhance import PIPELINES, PIPELINE_DESCRIPTIONS
from .score import score_audio, save_report
from .utils import get_project_root


def run_comparison(
    input_files: list[Path],
    output_dir: Path,
    report_path: Path,
    pipelines: list[str] | None = None,
) -> dict:
    """Run enhancement pipelines on input files and score results.

    Args:
        input_files: List of WAV file paths to process.
        output_dir: Directory for enhanced outputs.
        report_path: Path to save JSON quality report.
        pipelines: Pipeline names to run. None = all pipelines.

    Returns:
        Results dict: {filename: {pipeline: {scores}}}.
    """
    if pipelines is None:
        pipelines = list(PIPELINES.keys())

    results = {}

    for input_wav in input_files:
        fname = input_wav.stem
        if not input_wav.exists():
            print(f"[SKIP] {input_wav} not found")
            continue

        results[fname] = {}
        print(f"\n{'='*70}")
        print(f"FILE: {fname}")
        print(f"{'='*70}")

        for pipe_name in pipelines:
            pipe_fn = PIPELINES.get(pipe_name)
            if pipe_name not in PIPELINES:
                print(f"  [{pipe_name}] Unknown pipeline, skipping")
                continue

            pipe_dir = output_dir / pipe_name
            pipe_dir.mkdir(parents=True, exist_ok=True)

            if pipe_name == "original":
                wav_path = str(input_wav)
            else:
                wav_path = str(pipe_dir / f"{fname}.wav")

            # Enhance (skip if output already exists)
            if pipe_name != "original":
                if os.path.exists(wav_path):
                    print(f"  [{pipe_name}] Using cached output")
                else:
                    print(f"  [{pipe_name}] Enhancing...", end=" ", flush=True)
                    t0 = time.time()
                    try:
                        pipe_fn(str(input_wav), wav_path)
                        elapsed = time.time() - t0
                        print(f"done ({elapsed:.1f}s)")
                        results[fname].setdefault(pipe_name, {})["processing_time"] = elapsed
                    except Exception as e:
                        print(f"FAILED: {e}")
                        results[fname][pipe_name] = {"error": str(e)}
                        continue

            if not os.path.exists(wav_path):
                results[fname][pipe_name] = {"error": "output file not created"}
                continue

            # Score
            print(f"  [{pipe_name}] Scoring...", end=" ", flush=True)
            t0 = time.time()
            try:
                scores = score_audio(wav_path)
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s)")
                if pipe_name in results[fname] and isinstance(results[fname][pipe_name], dict):
                    results[fname][pipe_name].update(scores)
                else:
                    results[fname][pipe_name] = scores
            except Exception as e:
                print(f"FAILED: {e}")
                results[fname][pipe_name] = {"error": str(e)}

            _print_scores(results[fname].get(pipe_name, {}))

    save_report(results, report_path)
    print(f"\nFull report saved to: {report_path}")

    _print_summary(results, pipelines)
    return results


def _print_scores(scores: dict):
    """Print key scores for a single pipeline result."""
    if "dnsmos_ovrl" in scores:
        print(
            f"           DNSMOS: sig={scores.get('dnsmos_sig', 0):.2f} "
            f"bak={scores.get('dnsmos_bak', 0):.2f} "
            f"ovrl={scores.get('dnsmos_ovrl', 0):.2f}"
        )


def _print_summary(results: dict, pipelines: list[str]):
    """Print summary table of all results."""
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Pipeline':<22} {'DNSMOS Sig':>10} {'DNSMOS Bak':>10} {'DNSMOS Ovrl':>11}")
    print("-" * 60)

    for fname, file_results in results.items():
        print(f"\n  {fname}:")
        for pipe_name in pipelines:
            s = file_results.get(pipe_name, {})
            if "error" in s:
                print(f"    {pipe_name:<20} ERROR: {s['error'][:40]}")
            elif "dnsmos_ovrl" in s:
                print(
                    f"    {pipe_name:<20} "
                    f"{s.get('dnsmos_sig', 0):>8.2f}  "
                    f"{s.get('dnsmos_bak', 0):>8.2f}  "
                    f"{s.get('dnsmos_ovrl', 0):>9.2f}"
                )


def main():
    import argparse

    root = get_project_root()

    parser = argparse.ArgumentParser(
        description="Run audio enhancement pipeline comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available pipelines:
{chr(10).join(f'  {k:<22} {v}' for k, v in PIPELINE_DESCRIPTIONS.items())}

Examples:
  # Run all pipelines on input files
  python -m readingroom_audio.compare

  # Run specific pipelines
  python -m readingroom_audio.compare --pipelines original deepfilter_12dB hybrid_demucs_df

  # Use custom input directory
  python -m readingroom_audio.compare --input-dir data/audio/raw --output-dir data/audio/enhanced
""",
    )
    parser.add_argument("--input-dir", type=Path, default=root / "data" / "audio" / "compare" / "input",
                        help="Directory containing input WAV files")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "audio" / "compare" / "output",
                        help="Directory for enhanced outputs")
    parser.add_argument("--report", type=Path, default=root / "data" / "audio" / "compare" / "quality_report.json",
                        help="Path for quality report JSON")
    parser.add_argument("--pipelines", nargs="+", default=None,
                        choices=list(PIPELINES.keys()),
                        help="Pipelines to run (default: all)")
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.wav"))
    if not input_files:
        print(f"No WAV files found in {args.input_dir}")
        return

    print(f"Input files: {len(input_files)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Pipelines: {args.pipelines or 'all'}")

    run_comparison(input_files, args.output_dir, args.report, args.pipelines)


if __name__ == "__main__":
    main()
