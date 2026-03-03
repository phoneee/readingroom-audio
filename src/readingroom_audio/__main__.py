"""Unified CLI dispatcher for readingroom_audio.

Usage:
    python -m readingroom_audio <command> [args...]

Commands:
    compare        Run enhancement pipeline comparison
    download       Download audio from YouTube
    batch          Batch enhance all 429 videos
    benchmark      Systematic benchmark with statistical analysis
    mux            Mux enhanced audio with original video
    listening-test Generate listening test HTML page
"""

import sys


COMMANDS = {
    "compare": "readingroom_audio.compare",
    "download": "readingroom_audio.download",
    "batch": "readingroom_audio.batch",
    "benchmark": "readingroom_audio.benchmark",
    "mux": "readingroom_audio.mux",
    "listening-test": "readingroom_audio.listening_test",
    "listening_test": "readingroom_audio.listening_test",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        print("\nAvailable commands:")
        for cmd in sorted(set(COMMANDS.keys()) - {"listening_test"}):
            print(f"  {cmd}")
        print("\nFor command-specific help:")
        print("  python -m readingroom_audio <command> --help")
        print("  python -m readingroom_audio.<module> --help")
        return

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(sorted(set(COMMANDS.keys()) - {'listening_test'}))}")
        sys.exit(1)

    # Remove the command from argv so the submodule's argparse sees clean args
    sys.argv = [f"python -m {COMMANDS[command]}"] + sys.argv[2:]

    # Import and run the submodule's main()
    module_name = COMMANDS[command]
    module_path = module_name.split(".")[-1]

    import importlib
    mod = importlib.import_module(f".{module_path}", package="readingroom_audio")
    mod.main()


if __name__ == "__main__":
    main()
