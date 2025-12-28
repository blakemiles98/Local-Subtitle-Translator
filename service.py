from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core import collect_videos, run_batch


def main() -> None:
    p = argparse.ArgumentParser(description="Headless subtitle service (Whisper + NLLB)")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Video file or folder")
    group.add_argument("--inputs", nargs="+", help="One or more video file paths")
    group.add_argument("--stdin-json", action="store_true", help="Read JSON array of file paths from stdin")

    p.add_argument("--recursive", action="store_true", help="Scan subfolders if input is a folder")
    p.add_argument("--model", default="medium", help="Whisper model (small/medium/etc)")
    p.add_argument("--existing", choices=["skip", "overwrite"], default="skip", help="Existing SRT behavior")

    p.add_argument(
        "--english-output",
        choices=["all", "non_english_only"],
        default="all",
        help="Write .en.srt for all videos, or only for non-English videos (translate-only mode)",
    )

    p.add_argument("--json", action="store_true", help="Print JSON results to stdout (for calling apps)")
    args = p.parse_args()

    videos: list[Path] = []

    if args.stdin_json:
        import sys
        raw = sys.stdin.read()
        paths = json.loads(raw)
        if not isinstance(paths, list):
            raise SystemExit("stdin-json expects a JSON array of file paths")
        videos = [Path(x) for x in paths]

    elif args.inputs:
        videos = [Path(x) for x in args.inputs]

    else:
        inp = Path(args.input)
        if inp.is_file():
            videos = [inp]
        else:
            videos = collect_videos(inp, recursive=args.recursive)

    if not videos:
        raise SystemExit(1)

    def status(s: str, d: str) -> None:
        print(f"[{s}] {d}")

    # progress signature is now: (done, total, elapsed, done_work, total_work, did_work)
    def progress(done: int, total: int, elapsed: float, done_work: int, total_work: int, did_work: bool) -> None:
        # Keep it concise but useful in logs:
        # - done/total files
        # - done_work/total_work seconds (duration work)
        # - did_work indicates whether ETA should learn from this completion
        print(
            f"[PROGRESS] {done}/{total} "
            f"elapsed={elapsed:.1f}s "
            f"work={done_work}/{total_work} "
            f"did_work={did_work}"
        )

    results = run_batch(
        videos=videos,
        whisper_model=args.model,
        existing_srt_mode=args.existing,
        english_output_mode=args.english_output,
        status=status,
        progress=progress,
    )

    if args.json:
        out = [
            {"video": r.video, "ok": r.ok, "elapsed_s": r.elapsed_s, "message": r.message}
            for r in results
        ]
        print(json.dumps(out, indent=2))

    # Non-zero exit if any failed
    if any(not r.ok for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
