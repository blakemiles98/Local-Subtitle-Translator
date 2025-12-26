from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core import collect_videos, run_batch

def main():
    p = argparse.ArgumentParser(description="Headless subtitle service (Whisper + NLLB)")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Video file or folder")
    group.add_argument("--inputs", nargs="+", help="One or more video file paths")
    group.add_argument("--stdin-json", action="store_true", help="Read JSON array of file paths from stdin")

    p.add_argument("--recursive", action="store_true", help="Scan subfolders if input is a folder")
    p.add_argument("--model", default="medium", help="Whisper model (small/medium/etc)")
    p.add_argument("--existing", choices=["skip", "overwrite"], default="skip", help="Existing SRT behavior")
    p.add_argument("--json", action="store_true", help="Print JSON results to stdout (for calling apps)")
    args = p.parse_args()

    videos: list[Path] = []

    if args.stdin_json:
        import sys
        raw = sys.stdin.read()
        paths = json.loads(raw)
        if not isinstance(paths, list):
            raise SystemExit("stdin-json expects a JSON array of file paths")
        videos = [Path(p) for p in paths]

    elif args.inputs:
        videos = [Path(p) for p in args.inputs]

    else:
        inp = Path(args.input)
        if inp.is_file():
            videos = [inp]
        else:
            videos = collect_videos(inp, recursive=args.recursive)

    def status(s, d):
        # caller can parse this if they want; good for systemd logs
        print(f"[{s}] {d}")

    def progress(done, total, elapsed):
        print(f"[PROGRESS] {done}/{total} elapsed={elapsed:.1f}s")

    results = run_batch(
        videos=videos,
        whisper_model=args.model,
        existing_srt_mode=args.existing,
        status=status,
        progress=progress,
    )

    if args.json:
        out = [
            {"video": r.video, "ok": r.ok, "elapsed_s": r.elapsed_s, "message": r.message}
            for r in results
        ]
        print(json.dumps(out, indent=2))

    # exit code: nonzero if any failures
    if any(not r.ok for r in results):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
