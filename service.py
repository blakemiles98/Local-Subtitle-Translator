from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core import collect_videos, run_batch, chunked, filter_videos_for_run


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

    p.add_argument("--chunk-size", type=int, default=750, help="Process in chunks for very large libraries")
    p.add_argument("--chunk-threshold", type=int, default=1500, help="Only chunk when file count exceeds this")

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

    # fast prefilter for skip mode
    videos2 = filter_videos_for_run(videos, args.existing)
    if not videos2:
        if args.json:
            print("[]")
        raise SystemExit(0)

    def status(s: str, d: str) -> None:
        print(f"[{s}] {d}")

    # progress signature: (done, total, elapsed, done_work, total_work, did_work)
    def progress(done: int, total: int, elapsed: float, done_work: int, total_work: int, did_work: bool) -> None:
        print(
            f"[PROGRESS] {done}/{total} "
            f"elapsed={elapsed:.1f}s "
            f"work={done_work}/{total_work} "
            f"did_work={did_work}"
        )

    results = []

    # Chunk if huge
    if len(videos2) > args.chunk_threshold:
        status("Init", f"Large run detected ({len(videos2)} files). Processing in chunks of {args.chunk_size}…")

        total_all = len(videos2)
        global_done = 0

        def chunk_progress(done: int, total: int, elapsed: float, done_work: int, total_work: int, did_work: bool) -> None:
            # Map chunk done -> global done
            progress(global_done + done, total_all, elapsed, done_work, total_work, did_work)

        for part in chunked(videos2, args.chunk_size):
            part_results = run_batch(
                videos=part,
                whisper_model=args.model,
                existing_srt_mode=args.existing,
                english_output_mode=args.english_output,
                status=status,
                progress=chunk_progress,
            )
            results.extend(part_results)
            global_done += len(part)

    else:
        # Normal
        results = run_batch(
            videos=videos2,
            whisper_model=args.model,
            existing_srt_mode=args.existing,
            english_output_mode=args.english_output,
            status=status,
            progress=progress,
        )

    if args.json:
        out = [{"video": r.video, "ok": r.ok, "elapsed_s": r.elapsed_s, "message": r.message} for r in results]
        print(json.dumps(out, indent=2))

    if any(not r.ok for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
