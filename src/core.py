from __future__ import annotations

import time
import srt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.whisper_srt import transcribe_to_srt
from src.nllb_translate import NllbTranslator
from src.lang_map import WHISPER_TO_NLLB

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

def has_real_text(srt_text: str) -> bool:
    return any(ch.isalnum() for ch in srt_text)

def collect_videos(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        vids = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    else:
        vids = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(vids)

@dataclass
class Result:
    ok: bool
    message: str
    video: str
    elapsed_s: float

StatusFn = Callable[[str, str], None]
ProgressFn = Callable[[int, int, float], None]

def process_one_video(
    video_path: Path,
    whisper_model: str,
    existing_srt_mode: str,
    translator_cache: dict,
    whisper_cache: dict,
    status: StatusFn | None = None,
) -> Result:
    t0 = time.perf_counter()

    out_dir = video_path.parent
    base_name = video_path.stem

    final_srt_path = out_dir / f"{base_name}.en.srt"
    fallback_source_srt_path = out_dir / f"{base_name}.source.srt"

    # OPTIONAL GUARD: cap absurdly-long subtitle files so one giant video doesn't blow up translation
    MAX_SUBS = 8000  # adjust as you like (e.g., 3000-12000)

    if existing_srt_mode == "overwrite":
        fallback_source_srt_path.unlink(missing_ok=True)

    if final_srt_path.exists() and existing_srt_mode == "skip":
        if status:
            status("Skipped", f"Existing SRT found, skipping: {video_path.name}")
        return Result(True, "skipped (srt already exists)", video_path.name, time.perf_counter() - t0)

    if status:
        status("Whisper", f"Transcribing: {video_path.name}")
    try:
        detected_lang, probs, source_srt = transcribe_to_srt(
            str(video_path),
            model_name=whisper_model,
            whisper_cache=whisper_cache,
        )
    except Exception as e:
        if status:
            status("Skipped", f"Audio decode failed: {video_path.name}")
        return Result(False, f"audio decode failed ({e})", video_path.name, time.perf_counter() - t0)


    if not has_real_text(source_srt):
        if status:
            status("Skipped", f"No speech detected: {video_path.name}")
        return Result(False, "no srt created (no speech detected)", video_path.name, time.perf_counter() - t0)

    # NEW: segment-count guard (prevents translation blowups on huge videos)
    try:
        subs = list(srt.parse(source_srt))
    except Exception:
        subs = []

    if len(subs) > MAX_SUBS:
        # Keep the source so user still gets something, but skip translation to avoid crashes/time bombs
        fallback_source_srt_path.write_text(source_srt, encoding="utf-8")
        if status:
            status(
                "Skipped",
                f"Too many subtitle segments ({len(subs)}). "
                f"Saved source only: {fallback_source_srt_path.name}"
            )
        return Result(
            False,
            f"too many subtitle segments ({len(subs)}), saved source only",
            video_path.name,
            time.perf_counter() - t0,
        )

    if detected_lang == "en":
        if status:
            status("Finalize", f"Writing English SRT: {video_path.name}")
        final_srt_path.write_text(source_srt, encoding="utf-8")
        return Result(True, "english srt written", video_path.name, time.perf_counter() - t0)

    src_nllb = WHISPER_TO_NLLB.get(detected_lang)
    if not src_nllb:
        fallback_source_srt_path.write_text(source_srt, encoding="utf-8")
        if status:
            status("Translation skipped", f"Detected '{detected_lang}' but no mapping. Wrote source fallback.")
        return Result(
            False,
            f"no mapping for '{detected_lang}' (wrote source fallback)",
            video_path.name,
            time.perf_counter() - t0,
        )

    if status:
        status("Translate", f"Translating: {video_path.name} (detected {detected_lang})")

    translator = translator_cache.get(src_nllb)
    if translator is None:
        if status:
            status("Translate", f"Initializing language pipeline: {src_nllb}")
        translator = NllbTranslator(src_lang=src_nllb, tgt_lang="eng_Latn", device="cpu")
        translator_cache[src_nllb] = translator

    # If you still use batch_size-based translation, keep it. (Token-aware batching is better long-term.)
    english_srt = translator.translate_srt(source_srt, max_tokens=400)

    if status:
        status("Finalize", f"Writing English SRT: {video_path.name}")
    final_srt_path.write_text(english_srt, encoding="utf-8")
    fallback_source_srt_path.unlink(missing_ok=True)

    return Result(True, "translated to english", video_path.name, time.perf_counter() - t0)



def run_batch(
    videos: list[Path],
    whisper_model: str,
    existing_srt_mode: str,
    recursive: bool = False,
    status: StatusFn | None = None,
    progress: ProgressFn | None = None,
    translator_cache: dict | None = None,
    whisper_cache: dict | None = None,
) -> list[Result]:
    if translator_cache is None:
        translator_cache = {}
    if whisper_cache is None:
        whisper_cache = {}

    results: list[Result] = []

    t0 = time.perf_counter()
    total = len(videos)
    completed = 0

    for vid in videos:
        if progress:
            progress(completed, total, time.perf_counter() - t0)

        res = process_one_video(
            vid,
            whisper_model=whisper_model,
            existing_srt_mode=existing_srt_mode,
            translator_cache=translator_cache,
            whisper_cache=whisper_cache,
            status=status,
        )
        results.append(res)
        completed += 1

        if progress:
            progress(completed, total, time.perf_counter() - t0)

    return results
