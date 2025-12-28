from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import srt

from src.lang_map import WHISPER_TO_NLLB
from src.nllb_translate import NllbTranslator
from src.whisper_srt import transcribe_to_srt

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


# ----------------------------
# Meta JSON helpers
# ----------------------------
def _meta_path(video_path: Path) -> Path:
    return video_path.with_suffix(video_path.suffix + ".meta.json")


def _file_sig(p: Path) -> dict:
    st = p.stat()
    return {"size": st.st_size, "mtime": st.st_mtime}


def _read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _update_meta(path: Path, patch: dict) -> dict:
    doc = _read_json(path)
    if not isinstance(doc, dict):
        doc = {}
    _deep_merge(doc, patch)
    _atomic_write_json(path, doc)
    return doc


def _best_effort(fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception:
        return


def _ensure_media_id(meta_path: Path) -> str:
    doc = _read_json(meta_path)
    sys = doc.get("system") if isinstance(doc, dict) else None
    media_id = sys.get("media_id") if isinstance(sys, dict) else None
    if isinstance(media_id, str) and media_id.strip():
        return media_id

    new_id = str(uuid.uuid4())
    _best_effort(lambda: _update_meta(meta_path, {"system": {"media_id": new_id}}))
    return new_id


def _set_scan_fields(meta_path: Path, *, now: float) -> None:
    doc = _read_json(meta_path)
    sys = doc.get("system") if isinstance(doc, dict) else {}
    scan = sys.get("scan") if isinstance(sys, dict) else {}
    if not isinstance(scan, dict):
        scan = {}

    patch = {
        "system": {
            "scan": {
                "first_seen_utc": scan.get("first_seen_utc") or now,
                "last_seen_utc": now,
                "last_scanned_utc": now,
            }
        }
    }
    _best_effort(lambda: _update_meta(meta_path, patch))


def _set_library_relpath(meta_path: Path, video_path: Path, library_root: Path | None) -> None:
    if library_root is None:
        return
    try:
        rel = str(video_path.resolve().relative_to(library_root.resolve()))
    except Exception:
        return
    _best_effort(lambda: _update_meta(meta_path, {"system": {"library_relpath": rel}}))


# ----------------------------
# Fast skip helpers for huge libraries
# ----------------------------
def should_skip_fast(video_path: Path, existing_srt_mode: str) -> bool:
    """
    Fast skip without ffprobe/whisper.
    Only checks 'skip' mode + existing .en.srt.
    """
    if existing_srt_mode != "skip":
        return False
    final_srt = video_path.parent / f"{video_path.stem}.en.srt"
    return final_srt.exists()


def filter_videos_for_run(videos: list[Path], existing_srt_mode: str) -> list[Path]:
    """
    For huge libraries, do a fast prefilter to remove obvious skips.
    This massively reduces scan/probe work on 'resume' runs.
    """
    if existing_srt_mode != "skip":
        return videos
    return [v for v in videos if not should_skip_fast(v, existing_srt_mode)]


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ----------------------------
# Minimal media data (backend-friendly)
# ----------------------------
def _ffprobe_json(path: Path) -> dict:
    """
    IMPORTANT for Windows:
    - Force UTF-8 decoding and replace invalid bytes to avoid UnicodeDecodeError.
    """
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
        if p.returncode != 0:
            return {}
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _parse_fps(rate: str) -> float:
    try:
        if not rate:
            return 0.0
        if "/" in rate:
            a, b = rate.split("/", 1)
            return float(a) / float(b)
        return float(rate)
    except Exception:
        return 0.0


def _stream_language(stream: dict) -> str:
    tags = stream.get("tags") or {}
    lang = tags.get("language") or "und"
    return (lang or "und").strip() or "und"


def _stream_default(stream: dict) -> bool:
    disp = stream.get("disposition") or {}
    return bool(_to_int(disp.get("default", 0), 0))


def _stream_forced(stream: dict) -> bool:
    disp = stream.get("disposition") or {}
    return bool(_to_int(disp.get("forced", 0), 0))


def _parse_minimal_media_data(raw: dict, sig: dict) -> dict:
    fmt = raw.get("format") or {}
    streams = raw.get("streams") or []

    container = (fmt.get("format_name") or "").strip()
    duration_s = _to_float(fmt.get("duration"), 0.0)

    primary_video = None
    audio_tracks: list[dict] = []
    sub_tracks: list[dict] = []

    for st in streams:
        st_type = st.get("codec_type")

        if st_type == "video" and primary_video is None:
            primary_video = {
                "codec": (st.get("codec_name") or "").strip(),
                "width": _to_int(st.get("width"), 0),
                "height": _to_int(st.get("height"), 0),
                "fps": _parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate") or ""),
            }

        elif st_type == "audio":
            audio_tracks.append(
                {
                    "codec": (st.get("codec_name") or "").strip(),
                    "channels": _to_int(st.get("channels"), 0),
                    "language": _stream_language(st),
                    "default": _stream_default(st),
                }
            )

        elif st_type == "subtitle":
            sub_tracks.append(
                {
                    "codec": (st.get("codec_name") or "").strip(),
                    "language": _stream_language(st),
                    "default": _stream_default(st),
                    "forced": _stream_forced(st),
                }
            )

    if primary_video is None:
        primary_video = {"codec": "", "width": 0, "height": 0, "fps": 0.0}

    return {
        "file_sig": sig,
        "duration_s": duration_s,
        "container": container,
        "video": primary_video,
        "audio_tracks": audio_tracks,
        "sub_tracks": sub_tracks,
    }


def get_media_data(path: Path) -> tuple[dict, bool]:
    meta = _meta_path(path)
    sig = _file_sig(path)

    doc = _read_json(meta)
    data = doc.get("data") if isinstance(doc, dict) else None
    if isinstance(data, dict) and data.get("file_sig") == sig:
        return data, True

    raw = _ffprobe_json(path)
    if not raw:
        return {}, False

    parsed = _parse_minimal_media_data(raw, sig)
    _best_effort(lambda: _update_meta(meta, {"data": parsed}))
    return parsed, True


def get_media_duration_seconds(path: Path) -> tuple[float, bool]:
    data, ok = get_media_data(path)
    if not ok:
        return 0.0, False
    dur = float((data or {}).get("duration_s") or 0.0)
    return dur, (dur > 0.0)


# ----------------------------
# Core app logic
# ----------------------------
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
# (done_files, total_files, elapsed, done_work, total_work, did_work)
ProgressFn = Callable[[int, int, float, int, int, bool], None]
CancelFn = Callable[[], bool]


def _did_work_from_result(r: Result) -> bool:
    msg = (r.message or "").lower()
    if "skipped" in msg:
        return False
    if "no speech detected" in msg or "no-speech cached" in msg:
        return False
    if "translated to english" in msg:
        return True
    if "english srt written" in msg:
        return True
    if "wrote source fallback" in msg or "saved source only" in msg:
        return True
    return bool(r.ok)


def process_one_video(
    video_path: Path,
    whisper_model: str,
    existing_srt_mode: str,
    english_output_mode: str,
    translator_cache: dict,
    whisper_cache: dict,
    status: StatusFn | None = None,
    *,
    library_root: Path | None = None,
    tool_version: str = "dev",
    translator_model: str = "facebook/nllb-200-distilled-600M",
) -> Result:
    t0 = time.perf_counter()
    now = time.time()

    out_dir = video_path.parent
    meta = _meta_path(video_path)
    base_name = video_path.stem

    _ensure_media_id(meta)
    _set_scan_fields(meta, now=now)
    _set_library_relpath(meta, video_path, library_root)
    get_media_data(video_path)  # cache minimal media info best-effort

    final_srt_path = out_dir / f"{base_name}.en.srt"
    fallback_source_srt_path = out_dir / f"{base_name}.source.srt"

    MAX_SUBS = 8000

    # no-speech cached skip
    if existing_srt_mode == "skip":
        doc = _read_json(meta)
        sys = doc.get("system") if isinstance(doc, dict) else None
        no_speech = sys.get("no_speech") if isinstance(sys, dict) else None
        if (
            isinstance(no_speech, dict)
            and no_speech.get("detected") is True
            and no_speech.get("file_sig") == _file_sig(video_path)
        ):
            if status:
                status("Skipped", f"No-speech cached, skipping: {video_path.name}")
            return Result(True, "skipped (no-speech cached)", video_path.name, time.perf_counter() - t0)

    if existing_srt_mode == "overwrite":
        fallback_source_srt_path.unlink(missing_ok=True)
        _best_effort(lambda: _update_meta(meta, {"system": {"no_speech": None}}))

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

    # No speech -> cache and stop
    if not has_real_text(source_srt):
        _best_effort(
            lambda: _update_meta(
                meta,
                {
                    "system": {
                        "no_speech": {
                            "detected": True,
                            "whisper_model": whisper_model,
                            "created_utc": time.time(),
                            "file_sig": _file_sig(video_path),
                        }
                    }
                },
            )
        )
        if status:
            status("Skipped", f"No speech detected: {video_path.name}")
        return Result(False, "no srt created (no speech detected)", video_path.name, time.perf_counter() - t0)

    try:
        subs = list(srt.parse(source_srt))
    except Exception:
        subs = []

    if len(subs) > MAX_SUBS:
        fallback_source_srt_path.write_text(source_srt, encoding="utf-8")
        if status:
            status("Skipped", f"Too many subtitle segments ({len(subs)}). Saved source only.")
        return Result(False, f"too many subtitle segments ({len(subs)}), saved source only", video_path.name, time.perf_counter() - t0)

    if detected_lang == "en":
        if english_output_mode == "non_english_only":
            if existing_srt_mode == "overwrite":
                final_srt_path.unlink(missing_ok=True)

            if status:
                status("Skipped", f"English detected; not writing .en.srt (translate-only mode): {video_path.name}")

            _best_effort(
                lambda: _update_meta(
                    meta,
                    {
                        "system": {
                            "subtitle": {
                                "last_run_utc": time.time(),
                                "detected_lang": detected_lang,
                                "translated": False,
                                "wrote_en_srt": False,
                                "whisper_model": whisper_model,
                                "translator_model": translator_model,
                                "tool_version": tool_version,
                                "english_output_mode": english_output_mode,
                                "existing_srt_mode": existing_srt_mode,
                            },
                            "no_speech": {"detected": False, "file_sig": _file_sig(video_path)},
                        }
                    },
                )
            )

            return Result(True, "skipped (english; translate-only mode)", video_path.name, time.perf_counter() - t0)

        if status:
            status("Finalize", f"Writing English SRT: {video_path.name}")
        final_srt_path.write_text(source_srt, encoding="utf-8")

        _best_effort(
            lambda: _update_meta(
                meta,
                {
                    "system": {
                        "subtitle": {
                            "last_run_utc": time.time(),
                            "detected_lang": detected_lang,
                            "translated": False,
                            "wrote_en_srt": True,
                            "whisper_model": whisper_model,
                            "translator_model": translator_model,
                            "tool_version": tool_version,
                            "english_output_mode": english_output_mode,
                            "existing_srt_mode": existing_srt_mode,
                        },
                        "no_speech": {"detected": False, "file_sig": _file_sig(video_path)},
                    }
                },
            )
        )

        return Result(True, "english srt written", video_path.name, time.perf_counter() - t0)

    src_nllb = WHISPER_TO_NLLB.get(detected_lang)
    if not src_nllb:
        fallback_source_srt_path.write_text(source_srt, encoding="utf-8")
        if status:
            status("Translation skipped", f"Detected '{detected_lang}' but no mapping. Wrote source fallback.")

        _best_effort(
            lambda: _update_meta(
                meta,
                {
                    "system": {
                        "subtitle": {
                            "last_run_utc": time.time(),
                            "detected_lang": detected_lang,
                            "translated": False,
                            "wrote_en_srt": False,
                            "whisper_model": whisper_model,
                            "translator_model": translator_model,
                            "tool_version": tool_version,
                            "english_output_mode": english_output_mode,
                            "existing_srt_mode": existing_srt_mode,
                        },
                        "no_speech": {"detected": False, "file_sig": _file_sig(video_path)},
                    }
                },
            )
        )

        return Result(False, f"no mapping for '{detected_lang}' (wrote source fallback)", video_path.name, time.perf_counter() - t0)

    if status:
        status("Translate", f"Translating: {video_path.name} (detected {detected_lang})")

    translator = translator_cache.get(src_nllb)
    if translator is None:
        if status:
            status("Translate", f"Initializing language pipeline: {src_nllb}")
        translator = NllbTranslator(src_lang=src_nllb, tgt_lang="eng_Latn", device="cpu")
        translator_cache[src_nllb] = translator

    english_srt = translator.translate_srt(source_srt, max_tokens=400)

    if status:
        status("Finalize", f"Writing English SRT: {video_path.name}")
    final_srt_path.write_text(english_srt, encoding="utf-8")
    fallback_source_srt_path.unlink(missing_ok=True)

    _best_effort(
        lambda: _update_meta(
            meta,
            {
                "system": {
                    "subtitle": {
                        "last_run_utc": time.time(),
                        "detected_lang": detected_lang,
                        "translated": True,
                        "wrote_en_srt": True,
                        "whisper_model": whisper_model,
                        "translator_model": translator_model,
                        "tool_version": tool_version,
                        "english_output_mode": english_output_mode,
                        "existing_srt_mode": existing_srt_mode,
                    },
                    "no_speech": {"detected": False, "file_sig": _file_sig(video_path)},
                }
            },
        )
    )

    return Result(True, "translated to english", video_path.name, time.perf_counter() - t0)


def run_batch(
    videos: list[Path],
    whisper_model: str,
    existing_srt_mode: str,
    english_output_mode: str = "all",
    status: StatusFn | None = None,
    progress: ProgressFn | None = None,
    translator_cache: dict | None = None,
    whisper_cache: dict | None = None,
    should_cancel: CancelFn | None = None,
    *,
    library_root: Path | None = None,
    tool_version: str = "dev",
    translator_model: str = "facebook/nllb-200-distilled-600M",
) -> list[Result]:
    if translator_cache is None:
        translator_cache = {}
    if whisper_cache is None:
        whisper_cache = {}

    results: list[Result] = []
    videos = filter_videos_for_run(videos, existing_srt_mode)

    t0 = time.perf_counter()
    completed = 0

    durations: list[float] = []
    total_videos = len(videos)

    # duration + data caching scan
    for i, v in enumerate(videos, 1):
        if should_cancel and should_cancel():
            if status:
                status("Cancelled", "Stopped while scanning media info.")
            return results

        get_media_data(v)
        dur, ok = get_media_duration_seconds(v)
        durations.append(dur if ok else 0.0)

        _set_scan_fields(_meta_path(v), now=time.time())
        _set_library_relpath(_meta_path(v), v, library_root)
        _ensure_media_id(_meta_path(v))

        if status and (i == 1 or i % 50 == 0):
            status("Scanning", f"Reading media info ({i}/{total_videos})…")

    total_work = sum(durations)
    done_work = 0.0

    for i, vid in enumerate(videos):
        if should_cancel and should_cancel():
            if status:
                status("Cancelled", "Stopping after current file.")
            break

        if progress:
            progress(completed, total_videos, time.perf_counter() - t0, int(done_work), int(total_work), False)

        _set_scan_fields(_meta_path(vid), now=time.time())

        res = process_one_video(
            vid,
            whisper_model=whisper_model,
            existing_srt_mode=existing_srt_mode,
            english_output_mode=english_output_mode,
            translator_cache=translator_cache,
            whisper_cache=whisper_cache,
            status=status,
            library_root=library_root,
            tool_version=tool_version,
            translator_model=translator_model,
        )
        results.append(res)

        completed += 1
        done_work += durations[i]

        if progress:
            progress(
                completed,
                total_videos,
                time.perf_counter() - t0,
                int(done_work),
                int(total_work),
                _did_work_from_result(res),
            )

        if should_cancel and should_cancel():
            if status:
                status("Cancelled", "Stopping now.")
            break

    return results
