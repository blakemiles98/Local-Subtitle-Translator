from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import srt

from src.lang_map import WHISPER_TO_NLLB
from src.nllb_translate import NllbTranslator
from src.whisper_srt import transcribe_to_srt

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

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
    txt = json.dumps(data, ensure_ascii=False, indent=2)
    tmp.write_text(txt, encoding="utf-8")
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

def _did_work_from_result(r: Result) -> bool:
    """
    True when the file actually performed meaningful work (whisper/translate/write),
    not when it was skipped due to cache/existing srt/translate-only mode.
    """
    msg = (r.message or "").lower()

    if "skipped" in msg:
        return False
    if "no speech detected" in msg or "no-speech cached" in msg:
        return False

    # Treat these as real work outcomes:
    if "translated to english" in msg:
        return True
    if "english srt written" in msg:
        return True
    if "wrote source fallback" in msg or "saved source only" in msg:
        return True

    # Default: if it wasn't a skip and it was ok, count it as work
    return bool(r.ok)


def _best_effort(fn: Callable[[], None], *, note: str = "") -> None:
    """
    Run fn() and ignore any exception.
    Used for non-critical cache/cleanup operations (meta json, marker updates, optional unlinks).
    """
    try:
        fn()
    except Exception:
        # Best-effort: failures intentionally ignored.
        return

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
ProgressFn = Callable[[int, int, float, int, int, bool], None]
CancelFn = Callable[[], bool]


def process_one_video(
    video_path: Path,
    whisper_model: str,
    existing_srt_mode: str,
    english_output_mode: str,
    translator_cache: dict,
    whisper_cache: dict,
    status: StatusFn | None = None,
) -> Result:
    t0 = time.perf_counter()

    out_dir = video_path.parent
    meta = _meta_path(video_path)
    base_name = video_path.stem

    final_srt_path = out_dir / f"{base_name}.en.srt"
    fallback_source_srt_path = out_dir / f"{base_name}.source.srt"

    MAX_SUBS = 8000

    # ---- Skip rescans if we previously detected no speech and the file hasn't changed ----
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

    # ---- overwrite behavior ----
    if existing_srt_mode == "overwrite":
        fallback_source_srt_path.unlink(missing_ok=True)
        _best_effort(lambda: _update_meta(meta, {"system": {"no_speech": None}}), note="clear no_speech")

    # Existing .en.srt skip behavior
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

    # ---- No speech: write meta marker so we don't rescan next time ----
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
            ),
            note="write no_speech marker",
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
        if english_output_mode == "non_english_only":
            # translate-only mode: do not write .en.srt for English videos
            if existing_srt_mode == "overwrite":
                # Remove stale English subtitle if overwrite was requested
                final_srt_path.unlink(missing_ok=True)

            if status:
                status(
                    "Skipped",
                    f"English detected; not writing .en.srt (translate-only mode): {video_path.name}"
                )
            return Result(
                True,
                "skipped (english; translate-only mode)",
                video_path.name,
                time.perf_counter() - t0,
            )

        if status:
            status("Finalize", f"Writing English SRT: {video_path.name}")
        final_srt_path.write_text(source_srt, encoding="utf-8")

        # Clear/flip no_speech flag on success (prevents stale cache if file changes)
        _best_effort(
            lambda: _update_meta(
                meta,
                {"system": {"no_speech": {"detected": False, "file_sig": _file_sig(video_path)}}},
            ),
            note="clear no_speech on english success",
        )

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

    english_srt = translator.translate_srt(source_srt, max_tokens=400)

    if status:
        status("Finalize", f"Writing English SRT: {video_path.name}")
    final_srt_path.write_text(english_srt, encoding="utf-8")
    fallback_source_srt_path.unlink(missing_ok=True)

    _best_effort(
        lambda: _update_meta(
            meta,
            {"system": {"no_speech": {"detected": False, "file_sig": _file_sig(video_path)}}},
        ),
        note="clear no_speech on translate success",
    )

    return Result(True, "translated to english", video_path.name, time.perf_counter() - t0)


def run_batch(
    videos: list[Path],
    whisper_model: str,
    existing_srt_mode: str,
    english_output_mode: str = "all",
    recursive: bool = False,
    status: StatusFn | None = None,
    progress: ProgressFn | None = None,
    translator_cache: dict | None = None,
    whisper_cache: dict | None = None,
    should_cancel: CancelFn | None = None,
) -> list[Result]:
    if translator_cache is None:
        translator_cache = {}
    if whisper_cache is None:
        whisper_cache = {}

    results: list[Result] = []

    t0 = time.perf_counter()
    completed = 0

    durations: list[float] = []
    total_videos = len(videos)

    fail_count = 0
    scan_last_ui = 0.0

    for i, v in enumerate(videos, 1):
        if should_cancel and should_cancel():
            if status:
                status("Cancelled", "Stopped while scanning media durations.")
            return results

        now = time.perf_counter()
        if status and (i == 1 or (now - scan_last_ui) >= 0.75):
            status("Scanning", f"Reading media info ({i}/{total_videos})…")
            scan_last_ui = now

        dur, ok = get_media_duration_seconds(v)
        if not ok:
            fail_count += 1
        durations.append(dur)

    if status and fail_count:
        if fail_count >= max(3, total_videos // 10):
            status(
                "Warning",
                f"Couldn't read duration for {fail_count}/{total_videos} files. "
                f"ETA may be less accurate for those."
            )
        else:
            status("Scanning", f"Duration read failed for {fail_count} file(s); continuing.")

    total_work = sum(durations)
    done_work = 0.0

    meta = _meta_path(v)
    _best_effort(
        lambda: _update_meta(meta, {"system": {"scan": {"last_scanned_utc": time.time()}}}),
        note="update last_scanned",
    )

    for i, vid in enumerate(videos):
        if should_cancel and should_cancel():
            if status:
                status("Cancelled", "Stopping after current file.")
            break

        if progress:
            progress(completed, total_videos, time.perf_counter() - t0, int(done_work), int(total_work), False)

        res = process_one_video(
            vid,
            whisper_model=whisper_model,
            existing_srt_mode=existing_srt_mode,
            english_output_mode=english_output_mode,
            translator_cache=translator_cache,
            whisper_cache=whisper_cache,
            status=status,
        )
        results.append(res)

        completed += 1
        done_work += durations[i]
        did_work = _did_work_from_result(res)

        if progress:
            progress(completed, total_videos, time.perf_counter() - t0, int(done_work), int(total_work), did_work)

        if should_cancel and should_cancel():
            if status:
                status("Cancelled", "Stopping now.")
            break

    return results


def _ffprobe_json(path: Path) -> dict:
    """
    Run ffprobe and return parsed JSON. Returns {} on failure.
    """
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
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
    """
    Cached minimal media data for your backend.

    Stored in <video>.<ext>.meta.json under:
      data: {
        file_sig,
        duration_s,
        container,
        video: {codec,width,height,fps},
        audio_tracks: [{codec,channels,language,default}],
        sub_tracks: [{codec,language,default,forced}]
      }

    Returns (data_dict, ok).
    """
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

    _best_effort(lambda: _update_meta(meta, {"data": parsed}), note="cache minimal media data")
    return parsed, True


def get_media_duration_seconds(path: Path) -> tuple[float, bool]:
    """
    Duration derived from cached minimal media data.
    """
    data, ok = get_media_data(path)
    if not ok:
        return 0.0, False
    dur = float((data or {}).get("duration_s") or 0.0)
    return dur, (dur > 0.0)
