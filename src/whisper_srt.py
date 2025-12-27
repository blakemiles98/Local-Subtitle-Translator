from __future__ import annotations

from faster_whisper import WhisperModel
import srt
from datetime import timedelta

import subprocess
import tempfile
from pathlib import Path


def _td(seconds: float) -> timedelta:
    return timedelta(seconds=float(seconds))


def transcribe_to_srt(
    video_path: str,
    model_name: str = "medium",
    whisper_cache: dict | None = None,
) -> tuple[str, dict[str, float], str]:
    """
    Returns (detected_language, lang_probs, srt_text)

    Uses a cache so we don't reload the Whisper model for every file.
    Falls back to extracting a WAV with ffmpeg if container decoding fails.
    """
    if whisper_cache is None:
        whisper_cache = {}

    key = (model_name, "cpu", "int8")

    model = whisper_cache.get(key)
    if model is None:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        whisper_cache[key] = model

    def _do_transcribe(path: str):
        return model.transcribe(
            path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700),
            condition_on_previous_text=False,
        )

    # Try direct first, then WAV fallback
    try:
        segments, info = _do_transcribe(video_path)
    except Exception as e1:
        wav_path = None
        try:
            wav_path = _extract_audio_wav(video_path)
            segments, info = _do_transcribe(wav_path)
        except Exception as e2:
            raise RuntimeError(
                f"Whisper decode failed for '{video_path}'. "
                f"Direct error: {e1}. WAV fallback error: {e2}"
            ) from e2
        finally:
            if wav_path:
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass

    # From here down, info is ALWAYS defined (or we already raised)
    detected_lang = (getattr(info, "language", None) or "unknown")
    probs = {detected_lang: float(getattr(info, "language_probability", 0.0) or 0.0)}

    subs: list[srt.Subtitle] = []
    idx = 1
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        subs.append(
            srt.Subtitle(
                index=idx,
                start=_td(seg.start),
                end=_td(seg.end),
                content=text,
            )
        )
        idx += 1

    if not subs:
        return detected_lang, probs, ""

    return detected_lang, probs, srt.compose(subs)


def _extract_audio_wav(video_path: str) -> str:
    """
    Extracts audio to a temporary mono 16k WAV for robust decoding.
    Returns the temp wav path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    wav_path = tmp.name

    # -vn: no video, mono, 16k, PCM s16le
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",          # only show real errors
        "-fflags", "+genpts",   # regenerate timestamps
        "-err_detect", "ignore_err",
        "-i", video_path,
        "-map", "0:a:0",        # pick first audio stream explicitly
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        wav_path,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if proc.returncode != 0:
        # include stderr so you can see the actual ffmpeg reason
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}).\n\n{proc.stderr}")
