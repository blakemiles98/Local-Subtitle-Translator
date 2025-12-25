from __future__ import annotations

from faster_whisper import WhisperModel
import srt
from datetime import timedelta

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
    faster-whisper provides language + probability.
    """
    if whisper_cache is None:
        whisper_cache = {}

    # cache key includes model_name + device + compute_type
    key = (model_name, "cpu", "int8")

    model = whisper_cache.get(key)
    if model is None:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        whisper_cache[key] = model

    # beam_size: 1-2 faster, 5 better
    segments, info = model.transcribe(video_path, beam_size=2, vad_filter=True)

    detected_lang = info.language or "unknown"
    probs = {detected_lang: float(info.language_probability or 0.0)}

    subs = []
    for idx, seg in enumerate(segments, start=1):
        subs.append(
            srt.Subtitle(
                index=idx,
                start=_td(seg.start),
                end=_td(seg.end),
                content=seg.text.strip(),
            )
        )

    if not subs:
        return detected_lang, probs, ""

    return detected_lang, probs, srt.compose(subs)
