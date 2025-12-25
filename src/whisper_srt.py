from __future__ import annotations

import whisper
import srt
from datetime import timedelta

def _td(seconds: float) -> timedelta:
    return timedelta(seconds=float(seconds))

def transcribe_to_srt(video_path: str, model_name: str = "medium") -> tuple[str, str]:
    """
    Returns (detected_language, srt_text)
    detected_language is a whisper lang code like 'en', 'ja', 'es'
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)

    detected_language = result.get("language", "unknown")

    subtitles = []
    for idx, seg in enumerate(result["segments"], start=1):
        subtitles.append(
            srt.Subtitle(
                index=idx,
                start=_td(seg["start"]),
                end=_td(seg["end"]),
                content=seg["text"].strip(),
            )
        )

    srt_text = srt.compose(subtitles)
    return detected_language, srt_text
