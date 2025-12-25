from __future__ import annotations

import whisper
import srt
from datetime import timedelta

def _td(seconds: float) -> timedelta:
    return timedelta(seconds=float(seconds))

def detect_language_with_probs(model, video_path: str) -> tuple[str, dict[str, float]]:
    """
    Uses Whisper's language detection to return:
      (top_language_code, {lang_code: prob})
    """
    audio = whisper.load_audio(video_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    top_lang = max(probs, key=probs.get)
    return top_lang, probs

def transcribe_to_srt(video_path: str, model_name: str = "medium") -> tuple[str, dict[str, float], str]:
    """
    Returns (detected_language, lang_probs, srt_text)
    detected_language is a whisper lang code like 'en', 'ja', 'es'
    """
    model = whisper.load_model(model_name)

    detected_language, probs = detect_language_with_probs(model, video_path)

    # Hint language for transcription (helps stability/speed)
    result = model.transcribe(video_path, language=detected_language)

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
    return detected_language, probs, srt_text
