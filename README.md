# Local Subtitle Translator (Whisper + NLLB)

Pick a video file, generate subtitles locally with Whisper, detect language, and if not English translate to English using NLLB-200. Outputs a single `VideoName.srt` next to the video.

## Requirements
- Windows 10/11
- Python 3.12

## Initial Setup
```powershell
winget install Gyan.FFmpeg
ffmpeg -version
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```powershell
python app.py
```