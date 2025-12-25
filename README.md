# Local Subtitle Translator (Whisper + NLLB)

Pick a video file, generate subtitles locally with Whisper, detect language, and if not English translate to English using NLLB-200. Outputs a single `VideoName.srt` next to the video.

## Requirements
- Windows 10/11
- Python 3.10+
- ffmpeg installed

## Install ffmpeg
PowerShell:
```powershell
winget install Gyan.FFmpeg
ffmpeg -version
```

## Setup
```poweshell
git clone <your-repo-url>
cd local-subtitle-translator
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```powershell
python main.py
```