from __future__ import annotations

import json
from pathlib import Path


SETTINGS_PATH = Path("settings.json")

DEFAULT_SETTINGS = {
    "mode": "single",  # single | batch
    "scan_subfolders": True,
    "existing_srt_mode": "skip",  # skip | overwrite
    "english_output_mode": "all",  # all | non_english_only
}


def load_settings() -> dict:
    try:
        if not SETTINGS_PATH.exists():
            return DEFAULT_SETTINGS.copy()
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8") or "{}")
        if not isinstance(data, dict):
            return DEFAULT_SETTINGS.copy()
        merged = DEFAULT_SETTINGS.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
