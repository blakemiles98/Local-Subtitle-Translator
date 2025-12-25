from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from src.whisper_srt import transcribe_to_srt
from src.nllb_translate import NllbTranslator
from src.lang_map import WHISPER_TO_NLLB


WHISPER_MODEL = "medium"   # change to "small" for speed on CPU
TARGET_IS_ENGLISH = True


def pick_video_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
            ("All files", "*.*"),
        ],
    )
    return file_path or None


def main():
    video = pick_video_file()
    if not video:
        return

    video_path = Path(video)
    out_dir = video_path.parent
    base_name = video_path.stem

    # Final output always named like the video:
    final_srt_path = out_dir / f"{base_name}.srt"

    # Temp source SRT (only kept if no translation performed)
    temp_source_srt_path = out_dir / f"{base_name}.source.srt"

    try:
        detected_lang, source_srt = transcribe_to_srt(str(video_path), model_name=WHISPER_MODEL)

        # Write source subtitles first (temp)
        temp_source_srt_path.write_text(source_srt, encoding="utf-8")

        # If English, keep as final and remove temp naming
        if detected_lang == "en":
            final_srt_path.write_text(source_srt, encoding="utf-8")
            if temp_source_srt_path.exists():
                temp_source_srt_path.unlink(missing_ok=True)

            messagebox.showinfo(
                "Done",
                f"Detected language: English\nWrote: {final_srt_path}"
            )
            return

        # Not English -> translate using NLLB
        src_nllb = WHISPER_TO_NLLB.get(detected_lang)
        if not src_nllb:
            # If we can't map it, leave the source and explain
            messagebox.showwarning(
                "Translation skipped",
                f"Detected language: {detected_lang}\n"
                f"No NLLB mapping for this language yet.\n"
                f"Kept source subtitles as: {temp_source_srt_path}"
            )
            return

        translator = NllbTranslator(src_lang=src_nllb, tgt_lang="eng_Latn", device="cpu")
        english_srt = translator.translate_srt(source_srt, batch_size=12)

        final_srt_path.write_text(english_srt, encoding="utf-8")

        # remove source SRT so only English remains
        temp_source_srt_path.unlink(missing_ok=True)

        messagebox.showinfo(
            "Done",
            f"Detected language: {detected_lang}\nTranslated to English\nWrote: {final_srt_path}"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    main()
