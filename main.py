from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from src.whisper_srt import transcribe_to_srt
from src.nllb_translate import NllbTranslator
from src.lang_map import WHISPER_TO_NLLB
from src.ui_progress import ProgressWindow
from src.ui_options import ask_scan_subfolders

WHISPER_MODEL = "medium"  # set to "small" for faster CPU runs

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

# English skip-translation heuristics:
EN_PROB_STRONG = 0.70
EN_PROB_SOFT = 0.55
TOP_GAP_SOFT = 0.15


def pick_video_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"), ("All files", "*.*")],
    )
    return file_path or None


def pick_folder() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select a folder (videos will be processed recursively)")
    return folder or None


def is_effectively_english(detected_lang: str, probs: dict[str, float]) -> tuple[bool, str]:
    """
    Returns (should_treat_as_english, reason)
    """
    if detected_lang == "en":
        return True, "Detected language is English."

    en_prob = probs.get("en", 0.0)
    top_lang = max(probs, key=probs.get) if probs else detected_lang
    top_prob = probs.get(top_lang, 0.0)

    # Strong English probability → treat as English even if top lang isn't 'en'
    if en_prob >= EN_PROB_STRONG:
        return True, f"English probability is high ({en_prob:.2f})."

    # Soft case: English close to top language
    if en_prob >= EN_PROB_SOFT and (top_prob - en_prob) <= TOP_GAP_SOFT:
        return True, f"English probability is close to top ({en_prob:.2f} vs {top_prob:.2f})."

    return False, f"Non-English likely (top={top_lang}:{top_prob:.2f}, en={en_prob:.2f})."


def process_one_video(video_path: Path, progress: ProgressWindow) -> tuple[bool, str]:
    """
    Returns (success, message)
    Ensures the final output is: <VideoName>.srt (English)
    If translation performed, deletes <VideoName>.source.srt
    """
    out_dir = video_path.parent
    base_name = video_path.stem

    final_srt_path = out_dir / f"{base_name}.srt"
    temp_source_srt_path = out_dir / f"{base_name}.source.srt"

    progress.set_status("Whisper", f"Transcribing: {video_path.name}")

    detected_lang, probs, source_srt = transcribe_to_srt(str(video_path), model_name=WHISPER_MODEL)
    temp_source_srt_path.write_text(source_srt, encoding="utf-8")

    treat_as_en, reason = is_effectively_english(detected_lang, probs)

    if treat_as_en:
        progress.set_status("Finalize", f"Keeping English subtitles: {video_path.name}\n({reason})")
        final_srt_path.write_text(source_srt, encoding="utf-8")
        temp_source_srt_path.unlink(missing_ok=True)
        return True, f"{video_path.name}: English SRT written ({reason})"

    # Translate
    src_nllb = WHISPER_TO_NLLB.get(detected_lang)
    if not src_nllb:
        progress.set_status(
            "Skipped translation",
            f"{video_path.name}\nDetected '{detected_lang}' but no NLLB mapping exists.\nKept: {temp_source_srt_path.name}"
        )
        return False, f"{video_path.name}: No NLLB mapping for '{detected_lang}' (kept source SRT)."

    progress.set_status("Translate", f"Translating to English: {video_path.name}\n(detected {detected_lang})")
    translator = NllbTranslator(src_lang=src_nllb, tgt_lang="eng_Latn", device="cpu")
    english_srt = translator.translate_srt(source_srt, batch_size=12)

    progress.set_status("Finalize", f"Writing English SRT: {video_path.name}")
    final_srt_path.write_text(english_srt, encoding="utf-8")

    # Remove source if translation happened
    temp_source_srt_path.unlink(missing_ok=True)
    return True, f"{video_path.name}: Translated to English and wrote {final_srt_path.name}"


def collect_videos(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    else:
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)



def run_with_progress(videos: list[Path]):
    progress = ProgressWindow("Local Subtitle Translator (Whisper + NLLB)")
    progress.set_total(len(videos))

    results: list[str] = []

    def worker():
        try:
            for idx, vid in enumerate(videos, start=1):
                if progress.cancelled:
                    results.append("Cancelled by user.")
                    break

                progress.step_to(idx - 1)
                progress.set_status(
                    "Processing",
                    f"{vid.name}",
                    count=f"{idx} / {len(videos)}"
                )

                ok, msg = process_one_video(vid, progress)
                results.append(("OK: " if ok else "WARN: ") + msg)

                progress.step_to(idx)

            # Done
            if not progress.cancelled:
                progress.set_status("Done", "Finished processing.", count="")
        except Exception as e:
            results.append(f"ERROR: {e}")
            progress.set_status("Error", str(e))
        finally:
            # show summary popup
            summary = "\n".join(results[-20:])  # last 20 lines
            messagebox.showinfo("Summary", summary if summary else "No results.")
            progress.close()

    threading.Thread(target=worker, daemon=True).start()
    progress.root.mainloop()


def choose_mode_and_start():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    choice = messagebox.askyesno(
        "Mode",
        "Yes = Batch mode (choose a folder)\nNo = Single video (choose a file)"
    )
    root.destroy()

    if choice:  # Batch
        folder = pick_folder()
        if not folder:
            return

        scan_subfolders = ask_scan_subfolders(default=True)
        if scan_subfolders is None:
            return  # user cancelled

        vids = collect_videos(Path(folder), recursive=scan_subfolders)
        if not vids:
            messagebox.showwarning("No videos found", "No video files found with the selected options.")
            return

        run_with_progress(vids)
    else:  # Single
        file_path = pick_video_file()
        if not file_path:
            return
        run_with_progress([Path(file_path)])


if __name__ == "__main__":
    choose_mode_and_start()
