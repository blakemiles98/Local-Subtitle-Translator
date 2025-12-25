from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.whisper_srt import transcribe_to_srt
from src.nllb_translate import NllbTranslator
from src.lang_map import WHISPER_TO_NLLB

WHISPER_MODEL = "medium"  # "small" for speed on CPU
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

# English skip-translation heuristics:
EN_PROB_STRONG = 0.70
EN_PROB_SOFT = 0.55
TOP_GAP_SOFT = 0.15

@dataclass
class UiEvent:
    kind: str                 # "status", "progress", "done", "error"
    status: str = ""
    detail: str = ""
    current: int = 0
    total: int = 0
    summary: str = ""
    elapsed_s: float = 0.0

def is_effectively_english(detected_lang: str, probs: dict[str, float]) -> tuple[bool, str]:
    if detected_lang == "en":
        return True, "Detected language is English."

    en_prob = probs.get("en", 0.0)
    top_lang = max(probs, key=probs.get) if probs else detected_lang
    top_prob = probs.get(top_lang, 0.0)

    if en_prob >= EN_PROB_STRONG:
        return True, f"English probability is high ({en_prob:.2f})."

    if en_prob >= EN_PROB_SOFT and (top_prob - en_prob) <= TOP_GAP_SOFT:
        return True, f"English probability is close to top ({en_prob:.2f} vs {top_prob:.2f})."

    return False, f"Non-English likely (top={top_lang}:{top_prob:.2f}, en={en_prob:.2f})."

def collect_videos(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        vids = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    else:
        vids = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(vids)

def process_one_video(video_path: Path, uiq: queue.Queue, cancel_flag: threading.Event, translator_cache: dict, whisper_cache: dict) -> tuple[bool, str]:
    out_dir = video_path.parent
    base_name = video_path.stem

    final_srt_path = out_dir / f"{base_name}.srt"
    fallback_source_srt_path = out_dir / f"{base_name}.source.srt"

    if cancel_flag.is_set():
        return False, f"{video_path.name}: cancelled"

    uiq.put(UiEvent(kind="status", status="Whisper", detail=f"Transcribing: {video_path.name}"))
    detected_lang, probs, source_srt = transcribe_to_srt(str(video_path), model_name=WHISPER_MODEL, whisper_cache=whisper_cache)

    # Decide if we should treat as English
    treat_as_en, reason = is_effectively_english(detected_lang, probs)

    if treat_as_en:
        uiq.put(UiEvent(kind="status", status="Finalize", detail=f"Writing English SRT: {video_path.name}\n({reason})"))
        final_srt_path.write_text(source_srt, encoding="utf-8")
        return True, f"{video_path.name}: English SRT written ({reason})"

    # Translate (only if we have mapping)
    src_nllb = WHISPER_TO_NLLB.get(detected_lang)
    if not src_nllb:
        # Can't translate → write source as fallback
        fallback_source_srt_path.write_text(source_srt, encoding="utf-8")
        uiq.put(UiEvent(
            kind="status",
            status="Translation skipped",
            detail=f"{video_path.name}\nDetected '{detected_lang}' but no NLLB mapping.\nKept source: {fallback_source_srt_path.name}"
        ))
        return False, f"{video_path.name}: no NLLB mapping for '{detected_lang}' (wrote source fallback)."

    if cancel_flag.is_set():
        return False, f"{video_path.name}: cancelled"

    try:
        uiq.put(UiEvent(kind="status", status="Translate", detail=f"Translating: {video_path.name}\n(detected {detected_lang})"))
        translator = translator_cache.get(src_nllb)
        if translator is None:
            uiq.put(UiEvent(kind="status", status="Translate", detail=f"Loading translator model: {src_nllb}"))
            translator = NllbTranslator(src_lang=src_nllb, tgt_lang="eng_Latn", device="cpu")
            translator_cache[src_nllb] = translator

        english_srt = translator.translate_srt(source_srt, batch_size=24)

        uiq.put(UiEvent(kind="status", status="Finalize", detail=f"Writing English SRT: {video_path.name}"))
        final_srt_path.write_text(english_srt, encoding="utf-8")

        # Optionally: if an old fallback source exists from a previous run, delete it
        fallback_source_srt_path.unlink(missing_ok=True)

        return True, f"{video_path.name}: translated to English ({final_srt_path.name})"
    except Exception as e:
        # Translation failed → write source as fallback so user still gets something
        fallback_source_srt_path.write_text(source_srt, encoding="utf-8")
        return False, f"{video_path.name}: translation failed ({e}). Wrote source fallback."



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Subtitle Translator (Whisper + NLLB)")
        self.geometry("640x360")
        self.resizable(False, False)

        self.uiq: queue.Queue[UiEvent] = queue.Queue()
        self.cancel_flag = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.container = ttk.Frame(self, padding=14)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (SetupFrame, ProgressFrame):
            frame = F(parent=self.container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("SetupFrame")

        self.after(100, self.poll_ui_events)

    def show_frame(self, name: str):
        frame = self.frames[name]
        frame.tkraise()
        frame.on_show()

    def start_work(self, videos: list[Path]):
        if not videos:
            messagebox.showwarning("No videos", "No videos selected.")
            return

        self.cancel_flag.clear()
        self.frames["ProgressFrame"].set_total(len(videos))
        self.show_frame("ProgressFrame")

        def worker():
            results = []
            total = len(videos)
            batch_start = time.perf_counter()
            translator_cache = {}
            whisper_cache = {}
            try:
                completed = 0

                for idx, vid in enumerate(videos, start=1):
                    if self.cancel_flag.is_set():
                        results.append("Cancelled by user.")
                        break

                    elapsed = time.perf_counter() - batch_start

                    # 0/1 while working
                    self.uiq.put(UiEvent(
                        kind="progress",
                        current=completed,
                        total=total,
                        elapsed_s=elapsed
                    ))
                    self.uiq.put(UiEvent(
                        kind="status",
                        status="Processing",
                        detail=f"{vid.name}"
                    ))

                    # OPTIONAL: per-file timing
                    file_start = time.perf_counter()

                    ok, msg = process_one_video(
                        vid, self.uiq, self.cancel_flag, translator_cache, whisper_cache
                    )

                    file_elapsed = time.perf_counter() - file_start

                    # OPTIONAL: include per-file time in summary
                    results.append(f"{'OK' if ok else 'WARN'} ({file_elapsed:.1f}s): {msg}")

                    # Mark completed AFTER finishing the file
                    completed += 1
                    elapsed = time.perf_counter() - batch_start

                    self.uiq.put(UiEvent(
                        kind="progress",
                        current=completed,
                        total=total,
                        elapsed_s=elapsed
                    ))

                summary = "\n".join(results[-25:])
                elapsed = time.perf_counter() - batch_start  # 2C
                self.uiq.put(UiEvent(kind="done", summary=summary, elapsed_s=elapsed))
            except Exception as e:
                self.uiq.put(UiEvent(kind="error", summary=str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def poll_ui_events(self):
        try:
            while True:
                ev: UiEvent = self.uiq.get_nowait()
                if ev.kind == "status":
                    self.frames["ProgressFrame"].set_status(ev.status, ev.detail)
                elif ev.kind == "progress":
                    self.frames["ProgressFrame"].set_progress(ev.current, ev.total, ev.elapsed_s)
                elif ev.kind == "done":
                    self.frames["ProgressFrame"].set_status("Done", "Finished.")
                    secs = int(ev.elapsed_s)
                    h = secs // 3600
                    m = (secs % 3600) // 60
                    s = secs % 60
                    elapsed_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

                    messagebox.showinfo("Summary", f"Total time: {elapsed_str}\n\n{ev.summary or 'Done.'}")
                    self.show_frame("SetupFrame")
                elif ev.kind == "error":
                    messagebox.showerror("Error", ev.summary or "Unknown error")
                    self.show_frame("SetupFrame")
        except queue.Empty:
            pass
        self.after(100, self.poll_ui_events)


class SetupFrame(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        self.mode = tk.StringVar(value="single")
        self.scan_subfolders = tk.BooleanVar(value=True)

        ttk.Label(self, text="Subtitle Generator + Translator", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(self, text="Generates subtitles with Whisper. If not English, translates to English using NLLB-200.").pack(anchor="w", pady=(6, 12))

        box = ttk.LabelFrame(self, text="Mode", padding=10)
        box.pack(fill="x")

        ttk.Radiobutton(box, text="Single video", variable=self.mode, value="single").pack(anchor="w")
        ttk.Radiobutton(box, text="Batch (folder)", variable=self.mode, value="batch").pack(anchor="w")

        self.sub_chk = ttk.Checkbutton(box, text="Scan subfolders (batch only)", variable=self.scan_subfolders)
        self.sub_chk.pack(anchor="w", pady=(6, 0))

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(16, 0))

        ttk.Button(btns, text="Start", command=self.on_start).pack(side="right")
        ttk.Button(btns, text="Quit", command=self.controller.destroy).pack(side="right", padx=(0, 8))

        self.hint = ttk.Label(self, text=f"Whisper model: {WHISPER_MODEL} (edit in app.py)")
        self.hint.pack(anchor="w", pady=(14, 0))

    def on_show(self):
        pass

    def on_start(self):
        mode = self.mode.get()

        if mode == "single":
            path = filedialog.askopenfilename(
                title="Select a video",
                filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"), ("All files", "*.*")]
            )
            if not path:
                return
            self.controller.start_work([Path(path)])
            return

        # batch
        folder = filedialog.askdirectory(title="Select a folder of videos")
        if not folder:
            return
        vids = collect_videos(Path(folder), recursive=self.scan_subfolders.get())
        if not vids:
            messagebox.showwarning("No videos found", "No videos found with the chosen options.")
            return
        self.controller.start_work(vids)


class ProgressFrame(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        self.status_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="")

        ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(self, textvariable=self.detail_var, wraplength=600).pack(anchor="w", pady=(8, 0))
        ttk.Label(self, textvariable=self.count_var).pack(anchor="w", pady=(8, 0))

        self.bar = ttk.Progressbar(self, mode="determinate")
        self.bar.pack(fill="x", pady=(14, 0))

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(14, 0))

        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.on_cancel)
        self.cancel_btn.pack(side="right")

        self.time_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.time_var).pack(anchor="w", pady=(6, 0))

    def on_show(self):
        self.set_status("Starting…", "")
        self.set_progress(0, 1)
        self.cancel_btn.state(["!disabled"])

    def set_total(self, total: int):
        self.bar["maximum"] = max(total, 1)
        self.bar["value"] = 0
        self.count_var.set(f"0 / {total}")

    def set_progress(self, current: int, total: int, elapsed_s: float = 0.0):
        self.bar["maximum"] = max(total, 1)
        self.bar["value"] = current
        self.count_var.set(f"{current} / {total}")

        # format elapsed nicely
        secs = int(elapsed_s)
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        if h:
            self.time_var.set(f"Elapsed: {h}:{m:02d}:{s:02d}")
        else:
            self.time_var.set(f"Elapsed: {m}:{s:02d}")



    def set_status(self, status: str, detail: str):
        self.status_var.set(status)
        self.detail_var.set(detail)

    def on_cancel(self):
        self.controller.cancel_flag.set()
        self.cancel_btn.state(["disabled"])
        self.set_status("Cancelling…", "Finishing current step and then stopping.")


if __name__ == "__main__":
    App().mainloop()
