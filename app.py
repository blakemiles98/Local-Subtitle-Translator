from __future__ import annotations

import time
import threading
import queue
import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.core import collect_videos, process_one_video
from src.core import run_batch

WHISPER_MODEL = "medium"  # "small" for speed on CPU
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

# English skip-translation heuristics:
EN_PROB_STRONG = 0.70
EN_PROB_SOFT = 0.55
TOP_GAP_SOFT = 0.15

SETTINGS_FILE = Path("settings.json")

DEFAULT_SETTINGS = {
    "mode": "single",
    "scan_subfolders": True,
    "existing_srt_mode": "skip",
}

def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()

    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Corrupt or unreadable → fall back safely
        return DEFAULT_SETTINGS.copy()

    # Merge with defaults so missing keys don’t break anything
    settings = DEFAULT_SETTINGS.copy()
    settings.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
    return settings

def save_settings(settings: dict) -> None:
    try:
        with SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        # Non-fatal; ignore write errors
        pass

@dataclass
class UiEvent:
    kind: str                 # "status", "progress", "done", "error"
    status: str = ""
    detail: str = ""
    current: int = 0
    total: int = 0
    summary: str = ""
    elapsed_s: float = 0.0

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Subtitle Translator (Whisper + NLLB)")
        self.geometry("640x360")
        self.resizable(False, False)

        self.uiq: queue.Queue[UiEvent] = queue.Queue()
        self.cancel_flag = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.settings = load_settings()

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

    def start_work(self, videos: list[Path], existing_srt_mode: str):
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
                def status(s, d):
                    self.uiq.put(UiEvent(kind="status", status=s, detail=d))

                def progress(done, total, elapsed):
                    self.uiq.put(UiEvent(kind="progress", current=done, total=total, elapsed_s=elapsed))

                results = run_batch(
                    videos=videos,
                    whisper_model=WHISPER_MODEL,
                    existing_srt_mode=existing_srt_mode,
                    status=status,
                    progress=progress,
                )

                # build summary
                lines = []
                for r in results[-25:]:
                    tag = "OK" if r.ok else "WARN"
                    low = r.message.lower()
                    if "skipped" in low or "no speech" in low:
                        tag = "SKIP"
                    lines.append(f"{tag} ({r.elapsed_s:.1f}s): {r.video}: {r.message}")

                summary = "\n".join(lines)
                elapsed = sum(r.elapsed_s for r in results)  # or track elapsed in progress already
                self.uiq.put(UiEvent(kind="done", summary=summary, elapsed_s=(time.perf_counter() - batch_start)))
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

        settings = controller.settings

        self.mode = tk.StringVar(value=settings.get("mode", "single"))
        self.scan_subfolders = tk.BooleanVar(value=settings.get("scan_subfolders", True))
        self.existing_srt_mode = tk.StringVar(value=settings.get("existing_srt_mode", "skip"))

        ttk.Label(self, text="Subtitle Generator + Translator", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(self, text="Generates subtitles with Whisper. If not English, translates to English using NLLB-200.").pack(anchor="w", pady=(6, 12))

        box = ttk.LabelFrame(self, text="Mode", padding=10)
        box.pack(fill="x")

        ttk.Radiobutton(box, text="Single video", variable=self.mode, value="single").pack(anchor="w")
        ttk.Radiobutton(box, text="Batch (folder)", variable=self.mode, value="batch").pack(anchor="w")

        self.sub_chk = ttk.Checkbutton(box, text="Scan subfolders (batch only)", variable=self.scan_subfolders)
        self.sub_chk.pack(anchor="w", pady=(6, 0))

        opts = ttk.LabelFrame(self, text="Existing .srt behavior", padding=10)
        opts.pack(fill="x", pady=(10, 0))

        ttk.Radiobutton(
            opts,
            text="Skip videos that already have a .srt",
            variable=self.existing_srt_mode,
            value="skip",
        ).pack(anchor="w")

        ttk.Radiobutton(
            opts,
            text="Overwrite existing .srt files",
            variable=self.existing_srt_mode,
            value="overwrite",
        ).pack(anchor="w")

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(16, 0))

        ttk.Button(btns, text="Start", command=self.on_start).pack(side="right")
        ttk.Button(btns, text="Quit", command=self.controller.destroy).pack(side="right", padx=(0, 8))

        self.hint = ttk.Label(self, text=f"Whisper model: {WHISPER_MODEL} (edit in app.py)")
        self.hint.pack(anchor="w", pady=(14, 0))

    def on_show(self):
        pass

    def on_start(self):
        # Save current settings
        self.controller.settings = {
            "mode": self.mode.get(),
            "scan_subfolders": self.scan_subfolders.get(),
            "existing_srt_mode": self.existing_srt_mode.get(),
        }
        save_settings(self.controller.settings)
        
        mode = self.mode.get()

        if mode == "single":
            path = filedialog.askopenfilename(
                title="Select a video",
                filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"), ("All files", "*.*")]
            )
            if not path:
                return
            self.controller.start_work([Path(path)], existing_srt_mode=self.existing_srt_mode.get())
            return

        # batch
        folder = filedialog.askdirectory(title="Select a folder of videos")
        if not folder:
            return
        vids = collect_videos(Path(folder), recursive=self.scan_subfolders.get())
        if not vids:
            messagebox.showwarning("No videos found", "No videos found with the chosen options.")
            return
        self.controller.start_work(vids, existing_srt_mode=self.existing_srt_mode.get())

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
