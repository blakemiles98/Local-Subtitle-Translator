from __future__ import annotations

import time
import threading
import queue
import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback

from src.core import collect_videos, process_one_video
from src.core import run_batch
from src.nllb_translate import warmup as nllb_warmup

WHISPER_MODEL = "medium"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

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
        return DEFAULT_SETTINGS.copy()

    settings = DEFAULT_SETTINGS.copy()
    settings.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
    return settings

def save_settings(settings: dict) -> None:
    try:
        with SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

@dataclass
class UiEvent:
    kind: str
    status: str = ""
    detail: str = ""
    current: int = 0
    total: int = 0
    summary: str = ""
    elapsed_s: float = 0.0
    done_bytes: int = 0
    total_bytes: int = 0

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

        self.translator_cache = {}
        self.whisper_cache = {}

        self.container = ttk.Frame(self, padding=14)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (SetupFrame, ProgressFrame):
            frame = F(parent=self.container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("SetupFrame")

        threading.Thread(target=self._warmup_models, daemon=True).start()

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

                def progress(done, total, elapsed, done_bytes, total_bytes):
                    self.uiq.put(
                        UiEvent(
                            kind="progress",
                            current=done,
                            total=total,
                            elapsed_s=elapsed,
                            done_bytes=done_bytes,
                            total_bytes=total_bytes,
                        )
                    )

                results = run_batch(
                    videos=videos,
                    whisper_model=WHISPER_MODEL,
                    existing_srt_mode=existing_srt_mode,
                    status=status,
                    progress=progress,
                    translator_cache=self.translator_cache,
                    whisper_cache=self.whisper_cache,
                    should_cancel=self.cancel_flag.is_set,   # NEW
                )

                was_cancelled = self.cancel_flag.is_set()

                lines = []
                for r in results[-25:]:
                    tag = "OK" if r.ok else "WARN"
                    low = r.message.lower()
                    if "skipped" in low or "no speech" in low:
                        tag = "SKIP"
                    lines.append(f"{tag} ({r.elapsed_s:.1f}s): {r.video}: {r.message}")

                summary = "\n".join(lines)
                elapsed = sum(r.elapsed_s for r in results)
                done_status = "Cancelled" if was_cancelled else "Done"
                done_detail = "Stopped by user." if was_cancelled else "Finished."

                self.uiq.put(
                    UiEvent(
                        kind="done",
                        summary=summary,
                        elapsed_s=(time.perf_counter() - batch_start),
                        status=done_status,
                        detail=done_detail,
                    )
                )
            except Exception:
                tb = traceback.format_exc()
                Path("error.log").write_text(tb, encoding="utf-8")
                self.uiq.put(UiEvent(kind="error", summary=tb))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def poll_ui_events(self):
        try:
            while True:
                ev: UiEvent = self.uiq.get_nowait()
                if ev.kind == "status":
                    self.frames["ProgressFrame"].set_status(ev.status, ev.detail)
                    if "SetupFrame" in self.frames:
                        self.frames["SetupFrame"].set_status(ev.status, ev.detail)
                elif ev.kind == "progress":
                    self.frames["ProgressFrame"].set_progress(
                        ev.current, ev.total, ev.elapsed_s, ev.done_bytes, ev.total_bytes
                    )
                elif ev.kind == "done":
                    self.frames["ProgressFrame"].set_status(ev.status or "Done", ev.detail or "Finished.")
                    self.frames["ProgressFrame"]._timer_running = False
                    secs = int(ev.elapsed_s)
                    h = secs // 3600
                    m = (secs % 3600) // 60
                    s = secs % 60
                    elapsed_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

                    lines = ev.summary.splitlines() if ev.summary else ["Done."]

                    SummaryDialog(
                        parent=self,
                        title="Summary",
                        elapsed_str=elapsed_str,
                        lines=lines,
                    )

                    self.show_frame("SetupFrame")
                elif ev.kind == "error":
                    messagebox.showerror("Error", ev.summary or "Unknown error")
                    self.show_frame("SetupFrame")
        except queue.Empty:
            pass
        self.after(100, self.poll_ui_events)
    
    def _warmup_models(self):
        try:
            t0 = time.perf_counter()
            self.uiq.put(UiEvent(kind="status", status="Warmup", detail="Loading translation model..."))

            from src.nllb_translate import warmup as nllb_warmup
            nllb_warmup(device="cpu")

            dt = time.perf_counter() - t0
            self.uiq.put(UiEvent(kind="status", status="Warmup", detail=f"Translation model loaded ({dt:.1f}s)."))
        except Exception as e:
            self.uiq.put(UiEvent(kind="status", status="Warmup", detail=f"Warmup failed: {e}"))

class SetupFrame(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        settings = controller.settings

        self.setup_status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.setup_status_var, wraplength=600).pack(anchor="w", pady=(10, 0))

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

    def set_status(self, status: str, detail: str):
        self.setup_status_var.set(f"{status}: {detail}".strip())

    def on_show(self):
        pass

    def on_start(self):
        self.controller.settings["mode"] = self.mode.get()
        self.controller.settings["scan_subfolders"] = self.scan_subfolders.get()
        self.controller.settings["existing_srt_mode"] = self.existing_srt_mode.get()
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

        self._batch_start_ts: float | None = None
        self._timer_running = False
        self._current = 0
        self._total = 1
        self._done_bytes = 0
        self._total_bytes = 0


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

        self.eta_var = tk.StringVar(value="ETA: estimating…")
        ttk.Label(self, textvariable=self.eta_var).pack(anchor="w", pady=(2, 0))

        self._last_eta_update_done = 0
        self._last_cp_bytes = 0
        self._last_cp_elapsed = 0.0
        self._bps_ewma: float | None = None
        self._ewma_alpha = 0.30

    def on_show(self):
        self.set_status("Starting…", "")
        self.set_progress(0, 1, 0.0, 0, 0)
        self.cancel_btn.state(["!disabled"])

        if not self._timer_running:
            self._batch_start_ts = time.perf_counter()
            self._timer_running = True
            self._tick()

    def set_total(self, total: int):
        self.bar["maximum"] = max(total, 1)
        self.bar["value"] = 0
        self.count_var.set(f"0 / {total}")

    def set_progress(
        self,
        current: int,
        total: int,
        elapsed_s: float = 0.0,
        done_bytes: int = 0,
        total_bytes: int = 0,
    ):
        self._current = current
        self._total = max(total, 1)
        self._done_bytes = max(done_bytes, 0)
        self._total_bytes = max(total_bytes, 0)

        self.bar["maximum"] = self._total
        self.bar["value"] = current
        self.count_var.set(f"{current} / {total}")

        if current <= 0:
            self.eta_var.set("ETA: estimating…")
            self._last_eta_update_done = 0
            self._last_cp_bytes = 0
            self._last_cp_elapsed = 0.0
            self._bps_ewma = None
            return

        if current == self._last_eta_update_done:
            return

        self._last_eta_update_done = current

        if current < 2:
            self.eta_var.set("ETA: estimating…")
            self._last_cp_bytes = self._done_bytes
            self._last_cp_elapsed = elapsed_s
            return

        if self._total_bytes <= 0:
            if self._total > current and elapsed_s > 0.5:
                avg = elapsed_s / current
                eta_s = avg * (self._total - current)
                self._set_eta_seconds(eta_s)
            else:
                self.eta_var.set("ETA: estimating…")
            return

        delta_bytes = self._done_bytes - self._last_cp_bytes
        delta_time = elapsed_s - self._last_cp_elapsed

        inst_bps = None
        if delta_bytes > 0 and delta_time > 0.1:
            inst_bps = delta_bytes / delta_time
        elif self._done_bytes > 0 and elapsed_s > 0.5:
            inst_bps = self._done_bytes / elapsed_s

        if inst_bps is None:
            self.eta_var.set("ETA: estimating…")
            return

        if self._bps_ewma is None:
            self._bps_ewma = inst_bps
        else:
            self._bps_ewma = self._ewma_alpha * inst_bps + (1 - self._ewma_alpha) * self._bps_ewma

        self._last_cp_bytes = self._done_bytes
        self._last_cp_elapsed = elapsed_s

        remaining = max(0, self._total_bytes - self._done_bytes)
        eta_s = remaining / self._bps_ewma if self._bps_ewma > 0 else None
        if eta_s is None:
            self.eta_var.set("ETA: estimating…")
        else:
            self._set_eta_seconds(eta_s)

    def _set_eta_seconds(self, eta_s: float):
        eta_secs = int(max(0, eta_s))
        eh = eta_secs // 3600
        em = (eta_secs % 3600) // 60
        es = eta_secs % 60
        eta_str = f"{eh}:{em:02d}:{es:02d}" if eh else f"{em}:{es:02d}"
        self.eta_var.set(f"ETA: {eta_str} remaining")

    def set_status(self, status: str, detail: str):
        self.status_var.set(status)
        self.detail_var.set(detail)

    def on_cancel(self):
        self.controller.cancel_flag.set()
        self.cancel_btn.state(["disabled"])
        self.set_status("Cancelling…", "Finishing current step and then stopping.")
        self._timer_running = False

    def _tick(self):
        if not self._timer_running:
            return

        now = time.perf_counter()
        start = self._batch_start_ts or now
        elapsed = max(0.0, now - start)

        secs = int(elapsed)
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        elapsed_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        self.time_var.set(f"Elapsed: {elapsed_str}")

        self.after(250, self._tick)

class SummaryDialog(tk.Toplevel):
    def __init__(self, parent, title: str, elapsed_str: str, lines: list[str]):
        super().__init__(parent)
        self.title(title)
        self.geometry("820x460")
        self.transient(parent)
        self.grab_set()

        ttk.Label(
            self,
            text=f"Total time: {elapsed_str}",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 6))

        text = tk.Text(self, wrap="word", height=20, bg="white", fg="black")
        text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        text.tag_configure(
            "OK",
            background="#2e7d32",   # green
            foreground="white"
        )
        text.tag_configure(
            "WARN",
            background="#ed6c02",   # orange
            foreground="white"
        )
        text.tag_configure(
            "SKIP",
            background="#1565c0",   # blue
            foreground="white"
        )
        text.tag_configure("TEXT", foreground="black")

        for line in lines:
            if line.startswith("OK"):
                status = "OK"
                rest = line[2:].lstrip()
            elif line.startswith("WARN"):
                status = "WARN"
                rest = line[4:].lstrip()
            elif line.startswith("SKIP"):
                status = "SKIP"
                rest = line[4:].lstrip()
            else:
                status = "INFO"
                rest = line

            text.insert("end", " ", "TEXT")

            if status in ("OK", "WARN", "SKIP"):
                text.insert("end", status, status)
            else:
                text.insert("end", status, "TEXT")

            text.insert("end", "  ", "TEXT")

            text.insert("end", f"{rest}\n", "TEXT")

        text.config(state="disabled")

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))

if __name__ == "__main__":
    App().mainloop()
