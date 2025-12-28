from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from src.core import collect_videos, run_batch, chunked, filter_videos_for_run
from src.nllb_translate import warmup as nllb_warmup

from src.ui.settings_store import load_settings, save_settings
from src.ui.setup_frame import SetupFrame
from src.ui.progress_frame import ProgressFrame
from src.ui.summary_dialog import SummaryDialog


APP_VERSION = "1.0.0"  # bump whenever you release changes
WHISPER_MODEL = "medium"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# For big libraries: process in smaller groups for stability
DEFAULT_CHUNK_SIZE = 750
CHUNK_THRESHOLD = 1500  # if more than this, we chunk


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Subtitle Translator")
        self.geometry("720x520")

        self.settings = load_settings()
        self.uiq: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.cancel_flag = threading.Event()

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames: dict[str, ttk.Frame] = {}
        self.frames["SetupFrame"] = SetupFrame(container, controller=self)
        self.frames["ProgressFrame"] = ProgressFrame(container, controller=self)

        for f in self.frames.values():
            f.grid(row=0, column=0, sticky="nsew")

        self.show_frame("SetupFrame")
        self.after(100, self._poll_ui_events)

    def show_frame(self, name: str):
        self.frames[name].tkraise()

    def start_work(self, videos: list[Path], existing_srt_mode: str, english_output_mode: str, library_root: Path | None):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "Work is already running.")
            return

        self.cancel_flag.clear()
        self.show_frame("ProgressFrame")
        self.frames["ProgressFrame"].reset()

        def status(stage: str, detail: str):
            self.uiq.put(("status", stage, detail))

        def progress(done: int, total: int, elapsed: float, done_work: int, total_work: int, did_work: bool):
            self.uiq.put(("progress", done, total, elapsed, done_work, total_work, did_work))

        def worker():
            try:
                status("Init", "Warming up translation model…")
                try:
                    nllb_warmup("cpu")
                except Exception:
                    pass

                # Fast prefilter for skip mode (huge resume runs)
                videos2 = filter_videos_for_run(videos, existing_srt_mode)
                total_all = len(videos2)

                if total_all == 0:
                    self.uiq.put(("done", "Total files: 0", ["SKIP: Nothing to do (all .en.srt already exist)."], 0.0))
                    return

                # Run either single batch or chunked batches
                all_results = []
                global_done = 0
                start_perf = time.perf_counter()

                # Wrap progress so UI shows global counts
                def chunk_progress(done, total, elapsed, done_work, total_work, did_work):
                    nonlocal global_done
                    # Map chunk progress into global done count
                    # 'done' counts within chunk, so shift by global_done at chunk start
                    self.uiq.put(
                        ("progress", global_done + done, total_all, time.perf_counter() - start_perf, done_work, total_work, did_work)
                    )

                if total_all > CHUNK_THRESHOLD:
                    status("Init", f"Large run detected ({total_all} files). Processing in chunks of {DEFAULT_CHUNK_SIZE}…")

                    for part in chunked(videos2, DEFAULT_CHUNK_SIZE):
                        if self.cancel_flag.is_set():
                            status("Cancelled", "Stopping now.")
                            break

                        # IMPORTANT: progress wrapper uses global_done offset
                        results = run_batch(
                            videos=part,
                            whisper_model=WHISPER_MODEL,
                            existing_srt_mode=existing_srt_mode,
                            english_output_mode=english_output_mode,
                            status=status,
                            progress=chunk_progress,
                            should_cancel=self.cancel_flag.is_set,
                            library_root=library_root,
                            tool_version=APP_VERSION,
                            translator_model=NLLB_MODEL,
                        )
                        all_results.extend(results)
                        global_done += len(part)

                else:
                    # Normal run
                    all_results = run_batch(
                        videos=videos2,
                        whisper_model=WHISPER_MODEL,
                        existing_srt_mode=existing_srt_mode,
                        english_output_mode=english_output_mode,
                        status=status,
                        progress=progress,
                        should_cancel=self.cancel_flag.is_set,
                        library_root=library_root,
                        tool_version=APP_VERSION,
                        translator_model=NLLB_MODEL,
                    )

                # Build summary + totals (cap lines so UI doesn't melt on 15k files)
                translated = 0
                created = 0
                skipped = 0
                warned = 0
                lines: list[str] = []

                MAX_LINES = 5000

                for r in all_results:
                    msg_low = (r.message or "").lower()

                    is_skipped = ("skipped" in msg_low) or ("no speech" in msg_low)
                    is_translated = "translated to english" in msg_low
                    is_created = ("english srt written" in msg_low) or is_translated

                    if is_skipped:
                        skipped += 1
                        tag = "SKIP"
                    elif not r.ok:
                        warned += 1
                        tag = "WARN"
                    else:
                        tag = "OK"

                    if is_translated:
                        translated += 1
                    if is_created:
                        created += 1

                    if len(lines) < MAX_LINES:
                        lines.append(f"{tag} ({r.elapsed_s:.1f}s): {r.video}: {r.message}")

                if len(all_results) > MAX_LINES:
                    lines.append(f"SKIP: Summary truncated to {MAX_LINES} lines (processed {len(all_results)} files).")

                stats_line = (
                    f"Total files: {len(all_results)}    "
                    f"Files translated: {translated}    "
                    f"Subtitles created: {created}    "
                    f"Skipped: {skipped}"
                )
                elapsed_s = time.perf_counter() - start_perf

                self.uiq.put(("done", stats_line, lines, elapsed_s))

            except Exception as e:
                self.uiq.put(("error", str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def cancel_work(self):
        self.cancel_flag.set()

    def persist_settings(self):
        save_settings(self.settings)

    def _poll_ui_events(self):
        try:
            while True:
                ev = self.uiq.get_nowait()
                kind = ev[0]

                if kind == "status":
                    _, stage, detail = ev
                    self.frames["ProgressFrame"].set_status(stage, detail)

                elif kind == "progress":
                    _, done, total, elapsed, done_work, total_work, did_work = ev
                    self.frames["ProgressFrame"].set_progress(done, total, elapsed, done_work, total_work, did_work)

                elif kind == "done":
                    _, stats_line, lines, elapsed_s = ev
                    elapsed_str = self.frames["ProgressFrame"].fmt_hms(int(elapsed_s))
                    SummaryDialog(self, title="Summary", elapsed_str=elapsed_str, stats_line=stats_line, lines=lines)
                    self.show_frame("SetupFrame")

                elif kind == "error":
                    _, msg = ev
                    messagebox.showerror("Error", msg)
                    self.show_frame("SetupFrame")

        except queue.Empty:
            pass

        self.after(100, self._poll_ui_events)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
