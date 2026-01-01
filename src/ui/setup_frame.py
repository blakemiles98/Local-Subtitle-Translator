from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class SetupFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        s = controller.settings

        self.mode = tk.StringVar(value=s.get("mode", "single"))
        self.scan_subfolders = tk.BooleanVar(value=bool(s.get("scan_subfolders", True)))
        self.existing_srt_mode = tk.StringVar(value=s.get("existing_srt_mode", "skip"))
        self.english_output_mode = tk.StringVar(value=s.get("english_output_mode", "all"))

        # UI state
        self.busy_var = tk.StringVar(value="")
        self._busy = False

        ttk.Label(self, text="Subtitle Translator", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(10, 10))

        # Mode
        modes = ttk.LabelFrame(self, text="Mode", padding=10)
        modes.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(modes, text="Single file", variable=self.mode, value="single").pack(anchor="w")
        ttk.Radiobutton(modes, text="Batch folder", variable=self.mode, value="batch").pack(anchor="w")

        # Existing SRT
        ex = ttk.LabelFrame(self, text="Existing .en.srt behavior", padding=10)
        ex.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(ex, text="Skip if .en.srt exists", variable=self.existing_srt_mode, value="skip").pack(anchor="w")
        ttk.Radiobutton(ex, text="Overwrite", variable=self.existing_srt_mode, value="overwrite").pack(anchor="w")

        # English output
        out = ttk.LabelFrame(self, text="English output", padding=10)
        out.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(out, text="Generate English subtitles for all videos", variable=self.english_output_mode, value="all").pack(anchor="w")
        ttk.Radiobutton(out, text="Only generate English subtitles for non-English videos", variable=self.english_output_mode, value="non_english_only").pack(anchor="w")

        # Batch option
        batch = ttk.LabelFrame(self, text="Batch options", padding=10)
        batch.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(batch, text="Scan subfolders", variable=self.scan_subfolders).pack(anchor="w")

        # Busy indicator (hidden until start)
        busy_row = ttk.Frame(self)
        busy_row.pack(fill="x", pady=(6, 0))
        self.spinner = ttk.Progressbar(busy_row, mode="indeterminate", length=140)
        self.busy_label = ttk.Label(busy_row, textvariable=self.busy_var)
        # not packed yet; only shown when busy

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(10, 0))

        self.start_btn = ttk.Button(btns, text="Start…", command=self.on_start)
        self.start_btn.pack(side="left")
        ttk.Button(btns, text="Quit", command=self.controller.destroy).pack(side="right")

    def set_busy(self, busy: bool, message: str = ""):
        # Simple UI lock + indicator
        self._busy = busy
        if busy:
            self.busy_var.set(message or "Preparing…")
            self.start_btn.configure(state="disabled")

            # show spinner/label if not already visible
            if not self.spinner.winfo_ismapped():
                self.spinner.pack(side="left")
                self.busy_label.pack(side="left", padx=(10, 0))

            self.spinner.start(10)
        else:
            self.busy_var.set("")
            self.start_btn.configure(state="normal")

            if self.spinner.winfo_ismapped():
                self.spinner.stop()
                self.spinner.pack_forget()
                self.busy_label.pack_forget()

    def on_start(self):
        if self._busy:
            return

        # persist
        self.controller.settings["mode"] = self.mode.get()
        self.controller.settings["scan_subfolders"] = bool(self.scan_subfolders.get())
        self.controller.settings["existing_srt_mode"] = self.existing_srt_mode.get()
        self.controller.settings["english_output_mode"] = self.english_output_mode.get()
        self.controller.persist_settings()

        mode = self.mode.get()
        existing = self.existing_srt_mode.get()
        english_out = self.english_output_mode.get()

        try:
            if mode == "single":
                path = filedialog.askopenfilename(title="Select video file")
                if not path:
                    return
                video = Path(path)

                # show immediate feedback
                self.set_busy(True, "Starting…")
                self.after(50, lambda: self.controller.start_work([video], existing, english_out, library_root=video.parent))
                return

            # batch
            folder = filedialog.askdirectory(title="Select folder")
            if not folder:
                return
            root = Path(folder)

            # show immediate feedback BEFORE scanning, so UI doesn’t look frozen
            self.set_busy(True, "Scanning folder… This may take several minutes for large batches…")
            self.after(
                50,
                lambda: self._start_batch(root, existing, english_out),
            )
        except Exception as e:
            self.set_busy(False)
            messagebox.showerror("Error", str(e))

    def _start_batch(self, root: Path, existing: str, english_out: str):
        try:
            vids = self._collect_videos(root, recursive=bool(self.scan_subfolders.get()))
            if not vids:
                self.set_busy(False)
                messagebox.showinfo("No videos", "No video files found in this folder.")
                return

            # hand off to controller (worker thread will handle heavy work)
            self.controller.start_work(vids, existing, english_out, library_root=root)
        finally:
            # SetupFrame is going to be hidden, but if user comes back, unlock it.
            self.set_busy(False)

    def _collect_videos(self, folder: Path, recursive: bool) -> list[Path]:
        from src.core import collect_videos
        return collect_videos(folder, recursive=recursive)
