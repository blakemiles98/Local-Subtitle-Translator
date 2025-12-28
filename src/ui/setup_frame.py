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

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(10, 0))

        ttk.Button(btns, text="Start…", command=self.on_start).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.controller.destroy).pack(side="right")

    def on_start(self):
        # persist
        self.controller.settings["mode"] = self.mode.get()
        self.controller.settings["scan_subfolders"] = bool(self.scan_subfolders.get())
        self.controller.settings["existing_srt_mode"] = self.existing_srt_mode.get()
        self.controller.settings["english_output_mode"] = self.english_output_mode.get()
        self.controller.persist_settings()

        mode = self.mode.get()
        existing = self.existing_srt_mode.get()
        english_out = self.english_output_mode.get()

        if mode == "single":
            path = filedialog.askopenfilename(title="Select video file")
            if not path:
                return
            video = Path(path)
            self.controller.start_work([video], existing, english_out, library_root=video.parent)
            return

        # batch
        folder = filedialog.askdirectory(title="Select folder")
        if not folder:
            return
        root = Path(folder)
        vids = self._collect_videos(root, recursive=bool(self.scan_subfolders.get()))
        if not vids:
            messagebox.showinfo("No videos", "No video files found in this folder.")
            return
        self.controller.start_work(vids, existing, english_out, library_root=root)

    def _collect_videos(self, folder: Path, recursive: bool) -> list[Path]:
        from src.core import collect_videos
        return collect_videos(folder, recursive=recursive)
