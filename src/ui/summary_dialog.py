from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SummaryDialog(tk.Toplevel):
    def __init__(self, parent, title: str, elapsed_str: str, stats_line: str, lines: list[str]):
        super().__init__(parent)
        self.title(title)
        self.geometry("820x520")
        self.transient(parent)
        self.grab_set()

        ttk.Label(self, text=f"Total time: {elapsed_str}", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        if stats_line:
            ttk.Label(self, text=stats_line, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(0, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        text = tk.Text(body, wrap="word", height=20, bg="white", fg="black")
        text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
        scroll.pack(side="right", fill="y")
        text.configure(yscrollcommand=scroll.set)

        # tags
        text.tag_configure("OK", foreground="#1b5e20")
        text.tag_configure("WARN", foreground="#b26a00")
        text.tag_configure("SKIP", foreground="#1a237e")

        for line in lines:
            tag = "OK"
            if line.startswith("WARN"):
                tag = "WARN"
            elif line.startswith("SKIP"):
                tag = "SKIP"
            text.insert("end", line + "\n", tag)

        text.configure(state="disabled")

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 12))
