from __future__ import annotations

import tkinter as tk
from tkinter import ttk

class ProgressWindow:
    def __init__(self, title: str = "Subtitle Translator"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("520x180")
        self.root.resizable(False, False)

        self.status_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="")

        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(frm, textvariable=self.detail_var, wraplength=490).pack(anchor="w", pady=(6, 0))
        ttk.Label(frm, textvariable=self.count_var).pack(anchor="w", pady=(6, 0))

        self.progress = ttk.Progressbar(frm, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", pady=(12, 0))

        self.cancelled = False
        self.btn = ttk.Button(frm, text="Cancel", command=self._cancel)
        self.btn.pack(anchor="e", pady=(10, 0))

    def _cancel(self):
        self.cancelled = True
        self.status_var.set("Cancelling…")
        self.btn.state(["disabled"])
        self.root.update_idletasks()

    def set_total(self, total: int):
        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = 0
        self.root.update_idletasks()

    def step_to(self, value: int):
        self.progress["value"] = value
        self.root.update_idletasks()

    def set_status(self, status: str, detail: str = "", count: str = ""):
        self.status_var.set(status)
        self.detail_var.set(detail)
        self.count_var.set(count)
        self.root.update_idletasks()

    def close(self):
        self.root.destroy()
