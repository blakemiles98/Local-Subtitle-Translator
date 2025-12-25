from __future__ import annotations
import tkinter as tk
from tkinter import ttk

def ask_scan_subfolders(default: bool = True) -> bool | None:
    """
    Returns:
      True  -> scan subfolders
      False -> only top-level
      None  -> user cancelled
    """
    result = {"value": None}

    root = tk.Tk()
    root.title("Batch Options")
    root.geometry("300x140")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    var = tk.BooleanVar(value=default)

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Batch scan options", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    ttk.Checkbutton(frm, text="Scan subfolders", variable=var).pack(anchor="w", pady=(8, 0))

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(14, 0))

    def ok():
        result["value"] = var.get()
        root.destroy()

    def cancel():
        result["value"] = None
        root.destroy()

    ttk.Button(btns, text="Start", command=ok).pack(side="right")
    ttk.Button(btns, text="Cancel", command=cancel).pack(side="right", padx=(0, 8))

    root.mainloop()
    return result["value"]
