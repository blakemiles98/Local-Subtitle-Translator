from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk


class ProgressFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self._start_perf = time.perf_counter()

        # ETA learning state
        self._last_eta_learn_done = -1  # only advances on did_work=True
        self._last_cp_work = 0
        self._last_cp_elapsed = 0.0
        self._wps_ewma: float | None = None

        self.status_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="")
        self.elapsed_var = tk.StringVar(value="Elapsed: 0:00")
        self.eta_var = tk.StringVar(value="ETA: estimating…")
        self.duration_var = tk.StringVar(value="Duration: 0:00 done / 0:00 left (of 0:00)")

        # ---- GRID LAYOUT ----
        self.columnconfigure(0, weight=1)

        title = ttk.Label(self, text="Progress", font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 10))

        ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 12, "bold")).grid(
            row=1, column=0, sticky="w", padx=12
        )

        detail_lbl = ttk.Label(self, textvariable=self.detail_var, wraplength=680, justify="left")
        detail_lbl.grid(row=2, column=0, sticky="w", padx=12, pady=(2, 8))

        ttk.Label(self, textvariable=self.count_var).grid(row=3, column=0, sticky="w", padx=12)
        ttk.Label(self, textvariable=self.elapsed_var).grid(row=4, column=0, sticky="w", padx=12, pady=(10, 0))
        ttk.Label(self, textvariable=self.eta_var).grid(row=5, column=0, sticky="w", padx=12, pady=(2, 0))
        ttk.Label(self, textvariable=self.duration_var).grid(row=6, column=0, sticky="w", padx=12, pady=(2, 10))

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.grid(row=7, column=0, sticky="ew", padx=12, pady=(0, 10))

        btns = ttk.Frame(self)
        btns.grid(row=8, column=0, sticky="ew", padx=12, pady=(0, 12))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        ttk.Button(btns, text="Cancel", command=self.controller.cancel_work).grid(row=0, column=0, sticky="w")
        ttk.Button(btns, text="Back to Setup", command=lambda: controller.show_frame("SetupFrame")).grid(
            row=0, column=1, sticky="e"
        )

        def _on_resize(_evt):
            w = max(300, self.winfo_width() - 40)
            detail_lbl.configure(wraplength=w)

        self.bind("<Configure>", _on_resize)

    def reset(self):
        self._start_perf = time.perf_counter()

        self._last_eta_learn_done = -1
        self._last_cp_work = 0
        self._last_cp_elapsed = 0.0
        self._wps_ewma = None

        self.status_var.set("Starting…")
        self.detail_var.set("")
        self.count_var.set("")
        self.elapsed_var.set("Elapsed: 0:00")
        self.eta_var.set("ETA: estimating…")
        self.duration_var.set("Duration: 0:00 done / 0:00 left (of 0:00)")
        self.progress["value"] = 0
        self.progress["maximum"] = 1

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start_perf

    def set_status(self, status: str, detail: str):
        self.status_var.set(status)
        self.detail_var.set(detail)

    def set_progress(self, done: int, total: int, elapsed_s: float, done_work: int, total_work: int, did_work: bool):
        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = min(max(done, 0), max(total, 1))

        self.count_var.set(f"Files: {done}/{total}")
        self.elapsed_var.set(f"Elapsed: {self.fmt_hms(int(elapsed_s))}")

        if total_work > 0:
            remaining = max(0, total_work - done_work)
            self.duration_var.set(
                f"Duration: {self.fmt_hms(done_work)} done / {self.fmt_hms(remaining)} left (of {self.fmt_hms(total_work)})"
            )
        else:
            self.duration_var.set(f"Duration: {self.fmt_hms(done_work)} done")

        # ---- ETA behavior ----
        # If we don't have an estimate yet:
        if self._wps_ewma is None:
            if done >= 0:
                self.eta_var.set("ETA: estimating…")
            elif not did_work:
                self.eta_var.set("ETA: waiting for real processing…")
            # else: did_work=True will compute below
        else:
            # We DO have an estimate. Don't overwrite it just because the current event is did_work=False.
            # (Skip/pre-file ticks shouldn't clobber ETA.)
            if not did_work:
                return

        # Only learn on did_work=True
        if not did_work:
            return

        if done == self._last_eta_learn_done:
            return
        self._last_eta_learn_done = done

        if done_work <= 0 or total_work <= 0:
            self.eta_var.set("ETA: estimating…")
            return

        delta_work = done_work - self._last_cp_work
        delta_t = elapsed_s - self._last_cp_elapsed

        if delta_work > 0 and delta_t > 0.5:
            inst_wps = delta_work / delta_t
            if self._wps_ewma is None:
                self._wps_ewma = inst_wps
            else:
                alpha = 0.35
                self._wps_ewma = (alpha * inst_wps) + ((1 - alpha) * self._wps_ewma)

            self._last_cp_work = done_work
            self._last_cp_elapsed = elapsed_s

        if not self._wps_ewma or self._wps_ewma <= 0:
            self.eta_var.set("ETA: estimating…")
            return

        remaining = max(0, total_work - done_work)
        eta_s = int(remaining / self._wps_ewma)
        self.eta_var.set(f"ETA: {self.fmt_hms(eta_s)}")

    def fmt_hms(self, seconds: int) -> str:
        seconds = int(max(0, seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
