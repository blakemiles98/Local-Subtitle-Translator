from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk


class ProgressFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self._start_perf = time.perf_counter()
        self._last_eta_update_done = -1
        self._last_cp_work = 0
        self._last_cp_elapsed = 0.0
        self._wps_ewma: float | None = None

        self.status_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="")
        self.elapsed_var = tk.StringVar(value="Elapsed: 0:00")
        self.eta_var = tk.StringVar(value="ETA: estimating…")
        self.duration_var = tk.StringVar(value="Duration: 0:00 done / 0:00 left (of 0:00)")

        ttk.Label(self, text="Progress", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(10, 10))

        ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(self, textvariable=self.detail_var).pack(anchor="w", pady=(2, 8))
        ttk.Label(self, textvariable=self.count_var).pack(anchor="w")

        ttk.Label(self, textvariable=self.elapsed_var).pack(anchor="w", pady=(10, 0))
        ttk.Label(self, textvariable=self.eta_var).pack(anchor="w", pady=(2, 0))
        ttk.Label(self, textvariable=self.duration_var).pack(anchor="w", pady=(2, 0))

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", pady=(10, 10))

        btns = ttk.Frame(self)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=self.controller.cancel_work).pack(side="left")
        ttk.Button(btns, text="Back to Setup", command=lambda: controller.show_frame("SetupFrame")).pack(side="right")

    def reset(self):
        self._start_perf = time.perf_counter()
        self._last_eta_update_done = -1
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

        # duration line always updates
        if total_work > 0:
            remaining = max(0, total_work - done_work)
            self.duration_var.set(
                f"Duration: {self.fmt_hms(done_work)} done / {self.fmt_hms(remaining)} left (of {self.fmt_hms(total_work)})"
            )
        else:
            self.duration_var.set(f"Duration: {self.fmt_hms(done_work)} done")

        # Only update ETA on real work completions (avoids resume skip poisoning)
        if done == self._last_eta_update_done:
            return
        self._last_eta_update_done = done

        if done <= 0 or total_work <= 0 or not did_work:
            if done <= 0:
                self.eta_var.set("ETA: estimating…")
            else:
                self.eta_var.set("ETA: waiting for real processing…")
            return

        # Throughput based on work completed since last checkpoint
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
