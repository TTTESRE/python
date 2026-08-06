#!/usr/bin/env python3
"""
dashboard.py
Simple live dashboard for the OpenCL PyTorch backend.
"""

from dataclasses import dataclass
from threading import Thread, Event
import time
import os
import psutil


@dataclass
class Stats:
    episodes: int = 0
    reward: float = 0.0
    best_reward: float = float("-inf")
    avg50: float = 0.0

    forwards: int = 0
    backwards: int = 0
    samples: int = 0
    env_steps: int = 0

    opencl_calls: int = 0
    aten_calls: int = 0
    fallback_count: int = 0

    kernel_time: float = 0.0

    opencl_enabled: bool = False
    compute_chain: str = "opencl,aten"
    threshold: int = 8192


class Dashboard:
    def __init__(self, stats: Stats, fps: int = 8):
        self.stats = stats
        self.fps = fps
        self.stop_event = Event()
        self.start_time = time.perf_counter()

    def start(self):
        Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.stop_event.set()

    def _clear(self):
        os.system("clear")

    def _run(self):
        last_forwards = 0
        last_backwards = 0
        last_samples = 0
        last_steps = 0
        last_time = time.perf_counter()

        while not self.stop_event.is_set():
            now = time.perf_counter()
            elapsed = now - self.start_time
            dt = now - last_time

            fps = (self.stats.forwards - last_forwards) / dt if dt > 0 else 0
            bps = (self.stats.backwards - last_backwards) / dt if dt > 0 else 0
            sps = (self.stats.samples - last_samples) / dt if dt > 0 else 0
            esps = (self.stats.env_steps - last_steps) / dt if dt > 0 else 0

            last_forwards = self.stats.forwards
            last_backwards = self.stats.backwards
            last_samples = self.stats.samples
            last_steps = self.stats.env_steps
            last_time = now

            total = self.stats.opencl_calls + self.stats.aten_calls
            util = (self.stats.opencl_calls / total * 100) if total else 0.0

            avg_kernel = (
                self.stats.kernel_time / self.stats.opencl_calls * 1000
                if self.stats.opencl_calls else 0
            )

            try:
                mem = psutil.virtual_memory()
                ram_usage = f"{mem.percent:.1f}%"
            except Exception:
                ram_usage = "N/A"

            try:
                cpu_usage = f"{psutil.cpu_percent(interval=None):.1f}%"
            except Exception:
                cpu_usage = "N/A"

            self._clear()

            print("=" * 64)
            print("        OpenCL PyTorch Backend Dashboard")
            print("=" * 64)

            print(f"Elapsed:              {elapsed:8.1f} sec")
            print()

            print("Training")
            print("-" * 64)
            print(f"Episode:              {self.stats.episodes}")
            print(f"Reward:               {self.stats.reward:8.2f}")
            print(f"Avg50:                {self.stats.avg50:8.2f}")
            print(f"Best Reward:          {self.stats.best_reward:8.2f}")

            print()

            print("Performance")
            print("-" * 64)
            print(f"Forward passes/sec:   {fps:8.1f}")
            print(f"Backward passes/sec:  {bps:8.1f}")
            print(f"Samples/sec:          {sps:8.1f}")
            print(f"Env steps/sec:        {esps:8.1f}")

            print()

            print("OpenCL")
            print("-" * 64)
            print(f"OpenCL enabled:       {self.stats.opencl_enabled}")
            print(f"Compute chain:        {self.stats.compute_chain}")
            print(f"Dispatch threshold:   {self.stats.threshold}")
            print(f"OpenCL utilization:   {util:7.2f}%")
            print(f"OpenCL calls:         {self.stats.opencl_calls}")
            print(f"ATen calls:           {self.stats.aten_calls}")
            print(f"Fallback count:       {self.stats.fallback_count}")

            print()

            print("Kernel")
            print("-" * 64)
            print(f"Average kernel time:  {avg_kernel:.3f} ms")
            print(f"Total kernel time:    {self.stats.kernel_time:.3f} s")

            print()

            print("System")
            print("-" * 64)
            print(f"RAM usage:            {ram_usage}")
            print(f"CPU usage:            {cpu_usage}")
            print(f"Elapsed time:         {elapsed:.1f} s")

            print("=" * 64)

            time.sleep(1.0 / self.fps)
