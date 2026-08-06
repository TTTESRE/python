#!/usr/bin/env python3
"""
dashboard.py
Simple live dashboard for the OpenCL PyTorch backend.
"""

from dataclasses import dataclass
from threading import Thread, Event
import time
import os


@dataclass
class Stats:
    forwards: int = 0
    backwards: int = 0

    samples: int = 0
    env_steps: int = 0

    opencl_calls: int = 0
    aten_calls: int = 0
    fallback_count: int = 0

    kernel_time: float = 0.0

    episodes: int = 0
    reward: float = 0.0
    best_reward: float = float("-inf")

    threshold: int = 8192


class Dashboard:
    def __init__(self, stats: Stats):
        self.stats = stats
        self.stop_event = Event()

    def start(self):
        Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.stop_event.set()

    def _clear(self):
        os.system("clear")

    def _run(self):
        start = time.perf_counter()

        last_forwards = 0
        last_backwards = 0
        last_samples = 0
        last_steps = 0

        while not self.stop_event.is_set():

            now = time.perf_counter()
            elapsed = now - start

            dt = 1.0

            fps = self.stats.forwards - last_forwards
            bps = self.stats.backwards - last_backwards
            sps = self.stats.samples - last_samples
            esps = self.stats.env_steps - last_steps

            last_forwards = self.stats.forwards
            last_backwards = self.stats.backwards
            last_samples = self.stats.samples
            last_steps = self.stats.env_steps

            total = self.stats.opencl_calls + self.stats.aten_calls

            util = (
                self.stats.opencl_calls / total * 100
                if total else 0.0
            )

            avg_kernel = (
                self.stats.kernel_time /
                self.stats.opencl_calls * 1000
                if self.stats.opencl_calls else 0
            )

            self._clear()

            print("=" * 64)
            print("        OpenCL PyTorch Backend Dashboard")
            print("=" * 64)

            print(f"Elapsed:              {elapsed:8.1f} sec")
            print()

            print("Training")
            print("-" * 64)
            print(f"Episodes:             {self.stats.episodes}")
            print(f"Reward:               {self.stats.reward:8.2f}")
            print(f"Best Reward:          {self.stats.best_reward:8.2f}")

            print()

            print("Performance")
            print("-" * 64)
            print(f"Forward/sec:          {fps}")
            print(f"Backward/sec:         {bps}")
            print(f"Samples/sec:          {sps}")
            print(f"Env Steps/sec:        {esps}")

            print()

            print("Backend")
            print("-" * 64)
            print(f"OpenCL Calls:         {self.stats.opencl_calls}")
            print(f"ATen Calls:           {self.stats.aten_calls}")
            print(f"Fallback Count:       {self.stats.fallback_count}")
            print(f"OpenCL Utilization:   {util:7.2f}%")

            print()

            print("Kernel")
            print("-" * 64)
            print(f"Average Kernel:       {avg_kernel:.3f} ms")

            print()

            print("Configuration")
            print("-" * 64)
            print(f"Dispatch Threshold:   {self.stats.threshold}")

            print("=" * 64)

            time.sleep(dt)


if __name__ == "__main__":

    stats = Stats()

    dash = Dashboard(stats)
    dash.start()

    # Demo
    try:
        while True:
            time.sleep(0.05)

            stats.forwards += 28
            stats.backwards += 28

            stats.samples += 960
            stats.env_steps += 180

            stats.opencl_calls += 35
            stats.aten_calls += 1

            stats.fallback_count = stats.aten_calls

            stats.kernel_time += 0.00021 * 35

            stats.reward += 0.05
            stats.best_reward = max(stats.best_reward, stats.reward)

            stats.episodes += 1

    except KeyboardInterrupt:
        dash.stop()
